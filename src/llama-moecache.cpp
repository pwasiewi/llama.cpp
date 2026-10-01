#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)

    std::vector<bool>     slot_in_flight; // slot has an upload pending

    // windowed use counter (halved every half_life steps) used to gate
    // admission: an expert must be seen a few times *recently* before it is
    // worth a PCIe upload, so a one-off cannot displace a hot expert
    std::vector<uint8_t>  score;

    // host_table has entries that dev_table does not have yet; flushed as ONE
    // device write at the end of step() (see flush_tables)
    bool table_dirty = false;

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

struct moe_cache {
    int32_t n_slots     = 0;
    int32_t max_inserts = 2;
    int32_t max_batch   = 1;
    int32_t half_life   = 0;  // steps between halvings of the use counter (0 = all time)
    int32_t admit       = 1;  // sightings required inside the window before an upload

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    // telemetry: the cache's cost is the bytes it pulls over PCIe, its benefit is
    // the accesses it serves from VRAM, so track both sides
    uint64_t n_uploads   = 0;
    uint64_t n_evictions = 0;
    uint64_t bytes_uploaded = 0;

    // totals at the previous report, so the line shows the *interval* rate as
    // well as the cumulative one: a long-lived server flattens out, and a
    // cumulative average hides both the warm-up and any later degradation
    uint64_t rp_hits = 0, rp_misses = 0, rp_uploads = 0, rp_evictions = 0;

    // steps where the cache existed for this layer but could not be used (a
    // speculative batch wider than max_batch): a hit rate is misleading without
    // knowing how often the cache never got a chance
    uint64_t n_skip_wide_batch = 0;

    // table maintenance: device table writes are batched per layer per step
    // (one blocking copy per dirty layer instead of one per changed entry)
    uint64_t n_table_entries = 0; // entry changes (what the unbatched code would have copied one by one)
    uint64_t n_table_flushes = 0; // device copies actually issued

    // LLAMA_MOE_CACHE_VERIFY=<level>: 1 = read back every device table after each
    // step and compare with the host shadow; 2 = additionally read back every
    // freshly published slot and compare with the host expert bytes. Any
    // mismatch is logged and aborts the process. Zero cost when off.
    int      verify           = 0;
    uint64_t n_verify_tables  = 0;
    uint64_t n_verify_slots   = 0;

    const llama_model * model = nullptr; // whose tensors the table entries name

    // optional routing trace (LLAMA_MOE_CACHE_TRACE=<path>): one line per layer
    // per graph execution, "<step> <layer> <n_tokens> <ids...>", used offline to
    // evaluate eviction policies against real routing
    FILE *     trace     = nullptr;
    std::mutex trace_mtx;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;
};

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    // routing trace: written for every batch, including the ones the cache graph
    // is not built for, so the trace is a faithful record of decoder routing
    if (mc->trace) {
        std::lock_guard<std::mutex> tlock(mc->trace_mtx);
        fprintf(mc->trace, "%" PRIu64 " %d %" PRId64, mc->n_steps, il, n_tokens);
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t i = 0; i < n_ids; ++i) {
                const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                fprintf(mc->trace, " %d", id);
            }
        }
        fputc('\n', mc->trace);
    }

    if (n_tokens > mc->max_batch) {
        return; // the cache graph is not built for this batch size: don't pollute the LRU
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            if (ls->score[id] < 255) {
                ls->score[id]++;
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss++;
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
    }
}

void upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
}

// Record a table change on the host shadow only. The device table is brought
// up to date by flush_tables() before step() returns; step() runs between two
// graph executions (after the scheduler fence), so no graph can observe the
// device table in between and the deferred write is indistinguishable from the
// immediate one.
void set_table_entry(moe_cache * mc, layer_state & ls, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(ls.pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    ls.table_dirty = true;
    mc->n_table_entries++;
}

// One device write per dirty layer: the whole host shadow (n_expert x 4 bytes,
// 1-2 KB) replaces the per-entry 4-byte blocking copies. The device table ends
// up equal to the host shadow, which is exactly the state the per-entry writes
// would have produced (both received the same sequence of values).
void flush_tables(moe_cache * mc) {
    for (auto & ls : mc->layers) {
        if (!ls.table_dirty) {
            continue;
        }
        ggml_backend_tensor_set(ls.pub.dev_table, ls.pub.host_table->data, 0, ggml_nbytes(ls.pub.host_table));
        ls.table_dirty = false;
        mc->n_table_flushes++;
    }
}

// LLAMA_MOE_CACHE_VERIFY>=1: the device tables must equal the host shadows
// after every step. A mismatch means a lost or torn table write - the exact
// failure the batched flush could introduce - so abort instead of serving
// silently wrong experts.
void verify_tables(moe_cache * mc) {
    std::vector<int32_t> dev;
    for (auto & ls : mc->layers) {
        const size_t nb = ggml_nbytes(ls.pub.host_table);
        dev.resize(nb/sizeof(int32_t));
        ggml_backend_tensor_get(ls.pub.dev_table, dev.data(), 0, nb);
        const int32_t * host = (const int32_t *) ls.pub.host_table->data;
        for (size_t e = 0; e < dev.size(); ++e) {
            if (dev[e] != host[e]) {
                LLAMA_LOG_ERROR("moe-cache: VERIFY FAILED at step %" PRIu64 ": layer %d expert %zu: device table=%d host table=%d "
                        "(expert_slot=%d, n_slots=%d)\n",
                        mc->n_steps, ls.pub.il, e, dev[e], host[e], ls.expert_slot[e], mc->n_slots);
                GGML_ABORT("moe-cache: device/host table mismatch");
            }
        }
        mc->n_verify_tables++;
    }
}

// LLAMA_MOE_CACHE_VERIFY>=2: a freshly published slot must hold exactly the
// host expert's bytes in all three tensors (catches a bad upload before the
// prefill D2D path or the cache chain would serve it).
void verify_slot(moe_cache * mc, layer_state & ls, int32_t expert, int32_t slot) {
    const ggml_tensor * dst[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
    const ggml_tensor * src[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
    std::vector<uint8_t> buf;
    for (int k = 0; k < 3; ++k) {
        const size_t sz = src[k]->nb[2];
        buf.resize(sz);
        ggml_backend_tensor_get(dst[k], buf.data(), (size_t) slot*dst[k]->nb[2], sz);
        const uint8_t * ref = (const uint8_t *) src[k]->data + (size_t) expert*sz;
        if (memcmp(buf.data(), ref, sz) != 0) {
            size_t off = 0;
            while (off < sz && buf[off] == ref[off]) { off++; }
            LLAMA_LOG_ERROR("moe-cache: VERIFY FAILED at step %" PRIu64 ": layer %d slot %d expert %d tensor %s: "
                    "device bytes differ from host at offset %zu of %zu (dev=0x%02x host=0x%02x)\n",
                    mc->n_steps, ls.pub.il, slot, expert, dst[k]->name, off, sz, buf[off], ref[off]);
            GGML_ABORT("moe-cache: slot bytes differ from host expert");
        }
    }
    mc->n_verify_slots++;
}

} // namespace

// Observability: a mis-sized or bypassed cache is otherwise a silent no-op, so
// every decision point reports the VRAM headroom it saw and the bytes it needed.
static size_t moe_cache_free_vram() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        return 0;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    return props.memory_free;
}

void llama_moe_cache_note_wide_batch() {
    if (g_cache) {
        g_cache->n_skip_wide_batch++;
    }
}

// Full teardown of the singleton. Needed when a long-lived process replaces the
// model: the table entries name the old model's tensors, so lookups would never
// match again and the cache would silently stop applying.
static void llama_moe_cache_free() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        mc->stop = true;
    }
    mc->wcv.notify_all();
    if (mc->worker.joinable()) {
        mc->worker.join();
    }
    if (mc->trace) {
        fclose(mc->trace);
    }
    ggml_set_moe_obs_callback(nullptr, nullptr);
    for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
    for (auto * c : mc->ctxs) { ggml_free(c); }
    delete mc;
    g_cache     = nullptr;
    g_init_done = false;
}

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, int32_t max_batch,
        int32_t half_life, int32_t admit) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);

    if (n_slots <= 0 && g_cache) {
        // a context that does not request the cache (e.g. a speculative draft
        // context on its own model) must leave an existing cache in place:
        // freeing it here would silently disable the cache for the model that
        // does use it, with no error anywhere
        LLAMA_LOG_DEBUG("%s: no MoE expert cache requested, keeping the existing one\n", __func__);
        return;
    }

    if (g_cache && g_cache->model != &model) {
        // long-lived servers (e.g. --sleep-idle-seconds) load a new model into the
        // same process; rebuild the cache for it instead of silently going dead
        LLAMA_LOG_WARN("%s: MoE expert cache belongs to a previously loaded model - reinitialising\n", __func__);
        llama_moe_cache_free();
    }

    if (g_init_done) {
        LLAMA_LOG_DEBUG("%s: MoE expert cache already initialised for this process (slots=%d)\n", __func__, n_slots);
        return;
    }
    [&]() {
        if (n_slots <= 0) {
            // do not latch here: a context created without the cache options must
            // not disable the cache for the context that does request it
            LLAMA_LOG_DEBUG("%s: no MoE expert cache requested (slots=%d)\n", __func__, n_slots);
            return;
        }

        const size_t free_vram_before = moe_cache_free_vram();

        auto * mc = new moe_cache();
        mc->model = &model;
        mc->n_slots = n_slots;
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }
        mc->max_batch = max_batch > 0 ? max_batch : 1;
        mc->half_life = half_life > 0 ? half_life : 0;
        mc->admit     = admit > 0 ? admit : 1;

        // collect the host-resident expert layers; the cache itself is allocated
        // on the primary GPU device (it is a device-side copy of host weights,
        // so it does not belong to any particular layer's tensor placement and
        // must not depend on where the router happens to land - routers are
        // often bf16 and therefore host-resident on pre-Ampere hardware)
        struct cand { int il; const llama_layer * l; };
        std::vector<cand> cands;

        size_t n_expert_layers = 0;
        size_t n_gpu_experts   = 0;
        bool   weights_loaded  = false;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps) {
                continue;
            }
            n_expert_layers++;
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            weights_loaded = true;
            if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                n_gpu_experts++;
                continue; // experts already on a device: nothing to cache
            }
            cands.push_back({(int) il, &l});
        }

        if (cands.empty()) {
            if (weights_loaded) {
                LLAMA_LOG_WARN("%s: %d MoE expert cache slots requested but there is nothing to cache: "
                        "%zu of %zu expert layers are already on the GPU (placed there by --fit). "
                        "Cache disabled. Use --cpu-moe to keep experts in host memory, or lower --fit-target so "
                        "--fit stops promoting expert layers.\n",
                        __func__, n_slots, n_gpu_experts, n_expert_layers);
            } else {
                // the --fit memory probe creates a context from a weights-less model
                LLAMA_LOG_DEBUG("%s: no loaded expert weights (memory estimation model) - cache not initialised\n", __func__);
            }
            delete mc;
            return;
        }

        // Non-quantized experts (F16/BF16/F32) never take the duplicate-safe
        // vector kernel on NVIDIA, so a batch larger than one would hand them to
        // the batched kernels that assume distinct ids - limit to one token.
        for (const auto & c : cands) {
            if (!ggml_is_quantized(c.l->ffn_up_exps->type)   ||
                !ggml_is_quantized(c.l->ffn_gate_exps->type) ||
                !ggml_is_quantized(c.l->ffn_down_exps->type)) {
                if (mc->max_batch > 1) {
                    LLAMA_LOG_WARN("%s: non-quantized experts in layer %d - MoE expert cache "
                            "batch limited to 1 token\n", __func__, c.il);
                }
                mc->max_batch = 1;
                break;
            }
        }

        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) {
            LLAMA_LOG_WARN("%s: no GPU device found for the MoE expert cache - disabled\n", __func__);
            delete mc;
            return;
        }
        ggml_backend_buffer_type_t buft_dev = ggml_backend_dev_buffer_type(dev);

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            size_t buft_bytes = 0; // device bytes requested (for the warning below)

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots  = n_slots;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], n_slots + 1);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], n_slots + 1);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], n_slots + 1);
                    ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                    ggml_format_name(ls->pub.up_c,      "moe_cache_up.%d",   c.il);
                    ggml_format_name(ls->pub.gate_c,    "moe_cache_gate.%d", c.il);
                    ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
                    buft_bytes += ggml_nbytes(ls->pub.up_c) + ggml_nbytes(ls->pub.gate_c) + ggml_nbytes(ls->pub.down_c);
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                const size_t n_lay     = mc->layers.empty() ? 1 : mc->layers.size();
                const size_t per_slot  = buft_bytes/(size_t) n_slots;  // VRAM one slot adds across all cached layers
                const size_t free_now  = moe_cache_free_vram();
                const size_t deficit   = buft_bytes > free_now ? buft_bytes - free_now : 0;
                const size_t max_slots = per_slot ? free_now/per_slot : 0;
                LLAMA_LOG_ERROR("%s: FAILED to allocate the MoE expert cache: %zu MiB needed "
                        "(%.1f MiB per slot across %zu layers = %.2f MiB per expert x %d slots), "
                        "only %zu MiB of VRAM free on %s - short by %zu MiB. "
                        "The cache is DISABLED for the life of this process (throughput will be baseline and the "
                        "--fit-target space reserved for it is wasted). "
                        "Fix: raise --fit-target by at least %zu MiB (to %zu or more), or lower --moe-expert-cache to %zu.\n",
                        __func__, buft_bytes/1024/1024, per_slot/1024.0/1024.0, n_lay,
                        buft_bytes/(double)(n_slots * n_lay)/1024.0/1024.0, n_slots,
                        free_now/1024/1024, ggml_backend_buft_name(buft), deficit/1024/1024,
                        deficit/1024/1024 + 256, deficit/1024/1024 + 256 + 2700, max_slots);
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            mc->bufs.push_back(buf);
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), /*tables_only=*/true);
        if (ok) {
            ok = alloc_group(buft_dev, /*tables_only=*/false);
        }

        if (!ok) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
        }

        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            ls.slot_expert.assign(n_slots, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(n_slots, 0);
            ls.slot_in_flight.assign(n_slots, false);
            ls.score.assign(n_expert, 0);

            std::vector<int32_t> dummy(n_expert, n_slots);
            ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert, experts in %s\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2],
                    ls.pub.up_src->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(ls.pub.up_src->buffer)) : "no buffer");
        }

        if (const char * trace_path = getenv("LLAMA_MOE_CACHE_TRACE")) {
            if (trace_path[0]) {
                mc->trace = fopen(trace_path, "w");
                if (mc->trace) {
                    fprintf(mc->trace, "# llama moe cache routing trace v1\n");
                    fprintf(mc->trace, "# step layer n_tokens expert_ids...\n");
                    LLAMA_LOG_INFO("%s: MoE expert cache routing trace -> %s\n", __func__, trace_path);
                } else {
                    LLAMA_LOG_WARN("%s: could not open trace file '%s'\n", __func__, trace_path);
                }
            }
        }

        if (const char * v = getenv("LLAMA_MOE_CACHE_VERIFY")) {
            mc->verify = atoi(v);
            if (mc->verify > 0) {
                LLAMA_LOG_INFO("%s: MoE expert cache VERIFY level %d: device tables are read back and compared with the host "
                        "shadow after every step%s; a mismatch aborts the process\n", __func__, mc->verify,
                        mc->verify >= 2 ? ", every published slot is compared with the host expert bytes" : "");
            }
        }

        mc->worker = std::thread([mc]() {
            for (;;) {
                upload_job j;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                    if (mc->stop) {
                        return;
                    }
                    j = mc->todo.front();
                    mc->todo.pop_front();
                }
                auto & ls = mc->layers[j.layer_idx];
                upload_slice(ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
                upload_slice(ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
                upload_slice(ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
                {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    j.done = true;
                    mc->done.push_back(j);
                }
            }
        });

        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        g_cache = mc;
        g_init_done = true;

        // per-slot figures: the honest unit for sizing --fit-target is the VRAM one
        // slot adds across ALL cached layers (K x that = the cache's total), which is
        // what the old "K slots x 80 MiB" arithmetic was really expressing
        // where the cached layers' experts actually live: a miss on a CPU_Mapped
        // (disk-backed mmap) layer costs a disk read, a miss on CUDA_Host costs a
        // pinned-RAM read, so the two populations are not equally worth a slot
        size_t n_src_pinned = 0, n_src_mapped = 0, n_src_other = 0;
        for (const auto & ls : mc->layers) {
            const char * nm = ls.pub.up_src->buffer
                            ? ggml_backend_buft_name(ggml_backend_buffer_get_type(ls.pub.up_src->buffer)) : "?";
            if      (strstr(nm, "CUDA_Host")) n_src_pinned++;
            else if (strstr(nm, "Mapped"))    n_src_mapped++;
            else                              n_src_other++;
        }

        const double per_slot_all = vram/(double) n_slots/1024.0/1024.0;
        const double per_expert   = vram/(double) (n_slots * mc->layers.size())/1024.0/1024.0;
        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu of %zu expert layers x %d slots "
                "(%.1f MiB per slot across the cached layers = %.2f MiB per expert x %zu layers), "
                "%d inserts/step, %.1f MiB device memory, window=%d admit=%d, free VRAM %zu -> %zu MiB "
                "| sources: %zu pinned (CUDA_Host), %zu disk-mapped (CPU_Mapped), %zu other\n",
                __func__, mc->layers.size(), n_expert_layers, n_slots,
                per_slot_all, per_expert, mc->layers.size(),
                mc->max_inserts, vram/1024.0/1024.0,
                mc->half_life, mc->admit,
                free_vram_before/1024/1024, moe_cache_free_vram()/1024/1024,
                n_src_pinned, n_src_mapped, n_src_other);
    }();
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

void llama_moe_cache_step(ggml_backend_sched * sched) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    // Published mappings and retired slots must outlive every graph reader.
    // The host-side table writes below are not ordered against the graph that
    // was just submitted, so wait for that graph to finish before publishing.
    // (llama_context::decode leaves its own synchronize() commented out.)
    ggml_backend_sched_synchronize(sched);

    // 1) publish completed uploads (sync point: no graph is executing)
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & j : mc->done) {
            auto & ls = mc->layers[j.layer_idx];
            if (mc->verify >= 2) {
                verify_slot(mc, ls, j.expert, j.slot); // before the mapping is published
            }
            ls.slot_expert[j.slot]     = j.expert;
            ls.expert_slot[j.expert]   = j.slot;
            ls.slot_last_use[j.slot]   = ++mc->clock;
            ls.slot_in_flight[j.slot]  = false;
            set_table_entry(mc, ls, j.expert, j.slot);
        }
        mc->done.clear();
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // age the windowed use counters (integer halving = exponential decay)
    if (mc->half_life > 0 && mc->n_steps % mc->half_life == 0) {
        for (auto & ls : mc->layers) {
            for (auto & s : ls.score) {
                s >>= 1;
            }
        }
    }

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int budget = mc->max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it, --budget) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0) {
                continue;
            }
            if (mc->admit > 1 && ls.score[id] < mc->admit) {
                continue; // not used often enough recently to be worth an upload
            }

            // victim: an empty non-in-flight slot if any, else the LRU non-in-flight slot
            int32_t slot = -1;
            uint64_t best = UINT64_MAX;
            for (int32_t s = 0; s < mc->n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(mc, ls, victim, mc->n_slots);
                mc->n_evictions++;
            }
            ls.slot_in_flight[slot] = true;

            mc->n_uploads++;
            mc->bytes_uploaded += ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            mc->todo.push_back({li, id, slot});
        }
        ls.pending.clear();
    }
    mc->wcv.notify_one();

    // 3) publish the table changes of this step to the device: one write per
    //    dirty layer. Still inside the fence window (no graph submitted yet).
    flush_tables(mc);
    if (mc->verify >= 1) {
        verify_tables(mc);
    }

    // at --log-verbosity 4 (trace) DEBUG is suppressed, so report at INFO on step 1
    // (existence) and step 32 (a warmed hit rate), then every 256 steps - which is
    // roughly per request at server token rates - so a long-lived server keeps
    // showing whether the cache is engaging instead of going quiet after startup
    const bool info_report = mc->n_steps == 1 || mc->n_steps == 32 || mc->n_steps % 256 == 0;
    if (info_report || mc->n_steps % 64 == 0) {
        uint64_t h = 0, m = 0, served = 0;
        for (auto & ls : mc->layers) {
            h += ls.n_hit;
            m += ls.n_miss;
            served += ls.n_hit * (ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2]);
        }
        // the INFO lines carry free VRAM too: a squeezed card pages the model buffer
        // silently (no OOM, no error) and the only symptom is collapsing throughput
        const size_t free_now = moe_cache_free_vram();
        const uint64_t dh = h - mc->rp_hits, dm = m - mc->rp_misses;
        const uint64_t du = mc->n_uploads - mc->rp_uploads, de = mc->n_evictions - mc->rp_evictions;
        llama_log_internal(info_report ? GGML_LOG_LEVEL_INFO : GGML_LOG_LEVEL_DEBUG,
                "moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%% "
                "uploads=%" PRIu64 " evictions=%" PRIu64 " yield=%.1fx skipped-wide-batch=%" PRIu64
                " free-vram=%zu MiB | window: hit-rate=%.1f%% uploads=%" PRIu64 " evictions=%" PRIu64 "\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0,
                mc->n_uploads, mc->n_evictions,
                mc->bytes_uploaded ? (double) served/(double) mc->bytes_uploaded : 0.0,
                mc->n_skip_wide_batch, free_now/1024/1024,
                dh + dm ? 100.0*dh/(dh + dm) : 0.0, du, de);
        mc->rp_hits = h; mc->rp_misses = m;
        mc->rp_uploads = mc->n_uploads; mc->rp_evictions = mc->n_evictions;

        if (info_report && free_now < (size_t) 128*1024*1024) {
            LLAMA_LOG_WARN("%s: only %zu MiB of VRAM free with the MoE expert cache resident - Windows may already "
                    "be paging the model buffer, which collapses throughput with no other symptom. Raise "
                    "--fit-target (fewer layers promoted) or lower --moe-expert-cache.\n", __func__, free_now/1024/1024);
        }
    }
}

int llama_moe_cache_max_batch() {
    return g_cache ? g_cache->max_batch : 1;
}

void llama_moe_cache_print_stats() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    uint64_t h = 0, m = 0, cached = 0, total = 0, bytes_served = 0;
    {
        std::lock_guard<std::mutex> lock(g_cache->mtx);
        for (auto & ls : mc->layers) {
            h += ls.n_hit;
            m += ls.n_miss;
            bytes_served += ls.n_hit * (ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2]);
            for (int32_t e : ls.expert_slot) {
                cached += e >= 0;
            }
            total += ls.expert_slot.size();
        }
    }

    LLAMA_LOG_INFO("%s: MoE expert cache: steps=%" PRIu64 " layers=%zu slots=%d "
            "hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%% resident=%" PRIu64 "/%" PRIu64
            " window=%d admit=%d\n",
            __func__, mc->n_steps, mc->layers.size(), mc->n_slots,
            h, m, (h + m) ? 100.0*h/(h + m) : 0.0, cached, total, mc->half_life, mc->admit);
    LLAMA_LOG_INFO("%s: MoE expert cache traffic: uploads=%" PRIu64 " evictions=%" PRIu64
            " uploaded=%.1f MiB served=%.1f MiB yield=%.1fx (bytes served from VRAM per byte pulled over PCIe)\n",
            __func__, mc->n_uploads, mc->n_evictions,
            mc->bytes_uploaded/1024.0/1024.0, bytes_served/1024.0/1024.0,
            mc->bytes_uploaded ? (double) bytes_served/(double) mc->bytes_uploaded : 0.0);
    LLAMA_LOG_INFO("%s: MoE expert cache tables: entry-changes=%" PRIu64 " device-writes=%" PRIu64
            " (batched per layer per step)%s\n",
            __func__, mc->n_table_entries, mc->n_table_flushes,
            mc->verify > 0 ? " | VERIFY passed" : "");
    if (mc->verify > 0) {
        LLAMA_LOG_INFO("%s: MoE expert cache VERIFY level %d: table-checks=%" PRIu64 " slot-checks=%" PRIu64 " mismatches=0\n",
                __func__, mc->verify, mc->n_verify_tables, mc->n_verify_slots);
    }
    if (mc->n_skip_wide_batch) {
        LLAMA_LOG_INFO("%s: MoE expert cache bypassed on %" PRIu64 " steps: batch wider than --moe-expert-cache-max-batch (%d), "
                "e.g. speculative decoding drafts. The cache cannot be used for those tokens.\n",
                __func__, mc->n_skip_wide_batch, mc->max_batch);
    }
}
