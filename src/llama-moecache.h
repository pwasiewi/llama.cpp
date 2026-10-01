#pragma once

// GPU-resident LRU cache for MoE expert weights that -ot pinned to host memory.
//
// Motivation (measured on Qwen3.8-Flash-Next, 512 experts / 10 routed): expert
// routing has strong temporal locality (LRU-64 hit rate ~67% over a mixed
// workload) even though the long-run distribution is near-uniform. Decode on a
// host-offloaded MoE layer is bound by host RAM bandwidth, so serving the hot
// experts from VRAM removes most of the per-token DIMM traffic.
//
// Mechanism (no custom kernels):
//  - per cached layer, companion tensors up_c/gate_c/down_c of shape
//    [ne0, ne1, n_slots+1] live in the device buffer of that layer's router;
//    slot n_slots is permanently zero (the "dummy" slot).
//  - an I32 table[512] maps expert id -> slot, or n_slots when uncached.
//    One copy on device (read by get_rows to remap ids for the cache-side
//    mul_mat_id chain) and one on host (read by the CPU mul_mat_id via
//    src[3] to SKIP cached ids, zeroing their dst rows).
//  - the two down-projection outputs are summed; uncached ids contribute 0
//    through the cache chain (zero slot) and cached ids contribute 0 through
//    the CPU chain (skip), so the result is exact.
//  - llama_moe_cache_step(), called at the end of llama_context::decode(),
//    performs throttled LRU updates: at most LLAMA_MOE_CACHE_INSERTS expert
//    uploads per layer per step via ggml_backend_tensor_set.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).

#include <cstdint>

// Batch-size gate for the cache graph: decode runs either one token at a time
// or a speculative-decoding verification batch of (n_draft + 1) tokens. Prompt
// processing and other large batches are left to the stock graph (routing is
// flat there and would only thrash the cache).
//
// The upper limit is not arbitrary. Cached and uncached experts share one slot
// id space, so every uncached expert of a token maps to the SAME dummy slot and
// a token's slot-id row contains duplicates. The batched CUDA mul_mat_id kernels
// (MMQ / MM*_switch_ids / the host sorting fallback) assume a token's ids are
// distinct and collapse duplicates, leaving uninitialised entries behind - which
// shows up as an out-of-bounds write and "an illegal memory access was
// encountered". Only the vector kernels (mmvq/mmvf) are duplicate-safe, so the
// gate must stay inside their window: MMVQ_MAX_BATCH_SIZE (8) and, per type,
// get_mmvq_mmid_max_batch() - e.g. 6 for IQ3_S and 7 for IQ3_XXS/IQ2_S on
// sm_86. Non-quantized (F16/BF16) experts never take the vector path on NVIDIA
// and need a max batch of 1.
int llama_moe_cache_max_batch();

// a step could not use the cache because the batch was wider than
// llama_moe_cache_max_batch() (e.g. speculative decoding drafts): counted so the
// teardown telemetry can explain a low hit rate instead of leaving it a mystery
void llama_moe_cache_note_wide_batch();

struct llama_model;
struct ggml_tensor;

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0;

    // host-resident source weights (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // device-resident cache slots, ne[2] == n_slots + 1 (last slot all zeros)
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;
};

// build the cache for every host-resident expert layer of the model.
// Safe to call more than once; only the first call does work.
void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, int32_t max_batch,
        int32_t half_life, int32_t admit);

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps);

struct ggml_backend_sched;

// finish scheduled graph execution, then publish throttled LRU updates
void llama_moe_cache_step(ggml_backend_sched * sched);

// log cumulative cache statistics (safe to call at any time; no-op when disabled)
void llama_moe_cache_print_stats();
