#include "../include/model.hpp"
#include "hip/BLAS.hip"
#include "hip/attention.hip"
#include "hip/embed.hip"
#include "hip/gemms/gemm_logits.hip"
#include "hip/gemms/gemm_o.hip"
#include "hip/gemms/gemm_qkv.hip"
#include "hip/gemms/gemm_router.hip"
#include "hip/moe.hip"
#include "hip/rmsnorm.hip"
#include "hip/rope.hip"
#include "hip/scheduler.hip"
#include "hip/softmax.hip"
#include "hip/split_qkv.hip"
#include "hip/topk.hip"
#include "hip/vecadd.hip"
#include "profiler.cpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <omp.h>
#include <vector>

#ifndef ADAPT_MT_M
#define ADAPT_MT_M 64 // CTA tile on M (rows)
#endif
#ifndef ADAPT_MT_N
#define ADAPT_MT_N 32 // CTA tile on N (cols); rows_hint is a multiple of this
#endif
#ifndef ADAPT_CTAS_PER_SM_CAP
#define ADAPT_CTAS_PER_SM_CAP 2
#endif
#ifndef ADAPT_VERY_SPARSE_CUTOFF
#define ADAPT_VERY_SPARSE_CUTOFF 6 // if <= this many experts are active, treat all heavy
#endif
#ifndef ADAPT_HEAVY_FACTOR_SPARSE
#define ADAPT_HEAVY_FACTOR_SPARSE 1.5f // heavy if >= 1.5x mean when 7..12 active experts
#endif
#ifndef ADAPT_HEAVY_FACTOR_DEFAULT
#define ADAPT_HEAVY_FACTOR_DEFAULT 2.0f // heavy if >= 2x mean otherwise
#endif
#ifndef ADAPT_RH_MIN_TILES
#define ADAPT_RH_MIN_TILES 1
#endif
#ifndef ADAPT_RH_MAX_TILES
#define ADAPT_RH_MAX_TILES 32
#endif
#ifndef ADAPT_DEBUG
#define ADAPT_DEBUG 0
#endif

static int kGridCap = [] {
    hipDeviceProp_t prop{};
    int dev = 0;
    CHECK_HIP(hipGetDevice(&dev));
    CHECK_HIP(hipGetDeviceProperties(&prop, dev));
    return prop.multiProcessorCount * 2;
}();

// --- Stream scaffolding for async execution and overlap ---
namespace {
hipStream_t s_compute = nullptr;   // main compute stream
hipStream_t s_h2d = nullptr;       // host->device copies
hipStream_t s_d2h = nullptr;       // device->host copies
hipStream_t s_moe_heavy = nullptr; // MoE heavy bucket
hipStream_t s_moe_light = nullptr; // MoE light bucket
hipEvent_t evt_h2d_ready, evt_moe_heavy_done, evt_moe_light_done;
hipEvent_t evt_meta_ready, evt_meta_done;
hipEvent_t evt_offsets_ready, evt_offsets_done;
hipEvent_t evt_wq_heavy_ready, evt_wq_light_ready;

void create_streams_once() {
    if (s_compute)
        return;
    CHECK_HIP(hipStreamCreateWithFlags(&s_compute, hipStreamNonBlocking));
    CHECK_HIP(hipStreamCreateWithFlags(&s_h2d, hipStreamNonBlocking));
    CHECK_HIP(hipStreamCreateWithFlags(&s_d2h, hipStreamNonBlocking));

    int least_priority = 0, greatest_priority = 0;
    CHECK_HIP(hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    CHECK_HIP(hipStreamCreateWithPriority(&s_moe_heavy, hipStreamNonBlocking, least_priority));
    CHECK_HIP(hipStreamCreateWithPriority(&s_moe_light, hipStreamNonBlocking, greatest_priority));
    CHECK_HIP(hipEventCreateWithFlags(&evt_h2d_ready, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&evt_moe_heavy_done, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&evt_moe_light_done, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&evt_meta_ready, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&evt_meta_done, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&evt_offsets_ready, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&evt_offsets_done, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&evt_wq_heavy_ready, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&evt_wq_light_ready, hipEventDisableTiming));
}
} // namespace

template <class T>
static inline T clamp_val(T v, T lo, T hi) {
    return std::max(lo, std::min(hi, v));
}

float* forward(OssTransformerHybrid* transformer, int* tokens, const int* pos_per_token_h,
               int batch_size, const int* batch_indices_h, int B_stride) {
    OssConfig* p = &transformer->config;
    OssTransformerWeightsBFloat16* w = &transformer->weights;
    OssRunState* s = &transformer->state;

    float* x = s->x;
    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;
    size_t cos_sin_size = (head_dim / 2) * sizeof(float);

    int max_pos_in_batch = 0;
    for (int i = 0; i < batch_size; ++i)
        if (pos_per_token_h[i] > max_pos_in_batch)
            max_pos_in_batch = pos_per_token_h[i];

    // Initialize streams once
    create_streams_once();

    assert(batch_size <= s->h2d_stage_capacity);
    const int staging_slot = s->h2d_stage_cursor;
    const int next_slot = staging_slot ^ 1;
    s->h2d_stage_cursor = next_slot;

    // Ensure previous async copy using this staging slot completed before reusing buffers
    CHECK_HIP(hipEventSynchronize(s->h2d_stage_copied[staging_slot]));

    const size_t h2d_bytes = (size_t)batch_size * sizeof(int);
    std::memcpy(s->h_tokens_staging[staging_slot], tokens, h2d_bytes);
    std::memcpy(s->h_pos_staging[staging_slot], pos_per_token_h, h2d_bytes);
    std::memcpy(s->h_batch_indices_staging[staging_slot], batch_indices_h, h2d_bytes);

    // Start H2D copies on dedicated copy stream (async thanks to persistent pinned buffers)
    CHECK_HIP(hipMemcpyAsync(s->d_tokens, s->h_tokens_staging[staging_slot], h2d_bytes,
                             hipMemcpyHostToDevice, s_h2d));
    CHECK_HIP(hipMemcpyAsync(s->d_pos_per_token, s->h_pos_staging[staging_slot], h2d_bytes,
                             hipMemcpyHostToDevice, s_h2d));
    CHECK_HIP(hipMemcpyAsync(s->d_batch_indices, s->h_batch_indices_staging[staging_slot],
                             h2d_bytes, hipMemcpyHostToDevice, s_h2d));
    CHECK_HIP(hipEventRecord(s->h2d_stage_copied[staging_slot], s_h2d));
    CHECK_HIP(hipEventRecord(evt_h2d_ready, s_h2d));

    // Make compute wait only for the staged copies to finish (no global sync)
    CHECK_HIP(hipStreamWaitEvent(s_compute, evt_h2d_ready, 0));

    // ! Embedding (on compute stream)
    embed(x, w->token_embedding_table, s->d_tokens, batch_size, hidden_dim, s_compute);

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // ! RMSNorm (on compute stream)
        rmsnorm(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, batch_size, hidden_dim, s_compute);

        // ! QKV project (on compute stream)
        __hip_bfloat16* w_qkv =
            w->w_qkv +
            1ll * l * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        __hip_bfloat16* b_qkv =
            w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

        gemm_qkv(s->qkv, s->t, w_qkv, b_qkv, batch_size, hidden_dim,
                 (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim, s_compute);

        // ! Split QKV (on compute stream)
        split_qkv(s->qkv, s->q, s->key_cache, s->value_cache, batch_size, head_dim, p->n_attn_heads,
                  p->n_kv_heads, l, s->d_pos_per_token, p->seq_len, s->d_batch_indices, B_stride,
                  /*stream=*/s_compute, kv_dim, s->kv_cache_is_fp16, s->d_layer_kv_off,
                  s->d_layer_kv_cap, s->d_layer_is_local);

        // ! RoPE (Q and K) (on compute stream)
        const float ntk_beta = 32.0f;
        const float ntk_alpha = 1.0f;
        rope_qk(s->q, s->key_cache, batch_size, p->n_attn_heads, p->n_kv_heads, head_dim,
                p->seq_len, kv_dim, l, s->d_pos_per_token, s->d_batch_indices, s->kv_cache_is_fp16,
                B_stride, p->rope_theta, p->rope_scaling_factor, p->initial_context_length,
                ntk_beta, ntk_alpha,
                /*stream=*/s_compute, s->d_layer_kv_off, s->d_layer_kv_cap, s->d_layer_is_local);

        // ! Attention
        const int is_local_flag = s->h_layer_is_local ? s->h_layer_is_local[l] : 0;
        const int kv_cap = s->h_layer_kv_cap ? s->h_layer_kv_cap[l] : p->seq_len;
        const int is_local = (is_local_flag != 0);
        const float* mask_ptr =
            is_local ? nullptr : s->mask; // local => ring buffer path; dense => causal mask

        const int steps_so_far = max_pos_in_batch + 1;
        const int visible_len = is_local ? std::min(kv_cap, steps_so_far) : steps_so_far;

        fa(s->q, (const void*)s->key_cache, (const void*)s->value_cache, mask_ptr, w->attn_sinks,
           s->tb, batch_size, /*seq_len*/ visible_len, head_dim, kv_dim, kv_mul, p->sliding_window,
           (int)l, p->n_attn_heads, s->kv_cache_is_fp16, s->d_pos_per_token, s->d_batch_indices,
           (long long)B_stride, max_pos_in_batch, s->fa_partial_O, s->fa_partial_m, s->fa_partial_l,
           s_compute, s->d_layer_kv_off, s->d_layer_kv_cap, s->d_layer_is_local);

        // ! Output projection (on compute stream)
        __hip_bfloat16* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        __hip_bfloat16* b_o = w->b_o + 1ll * l * hidden_dim;

        gemm_o(s->tb2, s->tb, w_o, b_o, batch_size, head_dim * p->n_attn_heads, hidden_dim,
               s_compute);

        // ! Residual (on compute stream)
        vecadd(x, s->tb2, 1.0f, batch_size, hidden_dim, s_compute);

        // ! RMSNorm (on compute stream)
        rmsnorm(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, batch_size, hidden_dim, s_compute);

        // ! MoE Router (on compute stream)
        __hip_bfloat16* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        __hip_bfloat16* b_router = w->b_router + 1ll * l * n_experts;

        gemm_router(s->router_score, s->t, w_router, b_router, batch_size, hidden_dim, n_experts,
                    s_compute);

        // ! TopK and Softmax (on compute stream)
        topk(s->topk_v, s->topk_i, s->router_score, batch_size, n_experts, p->experts_per_token,
             s_compute);
        softmax(s->topk_v, batch_size, p->experts_per_token, s_compute);

        const int BKE = batch_size * p->experts_per_token;
        dim3 blk1(256), grd1((BKE + blk1.x - 1) / blk1.x);

        // ! Count & Compact (on compute stream)
        CHECK_HIP(hipMemsetAsync(s->expert_counts, 0, n_experts * sizeof(int), s_compute));
        hipLaunchKernelGGL(count_tokens_per_expert, grd1, blk1, 0, s_compute, s->topk_i, s->topk_v,
                           batch_size, p->experts_per_token, s->assign_expert, s->assign_token,
                           s->assign_weight, s->expert_counts, n_experts);
        hipLaunchKernelGGL(exclusive_scan_expert_offsets, dim3(1), dim3(1), 0, s_compute,
                           s->expert_counts, s->expert_offsets, n_experts);

        const int sumNe = BKE;
        hipLaunchKernelGGL(compact_by_expert_kernel, grd1, blk1, 0, s_compute, s->assign_expert,
                           s->assign_token, s->assign_weight, BKE, s->expert_offsets,
                           s->expert_counts, s->tokens_flat, s->weights_flat);

        const bool vecOK = ((reinterpret_cast<uintptr_t>(s->t) & 0xF) == 0) &&
                           ((reinterpret_cast<uintptr_t>(s->x_by_expert) & 0xF) == 0) &&
                           (hidden_dim % 4 == 0);
        if (vecOK) {
            const int H4 = hidden_dim >> 2;
            dim3 blk(256, 1, 1), grd((H4 + 255) / 256, sumNe);
            hipLaunchKernelGGL(gather_rows_vec4_kernel, grd, blk, 0, s_compute, s->t,
                               s->tokens_flat, s->x_by_expert, sumNe, H4);
        } else {
            dim3 blk(256, 1, 1), grd((hidden_dim + 255) / 256, sumNe);
            hipLaunchKernelGGL(gather_rows_kernel, grd, blk, 0, s_compute, s->t, s->tokens_flat,
                               s->x_by_expert, sumNe, hidden_dim);
        }

        // ! Build expert work queue
        thread_local static int* d_work_queue = nullptr;
        thread_local static int d_work_queue_capacity = 0;
        thread_local static Int2* d_meta = nullptr;

        const int required_ints = 3 * n_experts;
        if (required_ints > d_work_queue_capacity) {
            if (d_work_queue)
                CHECK_HIP(hipFree(d_work_queue));
            CHECK_HIP(hipMalloc(&d_work_queue, required_ints * sizeof(int)));
            d_work_queue_capacity = required_ints;
        }
        if (!d_meta)
            CHECK_HIP(hipMalloc(&d_meta, sizeof(Int2)));

        hipLaunchKernelGGL(build_expert_work_queue, dim3(1), dim3(1), 0, s_compute,
                           s->expert_offsets, d_work_queue, d_meta, n_experts);

        CHECK_HIP(hipEventRecord(evt_meta_ready, s_compute));
        CHECK_HIP(hipStreamWaitEvent(s_d2h, evt_meta_ready, 0));
        Int2 h_meta;
        CHECK_HIP(hipMemcpyAsync(&h_meta, d_meta, sizeof(Int2), hipMemcpyDeviceToHost, s_d2h));
        CHECK_HIP(hipEventRecord(evt_meta_done, s_d2h));
        CHECK_HIP(hipEventSynchronize(evt_meta_done));
        const int active_experts_all = h_meta.x;
        const int max_Ne_all = h_meta.y;

        // ---------------------------------------------------------------------
        // Adaptive heavy/light 2-bucket scheduler
        // ---------------------------------------------------------------------
        // Copy expert offsets to host and compute stats
        CHECK_HIP(hipEventRecord(evt_offsets_ready, s_compute));
        CHECK_HIP(hipStreamWaitEvent(s_d2h, evt_offsets_ready, 0));
        std::vector<int> h_offsets(n_experts + 1);
        CHECK_HIP(hipMemcpyAsync(h_offsets.data(), s->expert_offsets,
                                 (size_t)(n_experts + 1) * sizeof(int), hipMemcpyDeviceToHost,
                                 s_d2h));
        CHECK_HIP(hipEventRecord(evt_offsets_done, s_d2h));
        CHECK_HIP(hipEventSynchronize(evt_offsets_done));
        // compute per-expert counts, active list, avg and max
        int active_experts = 0;
        long long sumNe_ll = 0;
        int maxNe = 0;
        for (int e = 0; e < n_experts; ++e) {
            const int ne = h_offsets[e + 1] - h_offsets[e];
            if (ne > 0) {
                ++active_experts;
                sumNe_ll += ne;
                if (ne > maxNe)
                    maxNe = ne;
            }
        }
        if (active_experts == 0) {
            vecadd_and_zero(x, s->e_agg, 1.0f, batch_size, hidden_dim, s_compute);
            continue;
        }
        const float avgNe = float(sumNe_ll) / float(active_experts);

        // classify heavy vs light
        float heavy_factor = ADAPT_HEAVY_FACTOR_DEFAULT;
        if (active_experts <= ADAPT_VERY_SPARSE_CUTOFF) {
            heavy_factor = 1.0f; // treat everyone heavy for 1..6 actives
        } else if (active_experts <= 12) {
            heavy_factor = ADAPT_HEAVY_FACTOR_SPARSE;
        }
        const int heavy_thresh = int(std::ceil(heavy_factor * avgNe));

        std::vector<int> heavy_ids;
        std::vector<int> light_ids;
        heavy_ids.reserve(active_experts);
        light_ids.reserve(active_experts);
        int heavy_maxNe = 0;
        for (int e = 0; e < n_experts; ++e) {
            const int ne = h_offsets[e + 1] - h_offsets[e];
            if (ne <= 0)
                continue;
            if (active_experts <= ADAPT_VERY_SPARSE_CUTOFF || ne >= heavy_thresh) {
                heavy_ids.push_back(e);
                if (ne > heavy_maxNe)
                    heavy_maxNe = ne;
            } else {
                light_ids.push_back(e);
            }
        }
        const int heavy_count = (int)heavy_ids.size();
        const int light_count = (int)light_ids.size();

        // device info
        hipDeviceProp_t prop{};
        int dev = 0;
        CHECK_HIP(hipGetDevice(&dev));
        CHECK_HIP(hipGetDeviceProperties(&prop, dev));
        const int sms = std::max(1, prop.multiProcessorCount);

        // occupancy-guided rows_hint picker
        auto pick_rows_hint = [&](int bucket_experts, int grid_y, int min_tiles_per_exp,
                                  int max_tiles_per_exp) -> int {
            if (bucket_experts <= 0)
                return 0;
            int target_gridx = (sms * ADAPT_CTAS_PER_SM_CAP) / std::max(1, grid_y);
            target_gridx = clamp_val(target_gridx, 1, kGridCap);
            int tiles_per_exp =
                std::max(min_tiles_per_exp,
                         std::min(max_tiles_per_exp,
                                  std::max(1, target_gridx / std::max(1, bucket_experts))));
            const int RH_MIN = ADAPT_RH_MIN_TILES * ADAPT_MT_N;
            const int RH_MAX = ADAPT_RH_MAX_TILES * ADAPT_MT_N;
            int rows_hint = tiles_per_exp * ADAPT_MT_N;
            return clamp_val(rows_hint, RH_MIN, RH_MAX);
        };

        // N-dimension for the two MLPs: 2I for MLP1 GEMM, H for MLP2 GEMM
        const int grid_y_mlp1 = std::max(1, CEIL_DIV(2 * intermediate_dim, ADAPT_MT_M));
        const int grid_y_mlp2 = std::max(1, CEIL_DIV(hidden_dim, ADAPT_MT_M));

        int rh_heavy_mlp1 = pick_rows_hint(heavy_count, grid_y_mlp1, /*min*/ 2, /*max*/ 8);
        int rh_light_mlp1 = pick_rows_hint(light_count, grid_y_mlp1, /*min*/ 1, /*max*/ 4);
        int rh_heavy_mlp2 = pick_rows_hint(heavy_count, grid_y_mlp2, /*min*/ 2, /*max*/ 8);
        int rh_light_mlp2 = pick_rows_hint(light_count, grid_y_mlp2, /*min*/ 1, /*max*/ 4);

        if (heavy_count > 0) {
            const int cover_heavy = CEIL_DIV(heavy_maxNe, ADAPT_MT_N) * ADAPT_MT_N;
            rh_heavy_mlp1 = std::max(rh_heavy_mlp1, cover_heavy);
            rh_heavy_mlp2 = std::max(rh_heavy_mlp2, cover_heavy);
        }
        if (light_count > 0) {
            const int light_cap = std::max(ADAPT_MT_N, int(std::ceil(avgNe)));
            rh_light_mlp1 = std::min(rh_light_mlp1, light_cap);
            rh_light_mlp2 = std::min(rh_light_mlp2, light_cap);
        } else {
            rh_light_mlp1 = 0;
            rh_light_mlp2 = 0;
        }

        // Build compact per-bucket work queues (triples: e, offset, Ne)
        auto make_bucket_wq = [&](const std::vector<int>& ids) -> std::vector<int> {
            std::vector<int> wq;
            wq.reserve((size_t)ids.size() * 3);
            for (int e : ids) {
                const int off = h_offsets[e];
                const int ne = h_offsets[e + 1] - h_offsets[e];
                // skip safety
                if (ne <= 0)
                    continue;
                wq.push_back(e);
                wq.push_back(off);
                wq.push_back(ne);
            }
            return wq;
        };

        std::vector<int> wq_heavy_h = make_bucket_wq(heavy_ids);
        std::vector<int> wq_light_h = make_bucket_wq(light_ids);

        int* d_wq_heavy = s->d_wq_heavy;
        int* d_wq_light = s->d_wq_light;
        if (!wq_heavy_h.empty()) {
            assert(d_wq_heavy && (size_t)wq_heavy_h.size() <= (size_t)s->d_wq_heavy_capacity);
            CHECK_HIP(hipMemcpyAsync(d_wq_heavy, wq_heavy_h.data(),
                                     (size_t)wq_heavy_h.size() * sizeof(int), hipMemcpyHostToDevice,
                                     s_h2d));
            CHECK_HIP(hipEventRecord(evt_wq_heavy_ready, s_h2d));
            CHECK_HIP(hipStreamWaitEvent(s_moe_heavy, evt_wq_heavy_ready, 0));
        }
        if (!wq_light_h.empty()) {
            assert(d_wq_light && (size_t)wq_light_h.size() <= (size_t)s->d_wq_light_capacity);
            CHECK_HIP(hipMemcpyAsync(d_wq_light, wq_light_h.data(),
                                     (size_t)wq_light_h.size() * sizeof(int), hipMemcpyHostToDevice,
                                     s_h2d));
            CHECK_HIP(hipEventRecord(evt_wq_light_ready, s_h2d));
            CHECK_HIP(hipStreamWaitEvent(s_moe_light, evt_wq_light_ready, 0));
        }

        // Launch parameters shared with your kernels
        const long long mlp1_weight_stride = (2LL * p->intermediate_dim) * hidden_dim;
        const long long mlp1_bias_stride = (2LL * p->intermediate_dim);
        const long long mlp2_weight_stride = hidden_dim * (long long)p->intermediate_dim;
        const long long mlp2_bias_stride = hidden_dim;
        const float alpha = 1.702f;
        const int grid_cap_light = kGridCap;
        const int grid_cap_heavy = std::max(1, kGridCap - std::max(1, kGridCap / 3));

        // ---- Light bucket (higher priority stream) ----
        if (light_count > 0) {
            moe_mlp1_swiglu(d_wq_light, /*work_start*/ 0, /*work_count*/ light_count,
                            s->x_by_expert, s->gate_by_expert,
                            w->w_mlp1 + 1ll * l * p->n_experts * mlp1_weight_stride,
                            w->b_mlp1 + 1ll * l * p->n_experts * mlp1_bias_stride, hidden_dim,
                            p->intermediate_dim, mlp1_weight_stride, mlp1_bias_stride, alpha,
                            p->swiglu_limit, rh_light_mlp1, grid_cap_light, s_moe_light);

            moe_mlp2_scatter(d_wq_light, /*work_start*/ 0, /*work_count*/ light_count,
                             s->gate_by_expert, s->tokens_flat, s->weights_flat, s->e_agg,
                             w->w_mlp2 + 1ll * l * p->n_experts * mlp2_weight_stride,
                             w->b_mlp2 + 1ll * l * p->n_experts * mlp2_bias_stride,
                             p->intermediate_dim, hidden_dim, mlp2_weight_stride, mlp2_bias_stride,
                             rh_light_mlp2, grid_cap_light, s_moe_light);
            CHECK_HIP(hipEventRecord(evt_moe_light_done, s_moe_light));
        }

        // ---- Heavy bucket (throttled grid to leave SM headroom) ----
        if (heavy_count > 0) {
            moe_mlp1_swiglu(d_wq_heavy, /*work_start*/ 0, /*work_count*/ heavy_count,
                            s->x_by_expert, s->gate_by_expert,
                            w->w_mlp1 + 1ll * l * p->n_experts * mlp1_weight_stride,
                            w->b_mlp1 + 1ll * l * p->n_experts * mlp1_bias_stride, hidden_dim,
                            p->intermediate_dim, mlp1_weight_stride, mlp1_bias_stride, alpha,
                            p->swiglu_limit, rh_heavy_mlp1, grid_cap_heavy, s_moe_heavy);

            moe_mlp2_scatter(d_wq_heavy, /*work_start*/ 0, /*work_count*/ heavy_count,
                             s->gate_by_expert, s->tokens_flat, s->weights_flat, s->e_agg,
                             w->w_mlp2 + 1ll * l * p->n_experts * mlp2_weight_stride,
                             w->b_mlp2 + 1ll * l * p->n_experts * mlp2_bias_stride,
                             p->intermediate_dim, hidden_dim, mlp2_weight_stride, mlp2_bias_stride,
                             rh_heavy_mlp2, grid_cap_heavy, s_moe_heavy);
            CHECK_HIP(hipEventRecord(evt_moe_heavy_done, s_moe_heavy));
        }

        // Wait for both MoE buckets to complete before using e_agg
        if (heavy_count > 0)
            CHECK_HIP(hipStreamWaitEvent(s_compute, evt_moe_heavy_done, 0));
        if (light_count > 0)
            CHECK_HIP(hipStreamWaitEvent(s_compute, evt_moe_light_done, 0));

        // ! Residual (on compute stream)
        vecadd_and_zero(x, s->e_agg, 1.0f, batch_size, hidden_dim, s_compute);
    }

    auto launch_output_stage = [&](bool use_graph) {
        if (!use_graph) {
            rmsnorm(x, x, w->rms_out_w, batch_size, hidden_dim, s_compute);
            gemm_logits(s->logits, x, w->out, batch_size, hidden_dim, p->vocab_size, s_compute);
            return;
        }

        if (!s->forward_graph_ready || s->forward_graph_batch_size != batch_size) {
            if (s->forward_graph_exec) {
                CHECK_HIP(hipGraphExecDestroy(s->forward_graph_exec));
                s->forward_graph_exec = nullptr;
            }
            if (s->forward_graph) {
                CHECK_HIP(hipGraphDestroy(s->forward_graph));
                s->forward_graph = nullptr;
            }

            CHECK_HIP(hipStreamBeginCapture(s_compute, hipStreamCaptureModeThreadLocal));
            rmsnorm(x, x, w->rms_out_w, batch_size, hidden_dim, s_compute);
            gemm_logits(s->logits, x, w->out, batch_size, hidden_dim, p->vocab_size, s_compute);
            hipGraph_t graph = nullptr;
            CHECK_HIP(hipStreamEndCapture(s_compute, &graph));
            CHECK_HIP(hipGraphInstantiate(&s->forward_graph_exec, graph, nullptr, nullptr, 0));
            s->forward_graph = graph;
            s->forward_graph_ready = true;
            s->forward_graph_batch_size = batch_size;
        }

        CHECK_HIP(hipGraphLaunch(s->forward_graph_exec, s_compute));
    };

    launch_output_stage(s->forward_graph_enabled);

    // **CRITICAL**: Synchronize compute stream before returning results!
    // Without this, the host returns before GPU work completes, causing wrong/incomplete results
    CHECK_HIP(hipStreamSynchronize(s_compute));

    return s->logits;
}
