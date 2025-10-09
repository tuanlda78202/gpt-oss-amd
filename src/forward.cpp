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
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <omp.h>
#include <vector>

#ifndef ADAPT_MT_M
#define ADAPT_MT_M 64
#endif
#ifndef ADAPT_MT_N
#define ADAPT_MT_N 32
#endif
#ifndef ADAPT_CTAS_PER_SM_CAP
#define ADAPT_CTAS_PER_SM_CAP 2
#endif
#ifndef ADAPT_VERY_SPARSE_CUTOFF
#define ADAPT_VERY_SPARSE_CUTOFF 6
#endif
#ifndef ADAPT_HEAVY_FACTOR_SPARSE
#define ADAPT_HEAVY_FACTOR_SPARSE 1.5f
#endif
#ifndef ADAPT_HEAVY_FACTOR_DEFAULT
#define ADAPT_HEAVY_FACTOR_DEFAULT 2.0f
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

static thread_local int kGridCap = [] {
    hipDeviceProp_t prop{};
    int dev = 0;
    CHECK_HIP(hipGetDevice(&dev));
    CHECK_HIP(hipGetDeviceProperties(&prop, dev));
    return prop.multiProcessorCount * 2;
}();

thread_local static int* h_expert_offsets = nullptr;
thread_local static int h_expert_offsets_capacity = 0;

static inline void ensure_h_expert_offsets(int need) {
    if (need <= h_expert_offsets_capacity)
        return;
    if (h_expert_offsets)
        CHECK_HIP(hipHostFree(h_expert_offsets));
    CHECK_HIP(hipHostMalloc((void**)&h_expert_offsets, need * sizeof(int), hipHostMallocDefault));
    h_expert_offsets_capacity = need;
}

namespace {
struct StreamsAndEvents {
    hipStream_t compute = nullptr;   // main compute stream
    hipStream_t h2d = nullptr;       // host->device copies
    hipStream_t d2h = nullptr;       // device->host copies
    hipStream_t moe_heavy = nullptr; // MoE heavy bucket
    hipStream_t moe_light = nullptr; // MoE light bucket
    hipEvent_t evt_h2d_ready = nullptr;
    hipEvent_t evt_moe_heavy_done = nullptr;
    hipEvent_t evt_moe_light_done = nullptr;
    hipEvent_t evt_meta_ready = nullptr;
    hipEvent_t evt_offsets_ready = nullptr;
    hipEvent_t evt_offsets_done = nullptr;
    hipEvent_t evt_wq_heavy_ready = nullptr;
    hipEvent_t evt_wq_light_ready = nullptr;
};

static thread_local StreamsAndEvents tls;

void create_streams_once() {
    if (tls.compute)
        return;
    CHECK_HIP(hipStreamCreateWithFlags(&tls.compute, hipStreamNonBlocking));
    CHECK_HIP(hipStreamCreateWithFlags(&tls.h2d, hipStreamNonBlocking));
    CHECK_HIP(hipStreamCreateWithFlags(&tls.d2h, hipStreamNonBlocking));

    int least_priority = 0, greatest_priority = 0;
    CHECK_HIP(hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    CHECK_HIP(hipStreamCreateWithPriority(&tls.moe_heavy, hipStreamNonBlocking, least_priority));
    CHECK_HIP(hipStreamCreateWithPriority(&tls.moe_light, hipStreamNonBlocking, greatest_priority));
    CHECK_HIP(hipEventCreateWithFlags(&tls.evt_h2d_ready, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&tls.evt_moe_heavy_done, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&tls.evt_moe_light_done, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&tls.evt_meta_ready, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&tls.evt_offsets_ready, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&tls.evt_offsets_done, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&tls.evt_wq_heavy_ready, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&tls.evt_wq_light_ready, hipEventDisableTiming));
}
}

template <class T>
static inline T clamp_val(T v, T lo, T hi) {
    return std::max(lo, std::min(hi, v));
}

namespace {

void ensure_workspace_capacity(OssExpertWorkspace* workspace, int device_id, int required_tokens,
                               int hidden_dim, int intermediate_dim, int /*batch_size*/) {
    const bool needs_token_space = required_tokens > workspace->capacity_tokens;

    if (!needs_token_space)
        return;

    CHECK_HIP(hipSetDevice(device_id));

    if (needs_token_space) {
        if (workspace->x_by_expert)
            CHECK_HIP(hipFree(workspace->x_by_expert));
        if (workspace->mlp1_by_expert)
            CHECK_HIP(hipFree(workspace->mlp1_by_expert));
        if (workspace->gate_by_expert)
            CHECK_HIP(hipFree(workspace->gate_by_expert));
        if (workspace->y_by_expert)
            CHECK_HIP(hipFree(workspace->y_by_expert));
        if (workspace->tokens_by_expert)
            CHECK_HIP(hipFree(workspace->tokens_by_expert));
        if (workspace->weights_by_expert)
            CHECK_HIP(hipFree(workspace->weights_by_expert));

        size_t tokens_bytes = (size_t)required_tokens * hidden_dim * sizeof(float);
        size_t mlp1_bytes = (size_t)required_tokens * 2 * intermediate_dim * sizeof(float);
        size_t gate_bytes = (size_t)required_tokens * intermediate_dim * sizeof(float);
        size_t tokens_index_bytes = (size_t)required_tokens * sizeof(int);
        size_t weights_bytes = (size_t)required_tokens * sizeof(float);

        if (required_tokens > 0) {
            CHECK_HIP(hipMalloc(&workspace->x_by_expert, tokens_bytes));
            CHECK_HIP(hipMalloc(&workspace->mlp1_by_expert, mlp1_bytes));
            CHECK_HIP(hipMalloc(&workspace->gate_by_expert, gate_bytes));
            CHECK_HIP(hipMalloc(&workspace->y_by_expert, tokens_bytes));
            CHECK_HIP(hipMalloc(&workspace->tokens_by_expert, tokens_index_bytes));
            CHECK_HIP(hipMalloc(&workspace->weights_by_expert, weights_bytes));
        } else {
            workspace->x_by_expert = nullptr;
            workspace->mlp1_by_expert = nullptr;
            workspace->gate_by_expert = nullptr;
            workspace->y_by_expert = nullptr;
            workspace->tokens_by_expert = nullptr;
            workspace->weights_by_expert = nullptr;
        }

        workspace->capacity_tokens = required_tokens;
    }
}

std::mutex& shard_mutex(OssExpertShard* shard, int dp_rank) {
    auto** handles = reinterpret_cast<std::mutex**>(shard->mutex_handles);
    return *handles[dp_rank];
}

} // namespace

class HipDeviceGuard {
  public:
    explicit HipDeviceGuard(int device) : prev_(-1), restore_(false) {
        CHECK_HIP(hipGetDevice(&prev_));
        if (prev_ != device) {
            CHECK_HIP(hipSetDevice(device));
            restore_ = true;
        }
    }

    ~HipDeviceGuard() {
        if (restore_)
            CHECK_HIP(hipSetDevice(prev_));
    }

    HipDeviceGuard(const HipDeviceGuard&) = delete;
    HipDeviceGuard& operator=(const HipDeviceGuard&) = delete;

  private:
    int prev_;
    bool restore_;
};

float* forward(OssTransformerHybrid* transformer, OssExpertParallelGroup* ep_group, int* tokens,
               const int* pos_per_token_h, int batch_size, const int* batch_indices_h,
               int B_stride) {
    OssConfig* p = &transformer->config;
    OssTransformerWeightsBFloat16* w = &transformer->weights;
    OssRunState* s = &transformer->state;

    if (!ep_group || ep_group->ep_size <= 0 || !ep_group->shards) {
        fprintf(stderr, "Invalid expert parallel context\n");
        exit(EXIT_FAILURE);
    }

    const int ep_size = ep_group->ep_size;
    const int primary_device = transformer->device_id;
    CHECK_HIP(hipSetDevice(primary_device));

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

    create_streams_once();

    assert(batch_size <= s->h2d_stage_capacity);
    const int staging_slot = s->h2d_stage_cursor;
    const int next_slot = staging_slot ^ 1;
    s->h2d_stage_cursor = next_slot;

    CHECK_HIP(hipEventSynchronize(s->h2d_stage_copied[staging_slot]));

    const size_t h2d_bytes = (size_t)batch_size * sizeof(int);
    std::memcpy(s->h_tokens_staging[staging_slot], tokens, h2d_bytes);
    std::memcpy(s->h_pos_staging[staging_slot], pos_per_token_h, h2d_bytes);
    std::memcpy(s->h_batch_indices_staging[staging_slot], batch_indices_h, h2d_bytes);

    CHECK_HIP(hipMemcpyAsync(s->d_tokens, s->h_tokens_staging[staging_slot], h2d_bytes,
                             hipMemcpyHostToDevice, tls.h2d));
    CHECK_HIP(hipMemcpyAsync(s->d_pos_per_token, s->h_pos_staging[staging_slot], h2d_bytes,
                             hipMemcpyHostToDevice, tls.h2d));
    CHECK_HIP(hipMemcpyAsync(s->d_batch_indices, s->h_batch_indices_staging[staging_slot],
                             h2d_bytes, hipMemcpyHostToDevice, tls.h2d));
    CHECK_HIP(hipEventRecord(s->h2d_stage_copied[staging_slot], tls.h2d));
    CHECK_HIP(hipEventRecord(tls.evt_h2d_ready, tls.h2d));

    CHECK_HIP(hipStreamWaitEvent(tls.compute, tls.evt_h2d_ready, 0));

    // ! Embedding
    embed(x, w->token_embedding_table, s->d_tokens, batch_size, hidden_dim, tls.compute);

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // ! RMSNorm
        rmsnorm(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, batch_size, hidden_dim, tls.compute);

        // ! QKV project
        __hip_bfloat16* w_qkv =
            w->w_qkv +
            1ll * l * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        __hip_bfloat16* b_qkv =
            w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

        gemm_qkv(s->qkv, s->t, w_qkv, b_qkv, batch_size, hidden_dim,
                 (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim, tls.compute);

        // ! Split QKV
        split_qkv(s->qkv, s->q, s->key_cache, s->value_cache, batch_size, head_dim, p->n_attn_heads,
                  p->n_kv_heads, l, s->d_pos_per_token, p->seq_len, s->d_batch_indices, B_stride,
                  /*stream=*/tls.compute, kv_dim, s->kv_cache_is_fp16, s->d_layer_kv_off,
                  s->d_layer_kv_cap, s->d_layer_is_local);

        // ! RoPE (Q and K)
        const float ntk_beta = 32.0f;
        const float ntk_alpha = 1.0f;
        rope_qk(s->q, s->key_cache, batch_size, p->n_attn_heads, p->n_kv_heads, head_dim,
                p->seq_len, kv_dim, l, s->d_pos_per_token, s->d_batch_indices, s->kv_cache_is_fp16,
                B_stride, p->rope_theta, p->rope_scaling_factor, p->initial_context_length,
                ntk_beta, ntk_alpha,
                /*stream=*/tls.compute, s->d_layer_kv_off, s->d_layer_kv_cap, s->d_layer_is_local);

        // ! Attention
        const int is_local_flag = s->h_layer_is_local ? s->h_layer_is_local[l] : 0;
        const int kv_cap = s->h_layer_kv_cap ? s->h_layer_kv_cap[l] : p->seq_len;
        const int is_local = (is_local_flag != 0);
        const float* mask_ptr =
            is_local ? nullptr : s->mask;

        const int steps_so_far = max_pos_in_batch + 1;
        const int visible_len = is_local ? std::min(kv_cap, steps_so_far) : steps_so_far;

        fa(s->q, (const void*)s->key_cache, (const void*)s->value_cache, mask_ptr, w->attn_sinks,
           s->tb, batch_size, /*seq_len*/ visible_len, head_dim, kv_dim, kv_mul, p->sliding_window,
           (int)l, p->n_attn_heads, s->kv_cache_is_fp16, s->d_pos_per_token, s->d_batch_indices,
           (long long)B_stride, max_pos_in_batch, tls.compute, s->d_layer_kv_off, s->d_layer_kv_cap,
           s->d_layer_is_local);

        // ! Output projection
        __hip_bfloat16* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        __hip_bfloat16* b_o = w->b_o + 1ll * l * hidden_dim;

        gemm_o(s->tb2, s->tb, w_o, b_o, batch_size, head_dim * p->n_attn_heads, hidden_dim,
               tls.compute);

        // ! Residual
        vecadd(x, s->tb2, 1.0f, batch_size, hidden_dim, tls.compute);

        // ! RMSNorm
        rmsnorm(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, batch_size, hidden_dim, tls.compute);

        // ! MoE Router
        __hip_bfloat16* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        __hip_bfloat16* b_router = w->b_router + 1ll * l * n_experts;

        gemm_router(s->router_score, s->t, w_router, b_router, batch_size, hidden_dim, n_experts,
                    tls.compute);

        // ! TopK and Softmax
        topk(s->topk_v, s->topk_i, s->router_score, batch_size, n_experts, p->experts_per_token,
             tls.compute);
        softmax(s->topk_v, batch_size, p->experts_per_token, tls.compute);

        const int BKE = batch_size * p->experts_per_token;
        dim3 blk1(256), grd1((BKE + blk1.x - 1) / blk1.x);

        // ! Count & Compact
        CHECK_HIP(hipMemsetAsync(s->expert_counts, 0, n_experts * sizeof(int), tls.compute));
        hipLaunchKernelGGL(count_tokens_per_expert, grd1, blk1, 0, tls.compute, s->topk_i,
                           s->topk_v, batch_size, p->experts_per_token, s->assign_expert,
                           s->assign_token, s->assign_weight, s->expert_counts, n_experts);
        hipLaunchKernelGGL(exclusive_scan_expert_offsets, dim3(1), dim3(1), 0, tls.compute,
                           s->expert_counts, s->expert_offsets, n_experts);

        ensure_h_expert_offsets(n_experts + 1);
        CHECK_HIP(hipEventRecord(tls.evt_offsets_ready, tls.compute));
        CHECK_HIP(hipStreamWaitEvent(tls.d2h, tls.evt_offsets_ready, 0));
        CHECK_HIP(hipMemcpyAsync(h_expert_offsets,
                                 s->expert_offsets,
                                 (n_experts + 1) * sizeof(int), hipMemcpyDeviceToHost, tls.d2h));
        CHECK_HIP(hipEventRecord(tls.evt_offsets_done, tls.d2h));

        const int sumNe = BKE;
        hipLaunchKernelGGL(compact_by_expert_kernel, grd1, blk1, 0, tls.compute, s->assign_expert,
                           s->assign_token, s->assign_weight, BKE, s->expert_offsets,
                           s->expert_counts, s->tokens_flat, s->weights_flat);

        const bool vecOK = ((reinterpret_cast<uintptr_t>(s->t) & 0xF) == 0) &&
                           ((reinterpret_cast<uintptr_t>(s->x_by_expert) & 0xF) == 0) &&
                           (hidden_dim % 4 == 0);
        if (vecOK) {
            const int H4 = hidden_dim >> 2;
            dim3 blk(256, 1, 1), grd((H4 + 255) / 256, sumNe);
            hipLaunchKernelGGL(gather_rows_vec4_kernel, grd, blk, 0, tls.compute, s->t,
                               s->tokens_flat, s->x_by_expert, sumNe, H4);
        } else {
            dim3 blk(256, 1, 1), grd((hidden_dim + 255) / 256, sumNe);
            hipLaunchKernelGGL(gather_rows_kernel, grd, blk, 0, tls.compute, s->t, s->tokens_flat,
                               s->x_by_expert, sumNe, hidden_dim);
        }

        // ! Build expert work queue
        thread_local static MoEStats* d_moe_stats = nullptr;
        if (!d_moe_stats)
            CHECK_HIP(hipMalloc(&d_moe_stats, sizeof(MoEStats)));

        assert(3 * n_experts <= s->d_wq_heavy_capacity);
        assert(3 * n_experts <= s->d_wq_light_capacity);
        hipLaunchKernelGGL(classify_and_build_queues, dim3(1), dim3(1), 0, tls.compute,
                           s->expert_offsets, n_experts, ADAPT_VERY_SPARSE_CUTOFF,
                           ADAPT_HEAVY_FACTOR_SPARSE, ADAPT_HEAVY_FACTOR_DEFAULT, s->d_wq_heavy,
                           s->d_wq_light, d_moe_stats);

        CHECK_HIP(hipEventRecord(tls.evt_meta_ready, tls.compute));
        CHECK_HIP(hipStreamWaitEvent(tls.moe_heavy, tls.evt_meta_ready, 0));
        CHECK_HIP(hipStreamWaitEvent(tls.moe_light, tls.evt_meta_ready, 0));
        CHECK_HIP(hipEventSynchronize(tls.evt_offsets_done));

        int active = 0, heavy_count = 0, light_count = 0, heavy_maxNe = 0;
        long long total_assigned = 0;
        for (int e = 0; e < n_experts; ++e) {
            int off = h_expert_offsets[e];
            int ne = h_expert_offsets[e + 1] - off;
            if (ne > 0) {
                ++active;
                total_assigned += ne;
                if (ne > heavy_maxNe)
                    heavy_maxNe = ne;
            }
        }

        if (active == 0) {
            vecadd_and_zero(x, s->e_agg, 1.0f, batch_size, hidden_dim, tls.compute);
            continue;
        }

        float avgNe = float(total_assigned) / float(active);
        float heavy_factor = ADAPT_HEAVY_FACTOR_DEFAULT;
        if (active <= ADAPT_VERY_SPARSE_CUTOFF) {
            heavy_factor = 1.0f;
        } else if (active <= 12) {
            heavy_factor = ADAPT_HEAVY_FACTOR_SPARSE;
        }
        const int heavy_thresh = int(ceilf(heavy_factor * avgNe));

        for (int e = 0; e < n_experts; ++e) {
            int off = h_expert_offsets[e];
            int ne = h_expert_offsets[e + 1] - off;
            if (ne <= 0)
                continue;
            if (active <= ADAPT_VERY_SPARSE_CUTOFF || ne >= heavy_thresh)
                ++heavy_count;
            else
                ++light_count;
        }

        const int sms = std::max(1, kGridCap / 2);

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
            const float avgNe_f = float(total_assigned) / float(active);
            const int light_cap = std::max(ADAPT_MT_N, int(std::ceil(avgNe_f)));
            rh_light_mlp1 = std::min(rh_light_mlp1, light_cap);
            rh_light_mlp2 = std::min(rh_light_mlp2, light_cap);
        } else {
            rh_light_mlp1 = 0;
            rh_light_mlp2 = 0;
        }

        if (!ep_group || ep_group->ep_size <= 1 || transformer->ep_local_experts == 0) {
            // Single-GPU
            const long long mlp1_weight_stride = (2LL * p->intermediate_dim) * hidden_dim;
            const long long mlp1_bias_stride = (2LL * p->intermediate_dim);
            const long long mlp2_weight_stride = hidden_dim * (long long)p->intermediate_dim;
            const long long mlp2_bias_stride = hidden_dim;
            const float alpha = 1.702f;
            const int grid_cap_light = kGridCap;
            const int grid_cap_heavy = std::max(1, kGridCap - std::max(1, kGridCap / 3));

            // ---- Light bucket ----
            if (light_count > 0) {
                moe_mlp1_swiglu(s->d_wq_light, /*work_start*/ 0, /*work_count*/ light_count,
                                s->x_by_expert, s->gate_by_expert,
                                w->w_mlp1 + 1ll * l * p->n_experts * mlp1_weight_stride,
                                w->b_mlp1 + 1ll * l * p->n_experts * mlp1_bias_stride, hidden_dim,
                                p->intermediate_dim, mlp1_weight_stride, mlp1_bias_stride, alpha,
                                p->swiglu_limit, rh_light_mlp1, grid_cap_light, tls.moe_light);

                moe_mlp2_scatter(s->d_wq_light, /*work_start*/ 0, /*work_count*/ light_count,
                                 s->gate_by_expert, s->tokens_flat, s->weights_flat, s->e_agg,
                                 w->w_mlp2 + 1ll * l * p->n_experts * mlp2_weight_stride,
                                 w->b_mlp2 + 1ll * l * p->n_experts * mlp2_bias_stride,
                                 p->intermediate_dim, hidden_dim, mlp2_weight_stride,
                                 mlp2_bias_stride, rh_light_mlp2, grid_cap_light, tls.moe_light);
                CHECK_HIP(hipEventRecord(tls.evt_moe_light_done, tls.moe_light));
            }

            // ---- Heavy bucket ----
            if (heavy_count > 0) {
                moe_mlp1_swiglu(s->d_wq_heavy, /*work_start*/ 0, /*work_count*/ heavy_count,
                                s->x_by_expert, s->gate_by_expert,
                                w->w_mlp1 + 1ll * l * p->n_experts * mlp1_weight_stride,
                                w->b_mlp1 + 1ll * l * p->n_experts * mlp1_bias_stride, hidden_dim,
                                p->intermediate_dim, mlp1_weight_stride, mlp1_bias_stride, alpha,
                                p->swiglu_limit, rh_heavy_mlp1, grid_cap_heavy, tls.moe_heavy);

                moe_mlp2_scatter(s->d_wq_heavy, /*work_start*/ 0, /*work_count*/ heavy_count,
                                 s->gate_by_expert, s->tokens_flat, s->weights_flat, s->e_agg,
                                 w->w_mlp2 + 1ll * l * p->n_experts * mlp2_weight_stride,
                                 w->b_mlp2 + 1ll * l * p->n_experts * mlp2_bias_stride,
                                 p->intermediate_dim, hidden_dim, mlp2_weight_stride,
                                 mlp2_bias_stride, rh_heavy_mlp2, grid_cap_heavy, tls.moe_heavy);
                CHECK_HIP(hipEventRecord(tls.evt_moe_heavy_done, tls.moe_heavy));
            }

            if (heavy_count > 0)
                CHECK_HIP(hipStreamWaitEvent(tls.compute, tls.evt_moe_heavy_done, 0));
            if (light_count > 0)
                CHECK_HIP(hipStreamWaitEvent(tls.compute, tls.evt_moe_light_done, 0));

            // ! Residual
            vecadd_and_zero(x, s->e_agg, 1.0f, batch_size, hidden_dim, tls.compute);
        } else {
            // ------------------ Multi-GPUs ------------------
            const int active_experts = heavy_count + light_count;
            std::vector<int> wq_heavy_all;
            std::vector<int> wq_light_all;
            bool queued_wq_d2h = false;
            if (heavy_count > 0) {
                wq_heavy_all.resize(3 * heavy_count);
                if (!queued_wq_d2h)
                    CHECK_HIP(hipStreamWaitEvent(tls.d2h, tls.evt_meta_ready, 0));
                CHECK_HIP(hipMemcpyAsync(wq_heavy_all.data(), s->d_wq_heavy,
                                         wq_heavy_all.size() * sizeof(int), hipMemcpyDeviceToHost,
                                         tls.d2h));
                queued_wq_d2h = true;
            }
            if (light_count > 0) {
                wq_light_all.resize(3 * light_count);
                if (!queued_wq_d2h)
                    CHECK_HIP(hipStreamWaitEvent(tls.d2h, tls.evt_meta_ready, 0));
                CHECK_HIP(hipMemcpyAsync(wq_light_all.data(), s->d_wq_light,
                                         wq_light_all.size() * sizeof(int), hipMemcpyDeviceToHost,
                                         tls.d2h));
                queued_wq_d2h = true;
            }
            if (queued_wq_d2h)
                CHECK_HIP(hipStreamSynchronize(tls.d2h));
            std::vector<int> h_work_queue;
            if (active_experts > 0) {
                h_work_queue.reserve(wq_heavy_all.size() + wq_light_all.size());
                h_work_queue.insert(h_work_queue.end(), wq_heavy_all.begin(), wq_heavy_all.end());
                h_work_queue.insert(h_work_queue.end(), wq_light_all.begin(), wq_light_all.end());
            }

            struct ShardWork {
                std::vector<int> experts;
                std::vector<int> offsets;
                std::vector<int> counts;
                int total = 0;
                int maxNe = 0;
            };
            std::vector<ShardWork> shard_work(ep_size);

            int local_shard_index = -1;
            for (int sidx = 0; sidx < ep_size; ++sidx) {
                if (ep_group->shards[sidx]->model == transformer) {
                    local_shard_index = sidx;
                    break;
                }
            }

            for (int idx = 0; idx < active_experts; ++idx) {
                int global_e = h_work_queue[idx * 3 + 0];
                int offset = h_work_queue[idx * 3 + 1];
                int count = h_work_queue[idx * 3 + 2];

                int shard_idx = -1;
                for (int sidx = 0; sidx < ep_size; ++sidx) {
                    OssTransformerHybrid* shard_model = ep_group->shards[sidx]->model;
                    int start = shard_model->ep_expert_offset;
                    int span = shard_model->ep_local_experts;
                    if (span > 0 && global_e >= start && global_e < start + span) {
                        shard_idx = sidx;
                        break;
                    }
                }
                if (shard_idx < 0) {
                    fprintf(stderr, "Failed to map expert %d to shard\n", global_e);
                    exit(EXIT_FAILURE);
                }

                auto& sw = shard_work[shard_idx];
                sw.experts.push_back(global_e);
                sw.offsets.push_back(offset);
                sw.counts.push_back(count);
                sw.total += count;
                if (count > sw.maxNe)
                    sw.maxNe = count;
            }

            std::vector<int> local_wq_heavy_h;
            std::vector<int> local_wq_light_h;
            int local_heavy_count = 0;
            int local_light_count = 0;

            if (local_shard_index >= 0 && transformer->ep_local_experts > 0) {
                const int local_start = transformer->ep_expert_offset;
                const int local_span = transformer->ep_local_experts;

                auto filter_bucket = [&](const std::vector<int>& src, std::vector<int>& dst,
                                         int& count) {
                    for (size_t i = 0; i + 2 < src.size(); i += 3) {
                        const int global_e = src[i + 0];
                        if (global_e < local_start || global_e >= local_start + local_span)
                            continue;
                        const int local_e = global_e - local_start;
                        dst.push_back(local_e);
                        dst.push_back(src[i + 1]);
                        dst.push_back(src[i + 2]);
                        ++count;
                    }
                };

                local_wq_heavy_h.reserve(wq_heavy_all.size());
                filter_bucket(wq_heavy_all, local_wq_heavy_h, local_heavy_count);
                local_wq_light_h.reserve(wq_light_all.size());
                filter_bucket(wq_light_all, local_wq_light_h, local_light_count);
            }

            const size_t e_agg_bytes = (size_t)batch_size * hidden_dim * sizeof(float);
            CHECK_HIP(hipMemsetAsync(s->e_agg, 0, e_agg_bytes, tls.compute));

            const long long mlp1_weight_stride = (2LL * p->intermediate_dim) * hidden_dim;
            const long long mlp1_bias_stride = (2LL * p->intermediate_dim);
            const long long mlp2_weight_stride = hidden_dim * (long long)p->intermediate_dim;
            const long long mlp2_bias_stride = hidden_dim;
            const float alpha = 1.702f;

            struct RemoteShardTask {
                int shard_index;
                OssExpertShard* shard;
                hipEvent_t completion_event;
                size_t agg_bytes;
                int remote_device;
                OssExpertWorkspace* workspace;
                const ShardWork* work;
                int assignment_count;
            };

            std::vector<RemoteShardTask> remote_tasks;
            remote_tasks.reserve(ep_size);

            const int dp_rank = transformer->dp_rank >= 0 ? transformer->dp_rank : 0;

            hipStream_t* agg_streams = transformer->ep_aggregate_streams;
            hipEvent_t* agg_events = transformer->ep_aggregate_events;
            float** agg_buffers = transformer->ep_remote_buffers;
            float** weight_buffers = transformer->ep_remote_weight_buffers;
            const size_t agg_capacity_bytes = transformer->ep_remote_buffer_bytes;
            const size_t weight_capacity_bytes = transformer->ep_remote_weight_bytes;

            for (int shard_idx = 0; shard_idx < ep_size; ++shard_idx) {
                if (shard_idx == local_shard_index)
                    continue;

                OssExpertShard* shard = ep_group->shards[shard_idx];
                OssTransformerHybrid* shard_model = shard->model;
                if (!shard_model || shard_model->ep_local_experts == 0)
                    continue;

                const ShardWork& work = shard_work[shard_idx];
                if (work.experts.empty())
                    continue;

                if (dp_rank >= shard->workspace_count) {
                    fprintf(stderr, "dp_rank %d exceeds shard workspace slots %d\n", dp_rank,
                            shard->workspace_count);
                    exit(EXIT_FAILURE);
                }

                OssExpertWorkspace* workspace = &shard->workspaces[dp_rank];
                int spd = shard->streams_per_dp > 0 ? shard->streams_per_dp : 1;
                long long total_streams = (long long)shard->workspace_count * spd;
                long long base_index = (long long)dp_rank * spd;
                long long stream_index =
                    base_index < total_streams ? base_index : (total_streams - 1);
                hipStream_t remote_stream = nullptr;
                if (shard->streams && total_streams > 0 && stream_index >= 0)
                    remote_stream = shard->streams[stream_index];
                if (!remote_stream)
                    remote_stream = 0;
                const int remote_device = shard->device_id;
                HipDeviceGuard remote_device_guard(remote_device);

                const int queue_elems = (int)(work.experts.size() * 3);
                {
                    std::lock_guard<std::mutex> guard(shard_mutex(shard, dp_rank));
                    ensure_workspace_capacity(workspace, remote_device, work.total, hidden_dim,
                                              p->intermediate_dim, batch_size);

                    if (queue_elems > workspace->work_queue_capacity) {
                        if (workspace->work_queue)
                            CHECK_HIP(hipFree(workspace->work_queue));
                        if (queue_elems > 0)
                            CHECK_HIP(hipMalloc(&workspace->work_queue,
                                                (size_t)queue_elems * sizeof(int)));
                        else
                            workspace->work_queue = nullptr;
                        workspace->work_queue_capacity = queue_elems;
                    }
                }

                OssTransformerWeightsBFloat16* rw = &shard_model->weights;

                int remote_local_offset = 0;
                if (queue_elems > 0) {
                    if (workspace->h_work_queue_capacity < queue_elems) {
                        if (workspace->h_work_queue)
                            CHECK_HIP(hipHostFree(workspace->h_work_queue));
                        CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&workspace->h_work_queue),
                                                (size_t)queue_elems * sizeof(int),
                                                hipHostMallocDefault));
                        workspace->h_work_queue_capacity = queue_elems;
                    }

                    int* hq = workspace->h_work_queue;
                    remote_local_offset = 0;
                    for (size_t i = 0; i < work.experts.size(); ++i) {
                        const int global_e = work.experts[i];
                        const int cnt = work.counts[i];
                        const size_t base = i * 3;
                        hq[base + 0] = global_e - shard_model->ep_expert_offset;
                        hq[base + 1] = remote_local_offset;
                        hq[base + 2] = cnt;
                        remote_local_offset += cnt;
                    }

                    CHECK_HIP(hipMemcpyAsync(workspace->work_queue, hq,
                                             (size_t)queue_elems * sizeof(int),
                                             hipMemcpyHostToDevice, remote_stream));
                }

                if (work.total > 0) {
                    float* dispatch_tokens = nullptr;
                    if (agg_buffers && shard_idx < transformer->ep_size)
                        dispatch_tokens = agg_buffers[shard_idx];
                    float* dispatch_weights = nullptr;
                    if (weight_buffers && shard_idx < transformer->ep_size)
                        dispatch_weights = weight_buffers[shard_idx];

                    const size_t tokens_total_bytes =
                        (size_t)work.total * hidden_dim * sizeof(float);
                    const size_t weights_total_bytes = (size_t)work.total * sizeof(float);

                    if (!dispatch_tokens || tokens_total_bytes > agg_capacity_bytes) {
                        fprintf(
                            stderr,
                            "Remote dispatch scratch buffer unavailable or too small (need %zu, "
                            "have %zu)\n",
                            tokens_total_bytes, agg_capacity_bytes);
                        exit(EXIT_FAILURE);
                    }
                    if ((!dispatch_weights && weights_total_bytes > 0) ||
                        weights_total_bytes > weight_capacity_bytes) {
                        fprintf(stderr,
                                "Remote weight scratch buffer unavailable or too small (need %zu, "
                                "have %zu)\n",
                                weights_total_bytes, weight_capacity_bytes);
                        exit(EXIT_FAILURE);
                    }

                    hipStream_t pack_stream = tls.compute;
                    size_t token_elem_offset = 0;
                    size_t weight_elem_offset = 0;
                    for (size_t i = 0; i < work.experts.size(); ++i) {
                        const int cnt = work.counts[i];
                        if (cnt <= 0)
                            continue;
                        const int goff = work.offsets[i];
                        const size_t token_elem_count = (size_t)cnt * hidden_dim;
                        const size_t token_bytes = token_elem_count * sizeof(float);
                        const size_t weight_bytes = (size_t)cnt * sizeof(float);

                        CHECK_HIP(hipMemcpyAsync(dispatch_tokens + token_elem_offset,
                                                 s->x_by_expert + (size_t)goff * hidden_dim,
                                                 token_bytes, hipMemcpyDeviceToDevice,
                                                 pack_stream));

                        if (weight_bytes > 0) {
                            CHECK_HIP(hipMemcpyAsync(dispatch_weights + weight_elem_offset,
                                                     s->weights_flat + goff, weight_bytes,
                                                     hipMemcpyDeviceToDevice, pack_stream));
                        }

                        token_elem_offset += token_elem_count;
                        weight_elem_offset += (size_t)cnt;
                    }
                    CHECK_HIP(hipStreamSynchronize(pack_stream));

                    if (tokens_total_bytes > 0) {
                        CHECK_HIP(hipMemcpyPeerAsync(workspace->x_by_expert, remote_device,
                                                     dispatch_tokens, primary_device,
                                                     tokens_total_bytes, remote_stream));
                    }
                    if (weights_total_bytes > 0) {
                        CHECK_HIP(hipMemcpyPeerAsync(workspace->weights_by_expert, remote_device,
                                                     dispatch_weights, primary_device,
                                                     weights_total_bytes, remote_stream));
                    }
                }

                moe_mlp1_swiglu(
                    workspace->work_queue, 0, (int)work.experts.size(), workspace->x_by_expert,
                    workspace->gate_by_expert,
                    rw->w_mlp1 + 1ll * l * shard_model->ep_local_experts * mlp1_weight_stride,
                    rw->b_mlp1 + 1ll * l * shard_model->ep_local_experts * mlp1_bias_stride,
                    hidden_dim, p->intermediate_dim, mlp1_weight_stride, mlp1_bias_stride, alpha,
                    p->swiglu_limit, work.maxNe, kGridCap, remote_stream);

                moe_mlp2_store(
                    workspace->work_queue, 0, (int)work.experts.size(), workspace->gate_by_expert,
                    workspace->weights_by_expert, workspace->y_by_expert,
                    rw->w_mlp2 + 1ll * l * shard_model->ep_local_experts * mlp2_weight_stride,
                    rw->b_mlp2 + 1ll * l * shard_model->ep_local_experts * mlp2_bias_stride,
                    p->intermediate_dim, hidden_dim, mlp2_weight_stride, mlp2_bias_stride,
                    work.maxNe, kGridCap, remote_stream);

                hipEvent_t completion_event;
                CHECK_HIP(hipEventCreateWithFlags(&completion_event, hipEventDisableTiming));
                CHECK_HIP(hipEventRecord(completion_event, remote_stream));

                const size_t agg_bytes = (size_t)work.total * hidden_dim * sizeof(float);
                remote_tasks.push_back({shard_idx, shard, completion_event, agg_bytes,
                                        remote_device, workspace, &work, work.total});

                CHECK_HIP(hipSetDevice(primary_device));
            }
            CHECK_HIP(hipSetDevice(primary_device));

            const int grid_cap_light = kGridCap;
            const int grid_cap_heavy = std::max(1, kGridCap - std::max(1, kGridCap / 3));

            if (local_shard_index >= 0 && transformer->ep_local_experts > 0) {
                if (local_light_count > 0) {
                    int* d_wq_light = s->d_wq_light;
                    assert(d_wq_light &&
                           (size_t)local_wq_light_h.size() <= (size_t)s->d_wq_light_capacity);
                    CHECK_HIP(hipMemcpyAsync(d_wq_light, local_wq_light_h.data(),
                                             (size_t)local_wq_light_h.size() * sizeof(int),
                                             hipMemcpyHostToDevice, tls.h2d));
                    CHECK_HIP(hipEventRecord(tls.evt_wq_light_ready, tls.h2d));
                    CHECK_HIP(hipStreamWaitEvent(tls.moe_light, tls.evt_wq_light_ready, 0));

                    moe_mlp1_swiglu(
                        d_wq_light, 0, local_light_count, s->x_by_expert, s->gate_by_expert,
                        w->w_mlp1 + 1ll * l * transformer->ep_local_experts * mlp1_weight_stride,
                        w->b_mlp1 + 1ll * l * transformer->ep_local_experts * mlp1_bias_stride,
                        hidden_dim, p->intermediate_dim, mlp1_weight_stride, mlp1_bias_stride,
                        alpha, p->swiglu_limit, rh_light_mlp1, grid_cap_light, tls.moe_light);

                    moe_mlp2_scatter(
                        d_wq_light, 0, local_light_count, s->gate_by_expert, s->tokens_flat,
                        s->weights_flat, s->e_agg,
                        w->w_mlp2 + 1ll * l * transformer->ep_local_experts * mlp2_weight_stride,
                        w->b_mlp2 + 1ll * l * transformer->ep_local_experts * mlp2_bias_stride,
                        p->intermediate_dim, hidden_dim, mlp2_weight_stride, mlp2_bias_stride,
                        rh_light_mlp2, grid_cap_light, tls.moe_light);
                    CHECK_HIP(hipEventRecord(tls.evt_moe_light_done, tls.moe_light));
                }

                if (local_heavy_count > 0) {
                    int* d_wq_heavy = s->d_wq_heavy;
                    assert(d_wq_heavy &&
                           (size_t)local_wq_heavy_h.size() <= (size_t)s->d_wq_heavy_capacity);
                    CHECK_HIP(hipMemcpyAsync(d_wq_heavy, local_wq_heavy_h.data(),
                                             (size_t)local_wq_heavy_h.size() * sizeof(int),
                                             hipMemcpyHostToDevice, tls.h2d));
                    CHECK_HIP(hipEventRecord(tls.evt_wq_heavy_ready, tls.h2d));
                    CHECK_HIP(hipStreamWaitEvent(tls.moe_heavy, tls.evt_wq_heavy_ready, 0));

                    moe_mlp1_swiglu(
                        d_wq_heavy, 0, local_heavy_count, s->x_by_expert, s->gate_by_expert,
                        w->w_mlp1 + 1ll * l * transformer->ep_local_experts * mlp1_weight_stride,
                        w->b_mlp1 + 1ll * l * transformer->ep_local_experts * mlp1_bias_stride,
                        hidden_dim, p->intermediate_dim, mlp1_weight_stride, mlp1_bias_stride,
                        alpha, p->swiglu_limit, rh_heavy_mlp1, grid_cap_heavy, tls.moe_heavy);

                    moe_mlp2_scatter(
                        d_wq_heavy, 0, local_heavy_count, s->gate_by_expert, s->tokens_flat,
                        s->weights_flat, s->e_agg,
                        w->w_mlp2 + 1ll * l * transformer->ep_local_experts * mlp2_weight_stride,
                        w->b_mlp2 + 1ll * l * transformer->ep_local_experts * mlp2_bias_stride,
                        p->intermediate_dim, hidden_dim, mlp2_weight_stride, mlp2_bias_stride,
                        rh_heavy_mlp2, grid_cap_heavy, tls.moe_heavy);
                    CHECK_HIP(hipEventRecord(tls.evt_moe_heavy_done, tls.moe_heavy));
                }
            }

            if (!remote_tasks.empty()) {
                auto resolve_agg_resources = [&](const RemoteShardTask& task, hipStream_t& stream,
                                                 hipEvent_t& event, float*& buffer) {
                    stream = nullptr;
                    event = nullptr;
                    buffer = nullptr;
                    if (task.shard_index < ep_size) {
                        if (agg_streams)
                            stream = agg_streams[task.shard_index];
                        if (agg_events)
                            event = agg_events[task.shard_index];
                        if (agg_buffers)
                            buffer = agg_buffers[task.shard_index];
                    }
                    if (!stream)
                        stream = tls.compute;
                    if (!buffer) {
                        buffer = s->y_by_expert;
                        size_t fallback_bytes = (size_t)batch_size * hidden_dim * sizeof(float);
                        if (task.agg_bytes > fallback_bytes) {
                            fprintf(stderr,
                                    "Remote aggregation buffer insufficient (needed %zu bytes, "
                                    "have %zu)\n",
                                    task.agg_bytes, fallback_bytes);
                            exit(EXIT_FAILURE);
                        }
                    }
                };

#pragma omp parallel for schedule(static)
                for (int task_idx = 0; task_idx < (int)remote_tasks.size(); ++task_idx) {
                    RemoteShardTask& task = remote_tasks[task_idx];
                    CHECK_HIP(hipSetDevice(task.remote_device));
                    CHECK_HIP(hipEventSynchronize(task.completion_event));
                    CHECK_HIP(hipEventDestroy(task.completion_event));

                    CHECK_HIP(hipSetDevice(primary_device));

                    if (task.agg_bytes == 0)
                        continue;

                    hipStream_t agg_stream = nullptr;
                    hipEvent_t agg_event = nullptr;
                    float* agg_buffer = nullptr;
                    resolve_agg_resources(task, agg_stream, agg_event, agg_buffer);

                    CHECK_HIP(hipMemcpyPeerAsync(agg_buffer, primary_device,
                                                 task.workspace->y_by_expert, task.remote_device,
                                                 task.agg_bytes, agg_stream));

                    if (agg_event)
                        CHECK_HIP(hipEventRecord(agg_event, agg_stream));
                }

                if (local_heavy_count > 0)
                    CHECK_HIP(hipStreamWaitEvent(tls.compute, tls.evt_moe_heavy_done, 0));
                if (local_light_count > 0)
                    CHECK_HIP(hipStreamWaitEvent(tls.compute, tls.evt_moe_light_done, 0));

                for (size_t task_idx = 0; task_idx < remote_tasks.size(); ++task_idx) {
                    RemoteShardTask& task = remote_tasks[task_idx];
                    if (task.agg_bytes == 0 || task.assignment_count <= 0)
                        continue;

                    hipStream_t agg_stream = nullptr;
                    hipEvent_t agg_event = nullptr;
                    float* agg_buffer = nullptr;
                    resolve_agg_resources(task, agg_stream, agg_event, agg_buffer);

                    if (agg_event && agg_stream != tls.compute)
                        CHECK_HIP(hipStreamWaitEvent(tls.compute, agg_event, 0));
                    else if (agg_stream && agg_stream != tls.compute)
                        CHECK_HIP(hipStreamSynchronize(agg_stream));

                    const ShardWork* sw = task.work;
                    if (!sw)
                        continue;
                    size_t assignment_offset = 0;
                    for (size_t i = 0; i < sw->experts.size(); ++i) {
                        int cnt = sw->counts[i];
                        int off = sw->offsets[i];
                        if (cnt <= 0)
                            continue;
                        const float* shard_vals =
                            agg_buffer + assignment_offset * (size_t)hidden_dim;
                        accumulate_remote_assignments(shard_vals, s->tokens_flat, off, cnt,
                                                      hidden_dim, tls.compute, s->e_agg);
                        assignment_offset += cnt;
                    }
                }
            }
            CHECK_HIP(hipSetDevice(primary_device));

            if (local_heavy_count > 0)
                CHECK_HIP(hipStreamWaitEvent(tls.compute, tls.evt_moe_heavy_done, 0));
            if (local_light_count > 0)
                CHECK_HIP(hipStreamWaitEvent(tls.compute, tls.evt_moe_light_done, 0));

            // ! Residual
            vecadd_and_zero(x, s->e_agg, 1.0f, batch_size, hidden_dim, tls.compute);
        }
    }

    auto launch_output_stage = [&](bool use_graph) {
        if (!use_graph) {
            rmsnorm(x, x, w->rms_out_w, batch_size, hidden_dim, tls.compute);
            gemm_logits(s->logits, x, w->out, batch_size, hidden_dim, p->vocab_size, tls.compute);
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

            CHECK_HIP(hipStreamBeginCapture(tls.compute, hipStreamCaptureModeThreadLocal));
            rmsnorm(x, x, w->rms_out_w, batch_size, hidden_dim, tls.compute);
            gemm_logits(s->logits, x, w->out, batch_size, hidden_dim, p->vocab_size, tls.compute);
            hipGraph_t graph = nullptr;
            CHECK_HIP(hipStreamEndCapture(tls.compute, &graph));
            CHECK_HIP(hipGraphInstantiate(&s->forward_graph_exec, graph, nullptr, nullptr, 0));
            s->forward_graph = graph;
            s->forward_graph_ready = true;
            s->forward_graph_batch_size = batch_size;
        }

        CHECK_HIP(hipGraphLaunch(s->forward_graph_exec, tls.compute));
    };

    launch_output_stage(s->forward_graph_enabled);

    CHECK_HIP(hipStreamSynchronize(tls.compute));

    return s->logits;
}
