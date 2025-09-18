#include "../include/model_120b.hpp"
#include "hip/BLAS.hip"
#include "hip/attention.hip"
#include "hip/embed.hip"
#include "hip/matvec.hip"
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
#include <mutex>
#include <omp.h>
#include <vector>

namespace {

void ensure_workspace_capacity(OssExpertShard* shard, int required_tokens, int hidden_dim,
                               int intermediate_dim) {
    if (required_tokens <= shard->workspace.capacity_tokens)
        return;

    CHECK_HIP(hipSetDevice(shard->device_id));

    if (shard->workspace.x_by_expert)
        CHECK_HIP(hipFree(shard->workspace.x_by_expert));
    if (shard->workspace.mlp1_by_expert)
        CHECK_HIP(hipFree(shard->workspace.mlp1_by_expert));
    if (shard->workspace.gate_by_expert)
        CHECK_HIP(hipFree(shard->workspace.gate_by_expert));
    if (shard->workspace.y_by_expert)
        CHECK_HIP(hipFree(shard->workspace.y_by_expert));

    size_t tokens_bytes = (size_t)required_tokens * hidden_dim * sizeof(float);
    size_t mlp1_bytes = (size_t)required_tokens * 2 * intermediate_dim * sizeof(float);
    size_t gate_bytes = (size_t)required_tokens * intermediate_dim * sizeof(float);

    CHECK_HIP(hipMalloc(&shard->workspace.x_by_expert, tokens_bytes));
    CHECK_HIP(hipMalloc(&shard->workspace.mlp1_by_expert, mlp1_bytes));
    CHECK_HIP(hipMalloc(&shard->workspace.gate_by_expert, gate_bytes));
    CHECK_HIP(hipMalloc(&shard->workspace.y_by_expert, tokens_bytes));

    shard->workspace.capacity_tokens = required_tokens;
}

std::mutex& shard_mutex(OssExpertShard* shard) {
    auto* mtx = reinterpret_cast<std::mutex*>(shard->mutex_handle);
    return *mtx;
}

} // namespace

float* forward(OssTransformerHybrid* transformer, OssExpertParallelGroup* ep_group, int* tokens,
               const int* pos_per_token_h, int batch_size, const int* batch_indices_h,
               int B_stride) {
    hipEvent_t start_total, end_total;
    hipEvent_t start_section, end_section;
    hipEvent_t start_moe_sub, end_moe_sub;

    if (g_enable_profiling) {
        CHECK_HIP(hipEventCreate(&start_total));
        CHECK_HIP(hipEventCreate(&end_total));
        CHECK_HIP(hipEventCreate(&start_section));
        CHECK_HIP(hipEventCreate(&end_section));
        CHECK_HIP(hipEventCreate(&start_moe_sub));
        CHECK_HIP(hipEventCreate(&end_moe_sub));
        CHECK_HIP(hipEventRecord(start_total, 0));
        CHECK_HIP(hipEventRecord(start_section, 0));
    }

    OssConfig* p = &transformer->config;
    OssTransformerWeightsHalf* w = &transformer->weights;
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

    CHECK_HIP(hipMemcpy(s->d_batch_indices, batch_indices_h, batch_size * sizeof(int),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s->d_tokens, tokens, batch_size * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s->d_pos_per_token, pos_per_token_h, batch_size * sizeof(int),
                        hipMemcpyHostToDevice));

    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(end_section, 0));
        CHECK_HIP(hipEventSynchronize(end_section));
        float setup_ms;
        CHECK_HIP(hipEventElapsedTime(&setup_ms, start_section, end_section));
        g_batch_timings.setup_time += setup_ms;
    }

    // ! Embedding
    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(start_section, 0));
    }
    embed(x, w->token_embedding_table, s->d_tokens, batch_size, hidden_dim);
    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(end_section, 0));
        CHECK_HIP(hipEventSynchronize(end_section));
        float embedding_ms;
        CHECK_HIP(hipEventElapsedTime(&embedding_ms, start_section, end_section));
        g_batch_timings.embedding_time += embedding_ms;
    }

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // ! RMSNorm
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        rmsnorm(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float rmsnorm_ms;
            CHECK_HIP(hipEventElapsedTime(&rmsnorm_ms, start_section, end_section));
            g_batch_timings.layer_rmsnorm_time += rmsnorm_ms;
        }

        // ! QKV project
        __half* w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                       (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        __half* b_qkv =
            w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        matvec(s->qkv, s->t, w_qkv, b_qkv, batch_size, hidden_dim,
               (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float qkv_ms;
            CHECK_HIP(hipEventElapsedTime(&qkv_ms, start_section, end_section));
            g_batch_timings.qkv_projection_time += qkv_ms;
        }

        // ! Split QKV
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        split_qkv(s->qkv, s->q, s->key_cache, s->value_cache, batch_size, head_dim, p->n_attn_heads,
                  p->n_kv_heads, l, s->d_pos_per_token, p->seq_len, s->d_batch_indices, B_stride,
                  /*stream=*/0, kv_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float split_qkv_ms;
            CHECK_HIP(hipEventElapsedTime(&split_qkv_ms, start_section, end_section));
            g_batch_timings.split_qkv_time += split_qkv_ms;
        }

        // ! RoPE (Q and K)
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        {
            const float ntk_beta = 32.0f;
            const float ntk_alpha = 1.0f;
            rope_qk(s->q, s->key_cache, batch_size, p->n_attn_heads, p->n_kv_heads, head_dim,
                    p->seq_len, kv_dim, l, s->d_pos_per_token, s->d_batch_indices, B_stride,
                    p->rope_theta, p->rope_scaling_factor, p->initial_context_length, ntk_beta,
                    ntk_alpha,
                    /*stream=*/0);
        }
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float rope_ms;
            CHECK_HIP(hipEventElapsedTime(&rope_ms, start_section, end_section));
            g_batch_timings.rope_time += rope_ms;
        }

        // ! Attention
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        fa(s->q, (const void*)s->key_cache, (const void*)s->value_cache, s->mask, w->attn_sinks,
           s->tb, batch_size, p->seq_len, head_dim, kv_dim, kv_mul, p->sliding_window, (int)l,
           p->n_attn_heads, s->d_pos_per_token, s->d_batch_indices, (long long)B_stride,
           max_pos_in_batch, 0);

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float attn_ms;
            CHECK_HIP(hipEventElapsedTime(&attn_ms, start_section, end_section));
            g_batch_timings.attention_time += attn_ms;
        }

        // ! Output projection
        __half* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        __half* b_o = w->b_o + 1ll * l * hidden_dim;

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        matvec(s->tb2, s->tb, w_o, b_o, batch_size, head_dim * p->n_attn_heads, hidden_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float out_proj_ms;
            CHECK_HIP(hipEventElapsedTime(&out_proj_ms, start_section, end_section));
            g_batch_timings.output_projection_time += out_proj_ms;
        }

        // ! Residual
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        vecadd(x, s->tb2, 1.0f, batch_size, hidden_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float residual_ms;
            CHECK_HIP(hipEventElapsedTime(&residual_ms, start_section, end_section));
            g_batch_timings.residual_time += residual_ms;
        }

        // ! RMSNorm
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        rmsnorm(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float mlp_rmsnorm_ms;
            CHECK_HIP(hipEventElapsedTime(&mlp_rmsnorm_ms, start_section, end_section));
            g_batch_timings.mlp_rmsnorm_time += mlp_rmsnorm_ms;
        }

        // ! MoE Router
        __half* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        __half* b_router = w->b_router + 1ll * l * n_experts;

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        matvec(s->router_score, s->t, w_router, b_router, batch_size, hidden_dim, n_experts);

        // ! TopK and Softmax
        topk(s->topk_v, s->topk_i, s->router_score, batch_size, n_experts, p->experts_per_token);

        softmax(s->topk_v, batch_size, p->experts_per_token);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float moe_routing_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_routing_ms, start_section, end_section));
            g_batch_timings.moe_routing_time += moe_routing_ms;
        }

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }

        // ! MoE Setup - Zero aggregation
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }
        CHECK_HIP(hipMemset(s->e_agg, 0, batch_size * hidden_dim * sizeof(float)));
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_setup_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_setup_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_setup_time += moe_setup_ms;
        }

        const int BKE = batch_size * p->experts_per_token;
        dim3 blk1(256), grd1((BKE + blk1.x - 1) / blk1.x);

        // ! Build assignments + counts
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }
        CHECK_HIP(hipMemset(s->expert_counts, 0, n_experts * sizeof(int)));
        hipLaunchKernelGGL(build_assignments_and_count_kernel, grd1, blk1, 0, 0, s->topk_i,
                           s->topk_v, batch_size, p->experts_per_token, s->assign_expert,
                           s->assign_token, s->assign_weight, s->expert_counts, n_experts);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_assignment_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_assignment_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_assignment_time += moe_assignment_ms;
        }

        // ! Exclusive scan on device -> expert_offsets[0..E]
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }
        hipLaunchKernelGGL(exclusive_scan_small_kernel, dim3(1), dim3(1), 0, 0, s->expert_counts,
                           s->expert_offsets, n_experts);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_scan_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_scan_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_scan_time += moe_scan_ms;
        }

        // ! Compact by expert
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }
        CHECK_HIP(hipMemset(s->expert_counts, 0, n_experts * sizeof(int)));
        hipLaunchKernelGGL(compact_by_expert_kernel, grd1, blk1, 0, 0, s->assign_expert,
                           s->assign_token, s->assign_weight, BKE, s->expert_offsets,
                           s->expert_counts, s->tokens_flat, s->weights_flat);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_compact_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_compact_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_compact_time += moe_compact_ms;
        }

        // ! Gather inputs X -> x_by_expert
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }
        {
            const bool vecOK = ((reinterpret_cast<uintptr_t>(s->t) & 0xF) == 0) &&
                               ((reinterpret_cast<uintptr_t>(s->x_by_expert) & 0xF) == 0) &&
                               (hidden_dim % 4 == 0);
            if (vecOK) {
                const int H4 = hidden_dim >> 2;
                dim3 blk(256, 1, 1), grd((H4 + 255) / 256, BKE);
                hipLaunchKernelGGL(gather_rows_vec4_kernel, grd, blk, 0, 0, s->t, s->tokens_flat,
                                   s->x_by_expert, BKE, H4);
            } else {
                dim3 blk(256, 1, 1), grd((hidden_dim + 255) / 256, BKE);
                hipLaunchKernelGGL(gather_rows_kernel, grd, blk, 0, 0, s->t, s->tokens_flat,
                                   s->x_by_expert, BKE, hidden_dim);
            }
        }
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_gather_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_gather_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_gather_time += moe_gather_ms;
        }

        // ! === Build work queue and read back {active_experts, max_Ne} ===
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }

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

        hipLaunchKernelGGL(build_expert_work_queue_kernel, dim3(1), dim3(1), 0, 0,
                           s->expert_offsets, d_work_queue, d_meta, n_experts);

        Int2 h_meta;
        CHECK_HIP(hipMemcpy(&h_meta, d_meta, sizeof(Int2), hipMemcpyDeviceToHost));
        const int active_experts = h_meta.x;
        const int max_Ne = h_meta.y;

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_queue_build_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_queue_build_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_queue_build_time += moe_queue_build_ms;
        }

        std::vector<int> h_work_queue;
        if (active_experts > 0) {
            h_work_queue.resize(3 * active_experts);
            CHECK_HIP(hipMemcpy(h_work_queue.data(), d_work_queue,
                                h_work_queue.size() * sizeof(int), hipMemcpyDeviceToHost));
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

        CHECK_HIP(hipMemset(s->e_agg, 0, batch_size * hidden_dim * sizeof(float)));

        const long long mlp1_weight_stride = (2LL * p->intermediate_dim) * hidden_dim;
        const long long mlp1_bias_stride = (2LL * p->intermediate_dim);
        const long long mlp2_weight_stride = hidden_dim * (long long)p->intermediate_dim;
        const long long mlp2_bias_stride = hidden_dim;
        const float alpha = 1.702f;

        float accumulated_expert_time = 0.0f;

        if (local_shard_index >= 0 && transformer->ep_local_experts > 0) {
            const ShardWork& local_work = shard_work[local_shard_index];
            if (!local_work.experts.empty()) {
                std::vector<int> local_queue;
                local_queue.reserve(local_work.experts.size() * 3);
                for (size_t i = 0; i < local_work.experts.size(); ++i) {
                    local_queue.push_back(local_work.experts[i] - transformer->ep_expert_offset);
                    local_queue.push_back(local_work.offsets[i]);
                    local_queue.push_back(local_work.counts[i]);
                }

                size_t queue_bytes = local_queue.size() * sizeof(int);
                CHECK_HIP(hipMemcpy(d_work_queue, local_queue.data(), queue_bytes,
                                    hipMemcpyHostToDevice));

                float mlp1_ms = 0.0f, swiglu_ms = 0.0f, mlp2_ms = 0.0f, scatter_ms = 0.0f;

                if (g_enable_profiling) {
                    CHECK_HIP(hipEventRecord(start_moe_sub, 0));
                }
                moe_matvec(d_work_queue, 0, (int)local_work.experts.size(), s->x_by_expert,
                           s->mlp1_by_expert,
                           w->w_mlp1 + 1ll * l * transformer->ep_local_experts * mlp1_weight_stride,
                           w->b_mlp1 + 1ll * l * transformer->ep_local_experts * mlp1_bias_stride,
                           hidden_dim, 2 * p->intermediate_dim, mlp1_weight_stride,
                           mlp1_bias_stride, local_work.maxNe, 0);
                if (g_enable_profiling) {
                    CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                    CHECK_HIP(hipEventSynchronize(end_moe_sub));
                    CHECK_HIP(hipEventElapsedTime(&mlp1_ms, start_moe_sub, end_moe_sub));
                    g_batch_timings.moe_mlp1_time += mlp1_ms;
                }

                if (g_enable_profiling) {
                    CHECK_HIP(hipEventRecord(start_moe_sub, 0));
                }
                moe_split_swiglu(d_work_queue, 0, (int)local_work.experts.size(), s->mlp1_by_expert,
                                 s->gate_by_expert, p->intermediate_dim, alpha, p->swiglu_limit,
                                 local_work.maxNe, 0);
                if (g_enable_profiling) {
                    CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                    CHECK_HIP(hipEventSynchronize(end_moe_sub));
                    CHECK_HIP(hipEventElapsedTime(&swiglu_ms, start_moe_sub, end_moe_sub));
                    g_batch_timings.moe_swiglu_time += swiglu_ms;
                }

                if (g_enable_profiling) {
                    CHECK_HIP(hipEventRecord(start_moe_sub, 0));
                }
                moe_matvec(d_work_queue, 0, (int)local_work.experts.size(), s->gate_by_expert,
                           s->y_by_expert,
                           w->w_mlp2 + 1ll * l * transformer->ep_local_experts * mlp2_weight_stride,
                           w->b_mlp2 + 1ll * l * transformer->ep_local_experts * mlp2_bias_stride,
                           p->intermediate_dim, hidden_dim, mlp2_weight_stride, mlp2_bias_stride,
                           local_work.maxNe, 0);
                if (g_enable_profiling) {
                    CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                    CHECK_HIP(hipEventSynchronize(end_moe_sub));
                    CHECK_HIP(hipEventElapsedTime(&mlp2_ms, start_moe_sub, end_moe_sub));
                    g_batch_timings.moe_mlp2_time += mlp2_ms;
                }

                if (g_enable_profiling) {
                    CHECK_HIP(hipEventRecord(start_moe_sub, 0));
                }
                moe_scale_scatter(d_work_queue, 0, (int)local_work.experts.size(), s->y_by_expert,
                                  s->tokens_flat, s->weights_flat, s->e_agg, hidden_dim, batch_size,
                                  local_work.maxNe, 0);
                if (g_enable_profiling) {
                    CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                    CHECK_HIP(hipEventSynchronize(end_moe_sub));
                    CHECK_HIP(hipEventElapsedTime(&scatter_ms, start_moe_sub, end_moe_sub));
                    g_batch_timings.moe_scatter_time += scatter_ms;
                    accumulated_expert_time += (mlp1_ms + swiglu_ms + mlp2_ms + scatter_ms);
                }
            }
        }

        // Remote shards (including local shard if it hosts no experts)
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

            int remote_device = shard->device_id;
            OssTransformerWeightsHalf* rw = &shard_model->weights;

            std::vector<int> remote_queue;
            remote_queue.reserve(work.experts.size() * 3);
            std::vector<int> agg_queue;
            agg_queue.reserve(work.experts.size() * 3);

            int local_offset = 0;
            for (size_t i = 0; i < work.experts.size(); ++i) {
                int global_e = work.experts[i];
                int count = work.counts[i];
                int global_offset = work.offsets[i];

                remote_queue.push_back(global_e - shard_model->ep_expert_offset);
                remote_queue.push_back(local_offset);
                remote_queue.push_back(count);

                agg_queue.push_back(0);
                agg_queue.push_back(global_offset);
                agg_queue.push_back(count);

                local_offset += count;
            }

            std::lock_guard<std::mutex> guard(shard_mutex(shard));
            ensure_workspace_capacity(shard, work.total, hidden_dim, p->intermediate_dim);

            CHECK_HIP(hipSetDevice(remote_device));
            int required_remote_ints = (int)remote_queue.size();
            if (required_remote_ints > shard_model->ep_work_queue_capacity) {
                if (shard_model->ep_work_queue)
                    CHECK_HIP(hipFree(shard_model->ep_work_queue));
                CHECK_HIP(
                    hipMalloc(&shard_model->ep_work_queue, required_remote_ints * sizeof(int)));
                shard_model->ep_work_queue_capacity = required_remote_ints;
            }
            CHECK_HIP(hipMemcpy(shard_model->ep_work_queue, remote_queue.data(),
                                remote_queue.size() * sizeof(int), hipMemcpyHostToDevice));

            local_offset = 0;
            for (size_t i = 0; i < work.experts.size(); ++i) {
                int count = work.counts[i];
                int global_offset = work.offsets[i];
                size_t x_bytes = (size_t)count * hidden_dim * sizeof(float);
                CHECK_HIP(hipMemcpyPeer(
                    shard->workspace.x_by_expert + (long long)local_offset * hidden_dim,
                    remote_device, s->x_by_expert + (long long)global_offset * hidden_dim,
                    primary_device, x_bytes));
                local_offset += count;
            }

            moe_matvec(shard_model->ep_work_queue, 0, (int)work.experts.size(),
                       shard->workspace.x_by_expert, shard->workspace.mlp1_by_expert,
                       rw->w_mlp1 + 1ll * l * shard_model->ep_local_experts * mlp1_weight_stride,
                       rw->b_mlp1 + 1ll * l * shard_model->ep_local_experts * mlp1_bias_stride,
                       hidden_dim, 2 * p->intermediate_dim, mlp1_weight_stride, mlp1_bias_stride,
                       work.maxNe, 0);
            moe_split_swiglu(shard_model->ep_work_queue, 0, (int)work.experts.size(),
                             shard->workspace.mlp1_by_expert, shard->workspace.gate_by_expert,
                             p->intermediate_dim, alpha, p->swiglu_limit, work.maxNe, 0);
            moe_matvec(shard_model->ep_work_queue, 0, (int)work.experts.size(),
                       shard->workspace.gate_by_expert, shard->workspace.y_by_expert,
                       rw->w_mlp2 + 1ll * l * shard_model->ep_local_experts * mlp2_weight_stride,
                       rw->b_mlp2 + 1ll * l * shard_model->ep_local_experts * mlp2_bias_stride,
                       p->intermediate_dim, hidden_dim, mlp2_weight_stride, mlp2_bias_stride,
                       work.maxNe, 0);

            local_offset = 0;
            for (size_t i = 0; i < work.experts.size(); ++i) {
                int count = work.counts[i];
                int global_offset = work.offsets[i];
                size_t bytes = (size_t)count * hidden_dim * sizeof(float);
                CHECK_HIP(hipMemcpyPeer(
                    s->y_by_expert + (long long)global_offset * hidden_dim, primary_device,
                    shard->workspace.y_by_expert + (long long)local_offset * hidden_dim,
                    remote_device, bytes));
                local_offset += count;
            }

            CHECK_HIP(hipSetDevice(primary_device));

            if (!agg_queue.empty()) {
                size_t bytes = agg_queue.size() * sizeof(int);
                CHECK_HIP(hipMemcpy(d_work_queue, agg_queue.data(), bytes, hipMemcpyHostToDevice));

                if (g_enable_profiling) {
                    CHECK_HIP(hipEventRecord(start_moe_sub, 0));
                }
                moe_scale_scatter(d_work_queue, 0, (int)work.experts.size(), s->y_by_expert,
                                  s->tokens_flat, s->weights_flat, s->e_agg, hidden_dim, batch_size,
                                  work.maxNe, 0);
                if (g_enable_profiling) {
                    CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                    CHECK_HIP(hipEventSynchronize(end_moe_sub));
                    float scatter_ms_remote;
                    CHECK_HIP(hipEventElapsedTime(&scatter_ms_remote, start_moe_sub, end_moe_sub));
                    g_batch_timings.moe_scatter_time += scatter_ms_remote;
                    accumulated_expert_time += scatter_ms_remote;
                }
            }
        }

        if (g_enable_profiling) {
            g_batch_timings.expert_processing_time += accumulated_expert_time;
        }

        // ! Residual
        vecadd(x, s->e_agg, 1.0f, batch_size, hidden_dim);
    }

    // ! RMSNorm
    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(start_section, 0));
    }
    rmsnorm(x, x, w->rms_out_w, batch_size, hidden_dim);

    // ! Linear logits
    matvec(s->logits, x, w->out, nullptr, batch_size, hidden_dim, p->vocab_size);
    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(end_section, 0));
        CHECK_HIP(hipEventSynchronize(end_section));
        float final_ops_ms;
        CHECK_HIP(hipEventElapsedTime(&final_ops_ms, start_section, end_section));
        g_batch_timings.final_ops_time += final_ops_ms;
    }

    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(end_total, 0));
        CHECK_HIP(hipEventSynchronize(end_total));
        float total_ms;
        CHECK_HIP(hipEventElapsedTime(&total_ms, start_total, end_total));
        g_batch_timings.total_time += total_ms;
        g_batch_timings.num_calls++;
        g_batch_timings.total_layers += p->n_layers;

        CHECK_HIP(hipEventDestroy(start_total));
        CHECK_HIP(hipEventDestroy(end_total));
        CHECK_HIP(hipEventDestroy(start_section));
        CHECK_HIP(hipEventDestroy(end_section));
        CHECK_HIP(hipEventDestroy(start_moe_sub));
        CHECK_HIP(hipEventDestroy(end_moe_sub));
    }

    return s->logits;
}
