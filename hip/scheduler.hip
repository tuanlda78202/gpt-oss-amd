#include "BLAS.hip"

struct MoEStats {
    int heavy_count;
    int light_count;
    int heavy_maxNe;
    int active_experts;
    long long sumNe;
};

__global__ void count_tokens_per_expert(const int* __restrict__ topk_i,
                                        const float* __restrict__ topk_v, int B, int K,
                                        int* __restrict__ out_expert, int* __restrict__ out_token,
                                        float* __restrict__ out_w, int* __restrict__ counts,
                                        int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int BK = B * K;
    if (idx >= BK)
        return;

    int b = idx / K;
    int k = idx % K;

    int e = topk_i[b * K + k];
    out_expert[idx] = e;
    out_token[idx] = b;
    out_w[idx] = topk_v[b * K + k];

    if ((unsigned)e < (unsigned)E)
        atomicAdd(&counts[e], 1);
}

__global__ void exclusive_scan_expert_offsets(int* __restrict__ counts, int* __restrict__ offsets,
                                              int E) {
    if (threadIdx.x == 0) {
        int acc = 0;
        offsets[0] = 0;
#pragma unroll
        for (int i = 0; i < E; ++i) {
            int c = counts[i];
            acc += c;
            offsets[i + 1] = acc;
            counts[i] = 0;
        }
    }
}

__global__ void
compact_by_expert_kernel(const int* __restrict__ expert_ids, const int* __restrict__ tokens,
                         const float* __restrict__ weights, int BK, const int* __restrict__ offsets,
                         int* __restrict__ write_counters, int* __restrict__ tokens_flat,
                         float* __restrict__ weights_flat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BK)
        return;
    int e = expert_ids[i];
    int pos = atomicAdd(&write_counters[e], 1);
    int dst = offsets[e] + pos;
    tokens_flat[dst] = tokens[i];
    weights_flat[dst] = weights[i];
}

__global__ void gather_rows_vec4_kernel(const float* __restrict__ X, const int* __restrict__ idx,
                                        float* __restrict__ Y, int BK, int H4) {
    int row = blockIdx.y;
    int col4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= BK || col4 >= H4)
        return;

    int b = idx[row];
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(X);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(Y);

    y4[row * H4 + col4] = x4[b * H4 + col4];
}

__global__ void gather_rows_kernel(const float* __restrict__ X, const int* __restrict__ idx,
                                   float* __restrict__ Y, int BK, int H) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= BK || col >= H)
        return;
    int b = idx[row];
    Y[row * H + col] = X[b * H + col];
}

__global__ void build_expert_work_queue(const int* __restrict__ expert_offsets,
                                        int* __restrict__ work_queue, Int2* __restrict__ meta,
                                        int E) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    int active = 0;
    int maxNe = 0;
    for (int e = 0; e < E; ++e) {
        int off = expert_offsets[e];
        int ne = expert_offsets[e + 1] - off;
        if (ne > 0) {
            int i3 = active * 3;
            work_queue[i3 + 0] = e;
            work_queue[i3 + 1] = off;
            work_queue[i3 + 2] = ne;
            ++active;
            if (ne > maxNe)
                maxNe = ne;
        }
    }
    meta[0].x = active;
    meta[0].y = maxNe;
}


__global__ void classify_and_build_queues(const int* __restrict__ expert_offsets, int E,
                                          int very_sparse_cutoff,     // ADAPT_VERY_SPARSE_CUTOFF
                                          float heavy_factor_sparse,  // ADAPT_HEAVY_FACTOR_SPARSE
                                          float heavy_factor_default, // ADAPT_HEAVY_FACTOR_DEFAULT
                                          int* __restrict__ wq_heavy, // [3*E]
                                          int* __restrict__ wq_light, // [3*E]
                                          MoEStats* __restrict__ stats_out // single struct
) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    int active = 0;
    int maxNe = 0;
    long long sumNe_ll = 0;
    for (int e = 0; e < E; ++e) {
        int off = expert_offsets[e];
        int ne = expert_offsets[e + 1] - off;
        if (ne > 0) {
            ++active;
            sumNe_ll += ne;
            if (ne > maxNe)
                maxNe = ne;
        }
    }
    if (active == 0) {
        stats_out->heavy_count = 0;
        stats_out->light_count = 0;
        stats_out->heavy_maxNe = 0;
        stats_out->active_experts = 0;
        stats_out->sumNe = 0;
        return;
    }

    float avgNe = float(sumNe_ll) / float(active);
    float heavy_factor = heavy_factor_default;
    if (active <= very_sparse_cutoff) {
        heavy_factor = 1.0f;
    } else if (active <= 12) {
        heavy_factor = heavy_factor_sparse;
    }
    const int heavy_thresh = int(ceilf(heavy_factor * avgNe));

    int heavy_count = 0;
    int light_count = 0;
    int heavy_max = 0;
    for (int e = 0; e < E; ++e) {
        int off = expert_offsets[e];
        int ne = expert_offsets[e + 1] - off;
        if (ne <= 0)
            continue;
        if (active <= very_sparse_cutoff || ne >= heavy_thresh) {
            int i3 = heavy_count * 3;
            wq_heavy[i3 + 0] = e;
            wq_heavy[i3 + 1] = off;
            wq_heavy[i3 + 2] = ne;
            ++heavy_count;
            if (ne > heavy_max)
                heavy_max = ne;
        } else {
            int i3 = light_count * 3;
            wq_light[i3 + 0] = e;
            wq_light[i3 + 1] = off;
            wq_light[i3 + 2] = ne;
            ++light_count;
        }
    }

    stats_out->heavy_count = heavy_count;
    stats_out->light_count = light_count;
    stats_out->heavy_maxNe = heavy_max;
    stats_out->active_experts = active;
    stats_out->sumNe = sumNe_ll;
}
