#include "BLAS.hip"

__global__ void build_expert_work_queue_kernel(
    const int* __restrict__ expert_offsets,
    int* __restrict__ work_queue,
    Int2* __restrict__ meta,
    int E)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int active = 0;
    int maxNe  = 0;
    for (int e = 0; e < E; ++e) {
        int off = expert_offsets[e];
        int ne  = expert_offsets[e + 1] - off;
        if (ne > 0) {
            int i3 = active * 3;
            work_queue[i3 + 0] = e;
            work_queue[i3 + 1] = off;
            work_queue[i3 + 2] = ne;
            ++active;
            if (ne > maxNe) maxNe = ne;
        }
    }
    meta[0].x = active;
    meta[0].y = maxNe;
}

__global__ void build_assignments_and_count_kernel(
    const int* __restrict__ topk_i,
    const float* __restrict__ topk_v,
    int B, int K,
    int* __restrict__ out_expert,
    int* __restrict__ out_token,
    float* __restrict__ out_w,
    int* __restrict__ counts,
    int E)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int BK = B * K;
    if (idx >= BK) return;

    int b = idx / K;
    int k = idx % K;

    int e = topk_i[b*K + k];
    out_expert[idx] = e;
    out_token[idx]  = b;
    out_w[idx]      = topk_v[b*K + k];

    if ((unsigned)e < (unsigned)E) atomicAdd(&counts[e], 1);
}

__global__ void exclusive_scan_small_kernel(
    const int* __restrict__ counts, int* __restrict__ offsets, int E)
{
    if (threadIdx.x == 0) {
        int acc = 0;
        offsets[0] = 0;
        #pragma unroll
        for (int i = 0; i < E; ++i) {
            int c = counts[i];
            acc += c;
            offsets[i+1] = acc;
        }
    }
}
__global__ void compact_by_expert_kernel(const int* __restrict__ expert_ids,
                                         const int* __restrict__ tokens,
                                         const float* __restrict__ weights,
                                         int BK, const int* __restrict__ offsets,
                                         int* __restrict__ write_counters,
                                         int* __restrict__ tokens_flat,
                                         float* __restrict__ weights_flat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BK) return;
    int e = expert_ids[i];
    int pos = atomicAdd(&write_counters[e], 1);
    int dst = offsets[e] + pos;
    tokens_flat[dst]  = tokens[i];
    weights_flat[dst] = weights[i];
}

__global__ void gather_rows_vec4_kernel(const float* __restrict__ X,
                                        const int*  __restrict__ idx,
                                        float* __restrict__ Y,
                                        int BK, int H4)
{
    int row = blockIdx.y;
    int col4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= BK || col4 >= H4) return;

    int b = idx[row];
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(X);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(Y);

    y4[row * H4 + col4] = x4[b * H4 + col4];
}

__global__ void gather_rows_kernel(const float* __restrict__ X,
                                   const int*  __restrict__ idx,
                                   float* __restrict__ Y,
                                   int BK, int H)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= BK || col >= H) return;
    int b = idx[row];
    Y[row*H + col] = X[b*H + col];
}
