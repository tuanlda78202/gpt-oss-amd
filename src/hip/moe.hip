#include "BLAS.hip"

// Build (B*K) assignment triples from (B,K) topk results
__global__ void build_assignments_kernel(const int* __restrict__ topk_i,
                                         const float* __restrict__ topk_v,
                                         int B, int K,
                                         int* out_expert, int* out_token, float* out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int BK = B * K;
    if (idx >= BK) return;
    int b = idx / K, k = idx % K;
    out_expert[idx] = topk_i[b*K + k];
    out_token[idx]  = b;
    out_w[idx]      = topk_v[b*K + k];
}

// Count tokens per expert
__global__ void count_by_expert_kernel(const int* __restrict__ expert_ids, int BK,
                                       int* __restrict__ counts, int E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BK) return;
    int e = expert_ids[i];
    if (e >= 0 && e < E) atomicAdd(&counts[e], 1);
}

// Compact assignments into expert-grouped flat lists
__global__ void compact_by_expert_kernel(const int* __restrict__ expert_ids,
                                         const int* __restrict__ tokens,
                                         const float* __restrict__ weights,
                                         int BK,
                                         const int* __restrict__ offsets,
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

// Gather input rows by token index
__global__ void gather_rows_kernel(const float* __restrict__ X, // [B,H]
                                   const int*  __restrict__ idx,// [BK]
                                   float* __restrict__ Y,       // [BK,H]
                                   int BK, int H) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= BK || col >= H) return;
    int b = idx[row];
    Y[row*H + col] = X[b*H + col];
}

// Scale rows by per-row weight
__global__ void scale_rows_kernel(float* __restrict__ Y, const float* __restrict__ w,
                                  int rows, int H) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= H) return;
    Y[row*H + col] *= w[row];
}

// Scatter-add back to aggregation buffer
__global__ void scatter_add_rows_kernel(float* __restrict__ Eagg, // [B,H]
                                        const float* __restrict__ Y, // [M,H]
                                        const int*   __restrict__ tok,// [M]
                                        int M, int H) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= H) return;
    int b = tok[row];
    atomicAdd(&Eagg[b*H + col], Y[row*H + col]);
}
