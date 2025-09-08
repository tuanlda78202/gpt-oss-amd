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

// Build (B*K) assignment triples *and* count per-expert, one pass
__global__ void build_assignments_and_count_kernel(
    const int* __restrict__ topk_i,
    const float* __restrict__ topk_v,
    int B, int K,
    int* __restrict__ out_expert,
    int* __restrict__ out_token,
    float* __restrict__ out_w,
    int* __restrict__ counts, // [E], zeroed before launch
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

    // Count per expert (bounds guard, avoids UB on bad data)
    if ((unsigned)e < (unsigned)E) atomicAdd(&counts[e], 1);
}

// offsets has size E+1, where offsets[0]=0, offsets[e+1]=sum_{i<=e} counts[i]
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
// Vectorized gather: X[B,H] -> Y[BK,H] using tokens
__global__ void gather_rows_vec4_kernel(const float* __restrict__ X,
                                        const int*  __restrict__ idx,
                                        float* __restrict__ Y,
                                        int BK, int H4) // H4 = H/4
{
    int row = blockIdx.y;                          // 0..BK-1
    int col4 = blockIdx.x * blockDim.x + threadIdx.x; // 0..H4-1
    if (row >= BK || col4 >= H4) return;

    int b = idx[row];
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(X);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(Y);

    y4[row * H4 + col4] = x4[b * H4 + col4];
}

// Fallback scalar gather (unaligned or H%4!=0)
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

// Fused scale(+w[row]) and scatter-add to Eagg[B,H]
__global__ void scale_scatter_rows_kernel(float* __restrict__ Eagg,    // [B,H]
                                          const float* __restrict__ Y, // [M,H]
                                          const int*   __restrict__ tok,//[M]
                                          const float* __restrict__ w, // [M]
                                          int M, int H)
{
    int row = blockIdx.y; // 0..M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..H-1
    if (row >= M || col >= H) return;
    int b = tok[row];
    float val = Y[row*H + col] * w[row];
    atomicAdd(&Eagg[b*H + col], val);
}
