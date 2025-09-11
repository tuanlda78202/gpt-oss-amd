#include "BLAS.hip"
#include <cstdio>

template<int VEC>
__global__ void split_qkv_kernel(
    const float* __restrict__ qkv,     // [B, (Hq+2Hkv)*D]
    float* __restrict__ q_out,         // [B, Hq*D]
    float* __restrict__ key_cache,     // global cache
    float* __restrict__ value_cache,   // global cache
    int B, int head_dim, int n_q_heads, int n_kv_heads,
    int layer, const int* __restrict__ pos_per_token, int seq_len,
    const int* __restrict__ batch_indices, long long B_stride,
    int kv_dim)                         // = head_dim * n_kv_heads
{
    const int t = blockIdx.y;
    const int colv = blockIdx.x * blockDim.x + threadIdx.x;

    if (t >= B) return;

    const int q_row    = n_q_heads * head_dim;
    const int qkv_row  = (n_q_heads + 2*n_kv_heads) * head_dim;

    const long long loff = (long long)layer * (long long)B_stride * (long long)seq_len * (long long)kv_dim;
    const int pos  = pos_per_token[t];
    const int slot = batch_indices[t];

    const float* __restrict__ qkv_t = qkv + (long long)t * qkv_row;
    float*       __restrict__ q_t   = q_out + (long long)t * q_row;

    // Vector copy helpers
    constexpr int V = VEC;
    const int q_col = colv * V;
    const int kv_col = colv * V;

    // 1) Copy Q
    if (q_col < q_row) {
        if constexpr (V == 4) {
            reinterpret_cast<float4*>(q_t)[colv] = reinterpret_cast<const float4*>(qkv_t)[colv];
        } else {
            #pragma unroll
            for (int i = 0; i < V; ++i) {
                int c = q_col + i;
                if (c < q_row) q_t[c] = qkv_t[c];
            }
        }
    }

    // 2) Scatter K and V to the cache at (layer, slot, pos)
    float* __restrict__ kv_dst = key_cache   + loff + (long long)slot * (long long)seq_len * kv_dim + (long long)pos * kv_dim;
    float* __restrict__ vv_dst = value_cache + loff + (long long)slot * (long long)seq_len * kv_dim + (long long)pos * kv_dim;

    const float* __restrict__ k_src = qkv_t + q_row;
    const float* __restrict__ v_src = k_src + kv_dim;

    if (kv_col < kv_dim) {
        if constexpr (V == 4) {
            reinterpret_cast<float4*>(kv_dst)[colv] = reinterpret_cast<const float4*>(k_src)[colv];
            reinterpret_cast<float4*>(vv_dst)[colv] = reinterpret_cast<const float4*>(v_src)[colv];
        } else {
            #pragma unroll
            for (int i = 0; i < V; ++i) {
                int c = kv_col + i;
                if (c < kv_dim) {
                    kv_dst[c] = k_src[c];
                    vv_dst[c] = v_src[c];
                }
            }
        }
    }
}

inline void split_qkv(
    float* qkv, float* q_out, float* key_cache, float* value_cache,
    int B, int head_dim, int n_q_heads, int n_kv_heads,
    int layer, const int* d_pos_per_token, int seq_len,
    const int* d_batch_indices, long long B_stride,
    hipStream_t stream, int kv_dim)
{
    const int q_row = n_q_heads * head_dim;
    const int vec = ( (q_row % 4 == 0) && (kv_dim % 4 == 0) ) ? 4 : 1;

    dim3 block(256);
    dim3 grid( ((max(q_row, kv_dim) + (vec-1)) / vec + block.x - 1) / block.x, B );

    if (vec == 4) {
        hipLaunchKernelGGL(split_qkv_kernel<4>, grid, block, 0, stream,
            qkv, q_out, key_cache, value_cache, B, head_dim, n_q_heads, n_kv_heads,
            layer, d_pos_per_token, seq_len, d_batch_indices, B_stride, kv_dim);
    } else {
        hipLaunchKernelGGL(split_qkv_kernel<1>, grid, block, 0, stream,
            qkv, q_out, key_cache, value_cache, B, head_dim, n_q_heads, n_kv_heads,
            layer, d_pos_per_token, seq_len, d_batch_indices, B_stride, kv_dim);
    }
    CHECK_HIP(hipGetLastError());
}
