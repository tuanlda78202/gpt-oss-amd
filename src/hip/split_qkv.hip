#include "BLAS.hip"
#include <cstdio>

template<int VEC>
__global__ void split_qkv_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ q_out,
    void* __restrict__ key_cache,
    void* __restrict__ value_cache,
    int B, int head_dim, int n_q_heads, int n_kv_heads,
    int layer, const int* __restrict__ pos_per_token, int seq_len,
    const int* __restrict__ batch_indices, long long B_stride,
    int kv_dim,
    int kv_cache_is_fp16,
    const long long* __restrict__ d_layer_kv_off,
    const int* __restrict__ d_layer_kv_cap,
    const int* __restrict__ d_layer_is_local)
{
    const int t = blockIdx.y;
    const int colv = blockIdx.x * blockDim.x + threadIdx.x;

    if (t >= B) return;

    const int q_row    = n_q_heads * head_dim;
    const int qkv_row  = (n_q_heads + 2*n_kv_heads) * head_dim;

    const long long kv_base = d_layer_kv_off[layer];
    const int       cap     = d_layer_kv_cap[layer];
    const int       local   = d_layer_is_local[layer];

    const int pos  = pos_per_token[t];
    const int slot = batch_indices[t];
    const int pos_wrapped = local ? (pos % cap) : pos;

    const float* __restrict__ qkv_t = qkv + (long long)t * qkv_row;
    float*       __restrict__ q_t   = q_out + (long long)t * q_row;

    constexpr int V = VEC;
    const int q_col = colv * V;
    const int kv_col = colv * V;

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

    const size_t element_size = kv_cache_is_fp16 ? sizeof(__hip_bfloat16) : sizeof(float);
    const long long elem_index = kv_base + (long long)slot * (long long)cap * kv_dim + (long long)pos_wrapped * kv_dim;

    void* kv_dst_base = (void*)((char*)key_cache   + elem_index * element_size);
    void* vv_dst_base = (void*)((char*)value_cache + elem_index * element_size);

    const float* __restrict__ k_src = qkv_t + q_row;
    const float* __restrict__ v_src = k_src + kv_dim;

    if (kv_col < kv_dim) {
        #pragma unroll
        for (int i = 0; i < V; ++i) {
            int c = kv_col + i;
            if (c < kv_dim) {
                if (kv_cache_is_fp16) {
                    __hip_bfloat16* kv_dst = (__hip_bfloat16*)kv_dst_base;
                    __hip_bfloat16* vv_dst = (__hip_bfloat16*)vv_dst_base;
                    kv_dst[c] = __float2bfloat16(k_src[c]);
                    vv_dst[c] = __float2bfloat16(v_src[c]);
                } else {
                    float* kv_dst = (float*)kv_dst_base;
                    float* vv_dst = (float*)vv_dst_base;
                    kv_dst[c] = k_src[c];
                    vv_dst[c] = v_src[c];
                }
            }
        }
    }
}

inline void split_qkv(
    float* qkv, float* q_out, void* key_cache, void* value_cache,
    int B, int head_dim, int n_q_heads, int n_kv_heads,
    int layer, const int* d_pos_per_token, int seq_len,
    const int* d_batch_indices, long long B_stride,
    hipStream_t stream, int kv_dim, int kv_cache_is_fp16,
    const long long* d_layer_kv_off, const int* d_layer_kv_cap, const int* d_layer_is_local)
{
    const int q_row = n_q_heads * head_dim;
    const int vec = (q_row % 4 == 0) ? 4 : 1;

    dim3 block(256);
    dim3 grid( ((max(q_row, kv_dim) + (vec-1)) / vec + block.x - 1) / block.x, B );

    if (vec == 4) {
        hipLaunchKernelGGL(split_qkv_kernel<4>, grid, block, 0, stream,
            qkv, q_out, key_cache, value_cache, B, head_dim, n_q_heads, n_kv_heads,
            layer, d_pos_per_token, seq_len, d_batch_indices, B_stride, kv_dim, kv_cache_is_fp16,
            d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
    } else {
        hipLaunchKernelGGL(split_qkv_kernel<1>, grid, block, 0, stream,
            qkv, q_out, key_cache, value_cache, B, head_dim, n_q_heads, n_kv_heads,
            layer, d_pos_per_token, seq_len, d_batch_indices, B_stride, kv_dim, kv_cache_is_fp16,
            d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
    }
    CHECK_HIP(hipGetLastError());
}
