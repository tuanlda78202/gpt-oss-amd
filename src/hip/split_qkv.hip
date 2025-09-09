#include "BLAS.hip"
#include <cstdio>

__global__ void split_qkv_kernel(
    const float* qkv,
    float* q,
    float* k,
    float* v,
    int head_dim,
    int n_attn_heads,
    int n_kv_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int q_size = head_dim * n_attn_heads;
    int k_size = head_dim * n_kv_heads;
    int v_size = head_dim * n_kv_heads;
    int total_size = q_size + k_size + v_size;

    if (idx < total_size) {
        if (idx < q_size) {
            // Copy Q part
            q[idx] = qkv[idx];
        } else if (idx < q_size + k_size) {
            // Copy K part
            int k_idx = idx - q_size;
            k[k_idx] = qkv[idx];
        } else {
            // Copy V part
            int v_idx = idx - q_size - k_size;
            v[v_idx] = qkv[idx];
        }
    }
}

__global__ void split_qkv_batch_kernel(
    const float* qkv_batch,         // (B, (q_heads + 2*kv_heads) * head_dim)
    float* q_batch,                 // (B, q_heads * head_dim)
    float* key_cache,               // (L, B, T, kv_dim) - write current K to cache
    float* value_cache,             // (L, B, T, kv_dim) - write current V to cache
    int batch_size,
    int head_dim,
    int n_attn_heads,
    int n_kv_heads,
    int layer,
    int pos,
    int seq_len,
    const int* __restrict__ batch_indices,
    int B_stride
) {
    int b = blockIdx.y;                            // batch index
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // feature index within QKV

    if (b >= batch_size) return;

    int q_size    = head_dim * n_attn_heads;
    int k_size    = head_dim * n_kv_heads;
    int v_size    = head_dim * n_kv_heads;
    int total_size= q_size + k_size + v_size;
    int kv_dim    = head_dim * n_kv_heads;

    const int gb = batch_indices[b];
    const long long layer_stride = 1ll * B_stride * seq_len * kv_dim;
    const long long batch_stride = 1ll * seq_len  * kv_dim;
    const long long base_t       = 1ll * layer * layer_stride
                                 + 1ll * gb    * batch_stride
                                 + 1ll * pos   * kv_dim;

    if (idx < total_size) {
        int qkv_offset = b * total_size;

        if (idx < q_size) {
            // Copy Q part to q_batch
            int q_offset = b * q_size;
            q_batch[q_offset + idx] = qkv_batch[qkv_offset + idx];
        } else if (idx < q_size + k_size) {
            // Copy K part to key_cache
            int k_idx = idx - q_size;
            long long cache_offset = base_t + k_idx;
            key_cache[cache_offset] = qkv_batch[qkv_offset + idx];
        } else {
            // Copy V part to value_cache
            int v_idx = idx - q_size - k_size;
            long long cache_offset = base_t + v_idx;
            value_cache[cache_offset] = qkv_batch[qkv_offset + idx];
        }
    }
}

void split_qkv_gpu(
    const float* qkv,
    float* q,
    float* k,
    float* v,
    int head_dim,
    int n_attn_heads,
    int n_kv_heads
) {
    int q_size = head_dim * n_attn_heads;
    int k_size = head_dim * n_kv_heads;
    int v_size = head_dim * n_kv_heads;
    int total_size = q_size + k_size + v_size;

    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    hipLaunchKernelGGL(split_qkv_kernel,
                       dim3(grid_size),
                       dim3(block_size),
                       0, 0,
                       qkv, q, k, v, head_dim, n_attn_heads, n_kv_heads);

    CHECK_HIP(hipGetLastError());
}

void split_qkv_batch_gpu(
    const float* qkv_batch,
    float* q_batch,
    float* key_cache,
    float* value_cache,
    int batch_size,
    int head_dim,
    int n_attn_heads,
    int n_kv_heads,
    int layer,
    int pos,
    int seq_len,
    const int* __restrict__ d_batch_indices,
    int B_stride
) {
    int q_size = head_dim * n_attn_heads;
    int k_size = head_dim * n_kv_heads;
    int v_size = head_dim * n_kv_heads;
    int total_size = q_size + k_size + v_size;

    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    hipLaunchKernelGGL(split_qkv_batch_kernel,
                       dim3(grid_size, batch_size),
                       dim3(block_size),
                       0, 0,
                       qkv_batch, q_batch, key_cache, value_cache,
                       batch_size, head_dim, n_attn_heads, n_kv_heads,
                       layer, pos, seq_len,
                       d_batch_indices, B_stride);

    CHECK_HIP(hipGetLastError());
}
