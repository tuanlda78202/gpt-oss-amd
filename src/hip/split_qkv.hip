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
