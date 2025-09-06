#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

__device__ __forceinline__ float clamp01(float x) {
    return fminf(1.0f, fmaxf(0.0f, x));
}

__global__ void compute_cos_sin_kernel_fast(
    int pos, float base, int head_dim, float scaling_factor,
    float initial_context_length, float ntk_beta, float ntk_alpha,
    float* __restrict__ cos_out, float* __restrict__ sin_out)
{
    const int d_half = head_dim >> 1;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_half) return;

    // freq[i] = base^(2i/head_dim); inv = 1/freq = base^(-2i/head_dim)
    const float two_over_hd = 2.0f / (float)head_dim;
    const float inv_freq_base = -two_over_hd * (float)i; // exponent for base^(...)

    float inv_freq = powf(base, inv_freq_base);

    float concentration = 1.0f;

    if (scaling_factor > 1.0f) {
        concentration = 0.1f * logf(scaling_factor) + 1.0f;

        const float logb = logf(base);
        const float two_pi = 6.28318530717958647692f;
        const float low  = (float)d_half * logf(initial_context_length / (ntk_beta  * two_pi)) / logb;
        const float high = (float)d_half * logf(initial_context_length / (ntk_alpha * two_pi)) / logb;

        // ramp in [0,1] across i
        float ramp = clamp01(((float)i - low) / (high - low));

        // interpolation vs extrapolation mix
        float interpolation = 1.0f / (scaling_factor * powf(base, (2.0f * (float)i) / (float)head_dim));
        float extrapolation = 1.0f / powf(base, (2.0f * (float)i) / (float)head_dim);
        inv_freq = ramp * interpolation + (1.0f - ramp) * extrapolation;
    }

    // theta = pos * inv_freq
    float theta = (float)pos * inv_freq;

    float s, c;
    // hip has sincosf on device
    sincosf(theta, &s, &c);

    cos_out[i] = c * concentration;
    sin_out[i] = s * concentration;
}

__global__ void apply_rotary_emb_kernel_fast(
    float* __restrict__ x,
    const float* __restrict__ cosv,
    const float* __restrict__ sinv,
    int n_heads, int head_dim)
{
    const int half = head_dim >> 1;
    const int total = n_heads * half;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int h = idx / half;
        int i = idx % half;

        float x1 = x[h * head_dim + i];
        float x2 = x[h * head_dim + half + i];

        float c = cosv[i];
        float s = sinv[i];

        // (x1, x2) <- rot((x1, x2), theta)
        float o1 = fmaf(-x2, s, x1 * c);  // x1*c - x2*s
        float o2 = fmaf( x1, s, x2 * c);  // x2*c + x1*s

        x[h * head_dim + i]         = o1;
        x[h * head_dim + half + i]  = o2;
    }
}

void compute_cosin_gpu(int pos, float base, int head_dim, float scaling_factor,
                       float initial_context_length, float ntk_beta, float ntk_alpha,
                       float* cos_out, float* sin_out)
{
    const int d_half = head_dim >> 1;
    const int block = 256;
    const int grid  = (d_half + block - 1) / block;

    hipLaunchKernelGGL(compute_cos_sin_kernel_fast, dim3(grid), dim3(block), 0, 0,
                       pos, base, head_dim, scaling_factor,
                       initial_context_length, ntk_beta, ntk_alpha,
                       cos_out, sin_out);
    CHECK_HIP(hipGetLastError());
}

__global__ void apply_rotary_emb_batch_kernel(
    float* __restrict__ x_batch,  // (B, n_heads, head_dim)
    const float* __restrict__ cosv,
    const float* __restrict__ sinv,
    int batch_size, int n_heads, int head_dim)
{
    const int half = head_dim >> 1;
    const int total_per_batch = n_heads * half;

    int b = blockIdx.y;  // batch index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // index within batch

    if (b >= batch_size || idx >= total_per_batch) return;

    int h = idx / half;  // head index
    int i = idx % half;  // position within head

    int x_offset = b * (n_heads * head_dim);  // offset for this batch

    float x1 = x_batch[x_offset + h * head_dim + i];
    float x2 = x_batch[x_offset + h * head_dim + half + i];

    float c = cosv[i];
    float s = sinv[i];

    // (x1, x2) <- rot((x1, x2), theta)
    float o1 = fmaf(-x2, s, x1 * c);  // x1*c - x2*s
    float o2 = fmaf( x1, s, x2 * c);  // x2*c + x1*s

    x_batch[x_offset + h * head_dim + i]         = o1;
    x_batch[x_offset + h * head_dim + half + i]  = o2;
}

void rope_gpu(float* x, float* cosv, float* sinv, int n_heads, int head_dim)
{
    const int half = head_dim >> 1;
    const int total = n_heads * half;
    const int block = 256;
    const int grid  = (total + block - 1) / block;

    hipLaunchKernelGGL(apply_rotary_emb_kernel_fast, dim3(grid), dim3(block), 0, 0,
                       x, cosv, sinv, n_heads, head_dim);
    CHECK_HIP(hipGetLastError());
}

void rope_q_batch_gpu(float* x_batch, float* cosv, float* sinv, int batch_size, int n_heads, int head_dim)
{
    const int half = head_dim >> 1;
    const int total_per_batch = n_heads * half;
    const int block = 256;
    const int grid_x = (total_per_batch + block - 1) / block;

    hipLaunchKernelGGL(apply_rotary_emb_batch_kernel, dim3(grid_x, batch_size), dim3(block), 0, 0,
                       x_batch, cosv, sinv, batch_size, n_heads, head_dim);
    CHECK_HIP(hipGetLastError());
}

__global__ void apply_rotary_to_kcache_at_pos_batch_kernel(float* k_cache,
                                                           const float* cosv,
                                                           const float* sinv,
                                                           int batch_size,
                                                           int n_kv_heads,
                                                           int head_dim,
                                                           int seq_len,
                                                           int kv_dim,
                                                           int layer_idx,
                                                           int pos,
                                                           const int* __restrict__ batch_indices,
                                                           int B_stride) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int half = head_dim >> 1;

    if (h >= n_kv_heads || b >= batch_size)
        return;

    const int gb = batch_indices[b];
    const long long layer_stride = 1ll * B_stride * seq_len * kv_dim;
    const long long batch_stride = 1ll * seq_len  * kv_dim;

    long long base = 1ll * layer_idx * layer_stride
                   + 1ll * gb        * batch_stride
                   + 1ll * pos       * kv_dim
                   + 1ll * h         * head_dim;

    float* x = k_cache + base;

    for (int i = tid; i < half; i += blockDim.x) {
        float x1 = x[i];
        float x2 = x[half + i];
        float c = cosv[i];
        float s = sinv[i];
        x[i] = fmaf(-x2, s, x1 * c);
        x[half + i] = fmaf(x1, s, x2 * c);
    }
}

inline void rope_k_batch_gpu(float* k_cache, float* cosv, float* sinv, int batch_size,
                                         int n_kv_heads, int head_dim, int seq_len, int kv_dim,
                                         int layer_idx, int pos,
                                         const int* __restrict__ batch_indices, int B_stride) {
    dim3 grid(n_kv_heads, batch_size);
    dim3 block(256);
    hipLaunchKernelGGL(apply_rotary_to_kcache_at_pos_batch_kernel, grid, block, 0, 0, k_cache, cosv,
                       sinv, batch_size, n_kv_heads, head_dim, seq_len, kv_dim, layer_idx, pos,
                       batch_indices, B_stride);
    CHECK_HIP(hipGetLastError());
}
