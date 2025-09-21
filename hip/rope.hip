#include "BLAS.hip"
#include <hip/hip_runtime.h>

__device__ __forceinline__
void cosin_kernel(int i, int head_dim,
                     float base, float scaling_factor,
                     float initial_ctx_len, float ntk_beta, float ntk_alpha,
                     float pos, float& c, float& s)
{
    const float d_half = (float)(head_dim >> 1);
    const float two_over_hd = 2.0f / (float)head_dim;
    const float inv_freq_base = -two_over_hd * (float)i;

    float inv_freq = powf(base, inv_freq_base);
    float concentration = 1.0f;

    if (scaling_factor > 1.0f) {
        concentration = 0.1f * logf(scaling_factor) + 1.0f;

        const float logb = logf(base);
        const float two_pi = 6.28318530717958647692f;
        const float low  = d_half * logf(initial_ctx_len / (ntk_beta  * two_pi)) / logb;
        const float high = d_half * logf(initial_ctx_len / (ntk_alpha * two_pi)) / logb;

        float ramp = fminf(1.0f, fmaxf(0.0f, ((float)i - low) / (high - low)));

        float interpolation = 1.0f / (scaling_factor * powf(base, (2.0f * (float)i) / (float)head_dim));
        float extrapolation = 1.0f / powf(base, (2.0f * (float)i) / (float)head_dim);
        inv_freq = ramp * interpolation + (1.0f - ramp) * extrapolation;
    }

    float theta = pos * inv_freq;
    float ss, cc;
    sincosf(theta, &ss, &cc);
    c = cc * concentration;
    s = ss * concentration;
}

__global__ void rope_qk_kernel(
    float* __restrict__ q_batch,
    void* __restrict__ key_cache,
    int B, int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, int kv_dim, int layer,
    const int* __restrict__ pos_per_token,
    int kv_cache_is_fp16,
    const int* __restrict__ batch_indices,
    long long B_stride,
    float base, float scaling_factor,
    float initial_ctx_len, float ntk_beta, float ntk_alpha,
    const long long* __restrict__ d_layer_kv_off,
    const int* __restrict__ d_layer_kv_cap,
    const int* __restrict__ d_layer_is_local)
{
    const int half = head_dim >> 1;
    const int t    = blockIdx.y;
    const int i0   = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B || i0 >= half) return;

    const int pos  = pos_per_token[t];
    const int slot = batch_indices[t];

    float c, s;
    cosin_kernel(i0, head_dim, base, scaling_factor, initial_ctx_len, ntk_beta, ntk_alpha, (float)pos, c, s);

    float* qrow = q_batch + (long long)t * (long long)(n_q_heads*head_dim);
    #pragma unroll 1
    for (int h = 0; h < n_q_heads; ++h) {
        float* qh = qrow + (long long)h*head_dim;
        float x1 = qh[i0];
        float x2 = qh[half + i0];
        qh[i0]        = fmaf(-x2, s, x1 * c);
        qh[half + i0] = fmaf( x1, s, x2 * c);
    }

    const long long kv_base = d_layer_kv_off[layer];
    const int       cap     = d_layer_kv_cap[layer];
    const int       local   = d_layer_is_local[layer];
    const int       pos_wrapped = local ? (pos % cap) : pos;

    const size_t element_size = kv_cache_is_fp16 ? sizeof(__hip_bfloat16) : sizeof(float);
    void* k_base = (void*)((char*)key_cache +
       (kv_base + (long long)slot * (long long)cap * kv_dim + (long long)pos_wrapped * kv_dim) * element_size);

    #pragma unroll 1
    for (int h = 0; h < n_kv_heads; ++h) {
        if (kv_cache_is_fp16) {
            __hip_bfloat16* kh = (__hip_bfloat16*)k_base + (long long)h*head_dim;
            float x1 = __bfloat162float(kh[i0]);
            float x2 = __bfloat162float(kh[half + i0]);
            float y1 = fmaf(-x2, s, x1 * c);
            float y2 = fmaf( x1, s, x2 * c);
            kh[i0]        = __float2bfloat16(y1);
            kh[half + i0] = __float2bfloat16(y2);
        } else {
            float* kh = (float*)k_base + (long long)h*head_dim;
            float x1 = kh[i0];
            float x2 = kh[half + i0];
            kh[i0]        = fmaf(-x2, s, x1 * c);
            kh[half + i0] = fmaf( x1, s, x2 * c);
        }
    }
}

inline void rope_qk(
    float* q_batch, void* key_cache,
    int B, int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, int kv_dim, int layer,
    const int* d_pos_per_token, const int* d_batch_indices,
    int kv_cache_is_fp16,
    long long B_stride,
    float base, float scaling_factor, float initial_ctx_len, float ntk_beta, float ntk_alpha,
    hipStream_t stream,
    const long long* d_layer_kv_off, const int* d_layer_kv_cap, const int* d_layer_is_local)
{
    const int half = head_dim >> 1;
    const dim3 block(256);
    const dim3 grid((half + block.x - 1) / block.x, B);
    hipLaunchKernelGGL(rope_qk_kernel, grid, block, 0, stream,
        q_batch, key_cache,
        B, n_q_heads, n_kv_heads, head_dim,
        seq_len, kv_dim, layer,
        d_pos_per_token, kv_cache_is_fp16, d_batch_indices, B_stride,
        base, scaling_factor, initial_ctx_len, ntk_beta, ntk_alpha,
        d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
    CHECK_HIP(hipGetLastError());
}
