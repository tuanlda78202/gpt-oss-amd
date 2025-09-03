#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

__device__ __forceinline__ float clamp01(float x) {
    return fminf(1.0f, fmaxf(0.0f, x));
}

// ====== FAST PARALLEL COS/SIN ======
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

    // Using powf(base, exp) is fine here (d_half is small). If you compile with -ffast-math, it's faster.
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

// ====== APPLY ROTARY ======
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

// ---------------- Host wrappers (drop-in) ----------------
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
