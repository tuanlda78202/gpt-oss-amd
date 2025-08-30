#include "BLAS.hip"
#include <cmath>
#include <cstdio>

// ====== COMPUTE COS SIN KERNEL ======
__global__ void compute_cos_sin_kernel(int pos, float base, int head_dim, float scaling_factor,
                                       float initial_context_length, float ntk_beta,
                                       float ntk_alpha, float* cos_out, float* sin_out) {
    int d_half = head_dim / 2;
    int idx = threadIdx.x;

    if (idx == 0) {
        // Get concentration + inv_freq
        float concentration;
        float* inv_freq = new float[d_half];

        // Call the concentration kernel (simplified inline version)
        {
            float* freq = new float[d_half];
            for (int i = 0; i < d_half; i++) {
                freq[i] = powf(base, ((float)(2 * i)) / (float)head_dim);
            }

            if (scaling_factor > 1.0f) {
                concentration = 0.1f * logf(scaling_factor) + 1.0f;
                float low =
                    d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
                float high =
                    d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);

                for (int i = 0; i < d_half; i++) {
                    float interpolation = 1.0f / (scaling_factor * freq[i]);
                    float extrapolation = 1.0f / freq[i];
                    float ramp = ((float)i - low) / (high - low);
                    if (ramp < 0)
                        ramp = 0;
                    if (ramp > 1)
                        ramp = 1;
                    float mask = 1.0f - ramp;
                    inv_freq[i] = interpolation * (1.0f - mask) + extrapolation * mask;
                }
            } else {
                concentration = 1.0f;
                for (int i = 0; i < d_half; i++) {
                    inv_freq[i] = 1.0f / freq[i];
                }
            }
            delete[] freq;
        }

        // Compute cos and sin for this position
        for (int j = 0; j < d_half; j++) {
            float val = (float)pos * inv_freq[j];
            cos_out[j] = cosf(val) * concentration;
            sin_out[j] = sinf(val) * concentration;
        }

        delete[] inv_freq;
    }
}

// ====== APPLY ROTARY EMBEDDING KERNEL ======
__global__ void apply_rotary_emb_kernel(float* x, float* cos, float* sin, int n_heads,
                                        int head_dim) {
    int half = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int total_elements = n_heads * half;

    for (int elem = idx; elem < total_elements; elem += total_threads) {
        int h = elem / half; // head index
        int i = elem % half; // dimension index

        // Indexing: head h, dim i
        float x1 = x[h * head_dim + i];        // first half
        float x2 = x[h * head_dim + half + i]; // second half

        float c = cos[i];
        float s = sin[i];

        float o1 = x1 * c - x2 * s;
        float o2 = x2 * c + x1 * s;

        x[h * head_dim + i] = o1;
        x[h * head_dim + half + i] = o2;
    }
}

void compute_cos_sin_hip_device(int pos, float base, int head_dim, float scaling_factor,
                                float initial_context_length, float ntk_beta, float ntk_alpha,
                                float* cos_out, float* sin_out) {
    dim3 blockDim(1); // Single thread for this computation
    dim3 gridDim(1);
    hipLaunchKernelGGL(compute_cos_sin_kernel, gridDim, blockDim, 0, 0, pos, base, head_dim,
                       scaling_factor, initial_context_length, ntk_beta, ntk_alpha, cos_out,
                       sin_out);

    CHECK_HIP(hipGetLastError());
}

void apply_rotary_emb_hip_device(float* x, float* cos, float* sin, int n_heads, int head_dim) {
    dim3 blockDim(256);
    dim3 gridDim((n_heads * (head_dim / 2) + blockDim.x - 1) / blockDim.x);
    hipLaunchKernelGGL(apply_rotary_emb_kernel, gridDim, blockDim, 0, 0, x, cos, sin, n_heads,
                       head_dim);

    CHECK_HIP(hipGetLastError());
}
