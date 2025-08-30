#include "BLAS.hip"
#include <hip/hip_runtime.h>

// GPU Argmax kernel
__global__ void argmax_kernel(float* logits, int vocab_size, int* result) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Shared memory for reduction
    __shared__ float shared_vals[1024];
    __shared__ int shared_indices[1024];

    float max_val = -INFINITY;
    int max_idx = 0;

    // Each thread finds max in its section
    for (int i = tid; i < vocab_size; i += blockSize) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    shared_vals[tid] = max_val;
    shared_indices[tid] = max_idx;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid] < shared_vals[tid + stride]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = shared_indices[0];
    }
}

// GPU Random number generation
__device__ unsigned int gpu_xorshift(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

__device__ float gpu_random_f32(unsigned long long* state) {
    return (gpu_xorshift(state) >> 8) / 16777216.0f;
}

// GPU Multinomial sampling kernel
__global__ void multinomial_sample_kernel(float* probabilities, int vocab_size,
                                        unsigned long long seed, int* result) {
    int tid = threadIdx.x;
    if (tid != 0) return;  // Only thread 0 does sampling

    unsigned long long rng_state = seed + blockIdx.x;
    float coin = gpu_random_f32(&rng_state);

    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            *result = i;
            return;
        }
    }
    *result = vocab_size - 1;
}

// Temperature scaling kernel
__global__ void temperature_scale_kernel(float* logits, int vocab_size, float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        logits[idx] /= temperature;
    }
}

// Device functions
void sample_argmax_hip_device(float* logits, int vocab_size, int* result_d) {
    dim3 block(1024);
    dim3 grid(1);
    hipLaunchKernelGGL(argmax_kernel, grid, block, 0, 0, logits, vocab_size, result_d);
    CHECK_HIP(hipGetLastError());
}

void sample_multinomial_hip_device(float* logits, int vocab_size, float temperature,
                                   unsigned long long seed, int* result_d) {
    // Apply temperature scaling
    if (temperature != 1.0f) {
        dim3 temp_block(256);
        dim3 temp_grid((vocab_size + 255) / 256);
        hipLaunchKernelGGL(temperature_scale_kernel, temp_grid, temp_block, 0, 0,
                          logits, vocab_size, temperature);
    }

    // Apply softmax
    softmax_hip_device(logits, vocab_size);

    // Sample
    dim3 block(256);
    dim3 grid(1);
    hipLaunchKernelGGL(multinomial_sample_kernel, grid, block, 0, 0,
                      logits, vocab_size, seed, result_d);
    CHECK_HIP(hipGetLastError());
}
