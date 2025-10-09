#include "BLAS.hip"
#include <hip/hip_runtime.h>

__global__ void argmax_kernel(float* logits, int vocab_size, int* result) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    __shared__ float shared_vals[1024];
    __shared__ int shared_indices[1024];

    float max_val = -INFINITY;
    int max_idx = 0;

    for (int i = tid; i < vocab_size; i += blockSize) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    shared_vals[tid] = max_val;
    shared_indices[tid] = max_idx;
    __syncthreads();

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

__global__ void argmax_batched_kernel(const float* __restrict__ logits,
                                      int vocab_size, int stride,
                                      const int* __restrict__ rows,
                                      int* __restrict__ out) {
    int seq = blockIdx.x;
    int tid = threadIdx.x;

    int row = rows ? rows[seq] : seq;
    int base = row * stride;

    __shared__ float svals[1024];
    __shared__ int sidx[1024];

    float max_val = -INFINITY;
    int max_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = logits[base + i];
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    svals[tid] = max_val;
    sidx[tid] = max_idx;
    __syncthreads();

    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            if (svals[tid] < svals[tid + step]) {
                svals[tid] = svals[tid + step];
                sidx[tid] = sidx[tid + step];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[seq] = sidx[0];
}

__device__ unsigned int gpu_xorshift(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

__device__ float gpu_random_f32(unsigned long long* state) {
    return (gpu_xorshift(state) >> 8) / 16777216.0f;
}

__global__ void multinomial_sample_kernel(float* probabilities, int vocab_size,
                                          unsigned long long seed, int* result) {
    int tid = threadIdx.x;
    if (tid != 0)
        return;

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

__global__ void temperature_scale_kernel(float* logits, int vocab_size, float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        logits[idx] /= temperature;
    }
}

void sample_argmax(float* logits, int vocab_size, int* result_d, hipStream_t stream = nullptr) {
    dim3 block(1024);
    dim3 grid(1);
    hipLaunchKernelGGL(argmax_kernel, grid, block, 0, stream, logits, vocab_size, result_d);
    CHECK_HIP(hipGetLastError());
}

void sample_argmax_batched(const float* logits, int vocab_size,
                           const int* rows_d, int batch,
                           int* results_d, hipStream_t stream) {
    dim3 block(512);
    dim3 grid(batch);
    hipLaunchKernelGGL(argmax_batched_kernel, grid, block, 0, stream,
                       logits, vocab_size, vocab_size, rows_d, results_d);
    CHECK_HIP(hipGetLastError());
}

void sample_multinomial(float* logits, int vocab_size, float temperature,
                                   unsigned long long seed, int* result_d) {
    if (temperature != 1.0f) {
        dim3 temp_block(256);
        dim3 temp_grid((vocab_size + 255) / 256);
        hipLaunchKernelGGL(temperature_scale_kernel, temp_grid, temp_block, 0, 0, logits,
                           vocab_size, temperature);
    }

    softmax(logits, vocab_size);

    dim3 block(256);
    dim3 grid(1);
    hipLaunchKernelGGL(multinomial_sample_kernel, grid, block, 0, 0, logits, vocab_size, seed,
                       result_d);
    CHECK_HIP(hipGetLastError());
}
