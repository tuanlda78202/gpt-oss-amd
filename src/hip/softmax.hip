#include "BLAS.hip"
#include <cstdio>
#include <omp.h>

__global__ void softmax_kernel(float* x, int size) {
    int lx = threadIdx.x;
    int bDim = blockDim.x;

    float private_max_val = -3.402e+38;
    __shared__ float max_val;
    for (int i = lx; i < size; i += bDim) {
        private_max_val = std::max(private_max_val, x[i]);
    }

    private_max_val = block_reduce_max(private_max_val);
    if (lx == 0) {
        max_val = private_max_val;
    }
    __syncthreads();
    private_max_val = max_val;

    float private_sum = 0.0f, tmp;
    __shared__ float sum;
    for (int i = lx; i < size; i += bDim) {
        tmp = expf(x[i] - private_max_val);
        x[i] = tmp;
        private_sum += tmp;
    }

    private_sum = block_reduce_sum(private_sum);
    if (lx == 0) {
        sum = private_sum;
    }
    __syncthreads();
    private_sum = sum;

    for (int i = lx; i < size; i += bDim) {
        x[i] /= private_sum;
    }
}

void softmax_gpu(float* x, int size) {
    dim3 blockDim(256);
    dim3 gridDim(1);
    hipLaunchKernelGGL(softmax_kernel, gridDim, blockDim, 0, 0, x, size);

    CHECK_HIP(hipGetLastError());
}

__global__ void softmax_batch_kernel(float* x, int batch_size, int size) {
    int batch_idx = blockIdx.x;  // Each block handles one batch element
    if (batch_idx >= batch_size) return;

    float* batch_x = x + batch_idx * size;  // Offset to this batch's data
    int lx = threadIdx.x;
    int bDim = blockDim.x;

    float private_max_val = -3.402e+38;
    __shared__ float max_val;
    for (int i = lx; i < size; i += bDim) {
        private_max_val = std::max(private_max_val, batch_x[i]);
    }

    private_max_val = block_reduce_max(private_max_val);
    if (lx == 0) {
        max_val = private_max_val;
    }
    __syncthreads();
    private_max_val = max_val;

    float private_sum = 0.0f, tmp;
    __shared__ float sum;
    for (int i = lx; i < size; i += bDim) {
        tmp = expf(batch_x[i] - private_max_val);
        batch_x[i] = tmp;
        private_sum += tmp;
    }

    private_sum = block_reduce_sum(private_sum);
    if (lx == 0) {
        sum = private_sum;
    }
    __syncthreads();
    private_sum = sum;

    for (int i = lx; i < size; i += bDim) {
        batch_x[i] /= private_sum;
    }
}

void softmax_batch_gpu(float* x, int batch_size, int size) {
    dim3 blockDim(256);
    dim3 gridDim(batch_size);
    hipLaunchKernelGGL(softmax_batch_kernel, gridDim, blockDim, 0, 0, x, batch_size, size);
    CHECK_HIP(hipGetLastError());
}
