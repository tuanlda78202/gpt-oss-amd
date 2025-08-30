#include "BLAS.hip"
#include <cstdio>
#include <omp.h>

__global__ void s_softmax_v2_kernel(float* x, int size) {
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

// _s_ = single persion (float)
// input: output, x allocated on device
// input: size = 32000
void s_softmax_v2(float* x, int size) {
    if (size == 0 || x == nullptr) {
        fprintf(stderr, "SOFTMAX ERROR: INVALID ARGUMENT\n");
        return;
    }

    dim3 blockDim(256);
    dim3 gridDim(1);

    hipLaunchKernelGGL(s_softmax_v2_kernel, gridDim, blockDim, 0, 0, x, size);

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

void softmax_hip(float* x, int size) {
    float* x_d;
    CHECK_HIP(hipMalloc(&x_d, size * sizeof(float)));
    CHECK_HIP(hipMemcpy(x_d, x, size * sizeof(float), hipMemcpyHostToDevice));

    s_softmax_v2(x_d, size);

    CHECK_HIP(hipMemcpy(x, x_d, size * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(x_d));
}

void softmax_gpu(float* x, int size) {
    dim3 blockDim(256);
    dim3 gridDim(1);
    hipLaunchKernelGGL(s_softmax_v2_kernel, gridDim, blockDim, 0, 0, x, size);

    CHECK_HIP(hipGetLastError());
}
