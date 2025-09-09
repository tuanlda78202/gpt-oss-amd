#include "BLAS.hip"

__global__ void rmsnorm_hybrid_kernel(float* o, float* x, __half* weight_half, int size) {
    int lx = threadIdx.x;
    float tmp;
    float ss = 0.0f;
    __shared__ float total_sum;

    for (int i = lx; i < size; i += blockDim.x) {
        tmp = x[i];
        ss += (float)tmp * (float)tmp;
    }

    // Reduce with high precision
    float ss_float = (float)ss;
    ss_float = block_reduce_sum(ss_float);

    if (lx == 0) {
        ss_float /= size;
        ss_float += 1e-5f;
        ss_float = 1.0f / sqrtf(ss_float);
        total_sum = ss_float;
    }
    __syncthreads();

    ss_float = total_sum;
    for (int i = lx; i < size; i += blockDim.x) {
        o[i] = __half2float(weight_half[i]) * ss_float * x[i];
    }
}

__global__ void rmsnorm_batch_kernel(float* o_batch, float* x_batch, __half* weight_half,
                                     int batch_size, int size) {
    int b = blockIdx.x;  // batch index
    int lx = threadIdx.x;

    if (b >= batch_size) return;

    int offset = b * size;
    float* x = x_batch + offset;
    float* o = o_batch + offset;

    float tmp;
    float ss = 0.0f;
    __shared__ float total_sum;

    for (int i = lx; i < size; i += blockDim.x) {
        tmp = x[i];
        ss += tmp * tmp;
    }

    // Reduce with high precision
    float ss_float = block_reduce_sum(ss);

    if (lx == 0) {
        ss_float /= size;
        ss_float += 1e-5f;
        ss_float = 1.0f / sqrtf(ss_float);
        total_sum = ss_float;
    }
    __syncthreads();

    ss_float = total_sum;
    for (int i = lx; i < size; i += blockDim.x) {
        o[i] = __half2float(weight_half[i]) * ss_float * x[i];
    }
}

void rmsnorm_gpu(float* o, float* x, __half* weight, int size) {
    dim3 blockDim(1024);
    dim3 gridDim(1);
    hipLaunchKernelGGL(rmsnorm_hybrid_kernel, gridDim, blockDim, 0, 0, o, x, weight, size);

    CHECK_HIP(hipGetLastError());
}

void rmsnorm_batch_gpu(float* o_batch, float* x_batch, __half* weight, int batch_size, int size) {
    dim3 blockDim(1024);
    dim3 gridDim(batch_size);
    hipLaunchKernelGGL(rmsnorm_batch_kernel, gridDim, blockDim, 0, 0, o_batch, x_batch, weight, batch_size, size);

    CHECK_HIP(hipGetLastError());
}
