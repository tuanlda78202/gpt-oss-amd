#include "BLAS.hip"

// ! Vec + Vec
__global__ void vecaddvec_kernel(float* a, float* b, float weight, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        a[i] += b[i] * weight;
}

void vec_add_vec_gpu(float* a, float* b, float weight, int size) {
    dim3 blockDim(64);
    dim3 gridDim((size + 64 - 1) / 64);
    hipLaunchKernelGGL(vecaddvec_kernel, gridDim, blockDim, 0, 0, a, b, weight, size);

    CHECK_HIP(hipGetLastError());
}

// ! Split Gate Up
__global__ void split_gate_up_kernel(float* mlp1_out, float* gate, float* up,
                                     int intermediate_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intermediate_dim) {
        gate[i] = mlp1_out[2 * i];
        up[i] = mlp1_out[2 * i + 1];
    }
}

void split_gate_up_gpu(float* mlp1_out, float* gate, float* up, int intermediate_dim) {
    dim3 blockDim(256);
    dim3 gridDim((intermediate_dim + 255) / 256);
    hipLaunchKernelGGL(split_gate_up_kernel, gridDim, blockDim, 0, 0, mlp1_out, gate, up,
                       intermediate_dim);

    CHECK_HIP(hipGetLastError());
}
