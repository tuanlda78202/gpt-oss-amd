#include "BLAS.hip"

__global__ void vecaddvec_kernel(float* a, float* b, float weight, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        a[i] += b[i] * weight;
}

void vecaddvec(float* a, float* b, float weight, int size) {
    dim3 blockDim(64);
    dim3 gridDim((size + 64 - 1) / 64);
    hipLaunchKernelGGL(vecaddvec_kernel, gridDim, blockDim, 0, 0, a, b, weight, size);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

void vecaddvec_hip(float* a, float* b, float weight, int size) {
    float *a_d, *b_d;
    CHECK_HIP(hipMalloc(&a_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&b_d, size * sizeof(float)));

    CHECK_HIP(hipMemcpy(a_d, a, size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(b_d, b, size * sizeof(float), hipMemcpyHostToDevice));

    vecaddvec(a_d, b_d, weight, size);

    CHECK_HIP(hipMemcpy(a, a_d, size * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipFree(a_d));
    CHECK_HIP(hipFree(b_d));
}

// ! High-precision hybrid vector addition kernel for FP16 bias + FP32 activations
__global__ void vecaddvec_hybrid_kernel(float* a, __half* b_half, float weight, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        __half bias_val = b_half[i];
        float b_fp32 = __half2float(bias_val);
        float weighted_bias = b_fp32 * weight;
        a[i] += weighted_bias;
    }
}

void vecaddvec_hip_hybrid_device(float* a, __half* b, float weight, int size) {
    dim3 blockDim(64);
    dim3 gridDim((size + 64 - 1) / 64);
    hipLaunchKernelGGL(vecaddvec_hybrid_kernel, gridDim, blockDim, 0, 0, a, b, weight, size);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

void vecaddvec_hip_device(float* a, float* b, float weight, int size) {
    dim3 blockDim(64);
    dim3 gridDim((size + 64 - 1) / 64);
    hipLaunchKernelGGL(vecaddvec_kernel, gridDim, blockDim, 0, 0, a, b, weight, size);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

__global__ void split_gate_up_kernel(float* mlp1_out, float* gate, float* up,
                                     int intermediate_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intermediate_dim) {
        gate[i] = mlp1_out[2 * i];
        up[i] = mlp1_out[2 * i + 1];
    }
}

void split_gate_up_hip_device(float* mlp1_out, float* gate, float* up, int intermediate_dim) {
    dim3 blockDim(256);
    dim3 gridDim((intermediate_dim + 255) / 256);
    hipLaunchKernelGGL(split_gate_up_kernel, gridDim, blockDim, 0, 0, mlp1_out, gate, up,
                       intermediate_dim);
    CHECK_HIP(hipGetLastError());
}
