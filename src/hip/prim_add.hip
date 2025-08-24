#include "../../include/hip_helper.hpp"
#include "../BLAS.hip"

__global__ void vecaddvec_kernel(float* a, float* b, float weight, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        a[i] += b[i] * weight;
}

void vecaddvec(float* a, float* b, float weight, int size) {
    // if (a == nullptr || b == nullptr || size == 0)
    // {
    //     printf("THABLAS VEC ADD VEC ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
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
