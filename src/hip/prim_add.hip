#include "../../include/hip_helper.hpp"
#include "../BLAS.hip"

__global__ void vecaddvec_kernel(float* a, float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        a[i] += b[i];
}

void vecaddvec_hip(float* a, float* b, int size) {
    // if (a == nullptr || b == nullptr || size == 0)
    // {
    //     printf("THABLAS VEC ADD VEC ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(64);
    dim3 gridDim((size + 64 - 1) / 64);
    hipLaunchKernelGGL(vecaddvec_kernel, gridDim, blockDim, 0, 0, a, b, size);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}
