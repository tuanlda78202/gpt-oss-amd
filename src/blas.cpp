#include <hip/hip_runtime.h>
// #include <hipblas.h>
#include <omp.h>

#include "../include/hip_helper.hpp"

#define WARP_SIZE 64
#define MAX_BLOCK_SIZE 1024
#define GEMM_BLOCK_DIM_X 32
#define GEMM_BLOCK_DIM_Y 4
#define VDS_BLOCK_DIM 32

// #define MATRIX_CORE 1

__device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

__device__ float block_reduce_sum(float val) {
    static __shared__ float shared[MAX_BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warp_reduce_sum(val);

    return val;
}

struct thablasHandle_t {
    int current_gpu_id;
    hipStream_t calc_stream;
    hipStream_t copy_stream;
};

int thablasCreate(thablasHandle_t* handle) {
    int current_gpu_id;
    CHECK_HIP(hipGetDevice(&current_gpu_id));
    handle->current_gpu_id = current_gpu_id;

    CHECK_HIP(hipStreamCreate(&handle->calc_stream));
    CHECK_HIP(hipStreamCreate(&handle->copy_stream));

    return 0;
}

int thablasDestroy(thablasHandle_t handle) {
    //
    return 0;
}
