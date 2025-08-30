#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <omp.h>

#define CHECK_HIP(cmd)                                                                             \
    do {                                                                                           \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "HIP Error: %s (%d): %s:%d\n", hipGetErrorString(error), error,        \
                    __FILE__, __LINE__);                                                           \
            fflush(stdout);                                                                        \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define WARP_SIZE 64
#define MAX_BLOCK_SIZE 1024
#define MAX_NUM_SUPPORTED_GPUS 32
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

__device__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = std::max(val, __shfl_xor(val, offset));

    return val;
}

__device__ float block_reduce_max(float val) {
    static __shared__ float shared[MAX_BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -3.402e+38;

    if (wid == 0)
        val = warp_reduce_max(val);

    return val;
}
