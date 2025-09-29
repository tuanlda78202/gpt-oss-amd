#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
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

using i16x4 = __attribute__((ext_vector_type(4))) short;
using f4 = __attribute__((ext_vector_type(4))) float;
using f2 = __attribute__((ext_vector_type(2))) float;

// Helper to get the raw 16-bit payload from a __hip_bfloat16
static inline __device__ short bf16_bits(__hip_bfloat16 v) {
    return *reinterpret_cast<const short*>(&v);
}

// A vectorized load function for B. It loads 4 floats from global memory, converts
// each to bfloat16, and returns their raw bits packed in an i16x4.
__device__ inline i16x4 load_and_convert_f32_to_bf16x4(const float* gmem_ptr) {
    float4 f_vals = *reinterpret_cast<const float4*>(gmem_ptr);
    i16x4 res;
    res[0] = bf16_bits(__float2bfloat16(f_vals.x));
    res[1] = bf16_bits(__float2bfloat16(f_vals.y));
    res[2] = bf16_bits(__float2bfloat16(f_vals.z));
    res[3] = bf16_bits(__float2bfloat16(f_vals.w));
    return res;
}

template <typename T>
__device__ __forceinline__ void nt_store(T* p, T v) {
    if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value ||
                  std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
                  std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value) {
        __builtin_nontemporal_store(v, p);
    } else {
        *p = v;
    }
}

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define IS_ALIGNED(ptr, bytes) ((((uintptr_t)(ptr)) & ((bytes)-1)) == 0)

// #define MATRIX_CORE 1

struct Int2 { int x; int y; };

__device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_64(float v) {
#pragma unroll
    for (int ofs = 32; ofs > 0; ofs >>= 1) v += __shfl_down(v, ofs, 64);
    return v;
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
        val = MAX(val, __shfl_xor(val, offset));

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
