#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ! Single
template <int RB /* rows per block */>
__global__ void matvec_warp_splitk_vec4(const __half* __restrict__ A_half,
                                        const float* __restrict__ B, float* __restrict__ C,
                                        const __half* __restrict__ bias, int M, int K) {
    const int lane = threadIdx.x; // 0..63
    const int warp = threadIdx.y; // 0..RB-1
    const int row = blockIdx.y * RB + warp;
    if (row >= M)
        return;

    const int a_base = row * K;
    float acc = 0.0f;

    // Process main part in chunks of 4
    const int K4 = K >> 2; // floor(K/4)
    for (int k4 = lane; k4 < K4; k4 += warpSize) {
        const int k = (k4 << 2);

        // Load B[k..k+3] as float4
        float4 bv = *reinterpret_cast<const float4*>(&B[k]);

        // Load A[row,k..k+3] as (half2 + half2)
        half2 a0 = *reinterpret_cast<const half2*>(&A_half[a_base + k + 0]);
        half2 a1 = *reinterpret_cast<const half2*>(&A_half[a_base + k + 2]);

        float2 af0 = __half22float2(a0);
        float2 af1 = __half22float2(a1);

        // FMA
        acc = fmaf(af0.x, bv.x, acc);
        acc = fmaf(af0.y, bv.y, acc);
        acc = fmaf(af1.x, bv.z, acc);
        acc = fmaf(af1.y, bv.w, acc);
    }

    // Tail (K not multiple of 4)
    for (int kt = (K4 << 2) + lane; kt < K; kt += warpSize) {
        acc = fmaf(__half2float(A_half[a_base + kt]), B[kt], acc);
    }

    // Warp reduce
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        float badd = bias ? __half2float(bias[row]) : 0.0f;
        C[row] = acc + badd;
    }
}

void matvec_gpu(float* xout, float* x, __half* w, __half* bias, int n, int d) {
    CHECK_HIP(hipMemset(xout, 0, d * sizeof(float)));

    constexpr int RB = 4; // 256 threads/block; try 8 if regs allow
    dim3 block(64, RB, 1);
    dim3 grid(1, CEIL_DIV(d, RB), 1);
    matvec_warp_splitk_vec4<RB><<<grid, block>>>(w, x, xout, bias, d, n);

    CHECK_HIP(hipGetLastError());
}

// ! Batched
__device__ __forceinline__ float warp_reduce_sum_64(float v) {
    // 64-lane shuffle reduction for AMD wavefront
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 64);
    }
    return v;
}

// Tunables: rows per block (RB) and K tile size (KTILE)
template <int RB, int KTILE>
__global__ void matvec_batch_kernel_tiled(const __half* __restrict__ A_half,
                                          const float* __restrict__ B_batch,
                                          float* __restrict__ C_batch,
                                          const __half* __restrict__ bias,
                                          int batch_size, int M, int K) {
    const int lane = threadIdx.x;   // 0..63
    const int warp = threadIdx.y;   // 0..RB-1
    const int row  = blockIdx.y * RB + warp;
    const int bidx = blockIdx.z;

    if (row >= M || bidx >= batch_size) return;

    const float* __restrict__ B = B_batch + bidx * K; // this batch’s vector
    float*       __restrict__ C = C_batch + bidx * M;

    const int a_base = row * K;
    float acc = 0.0f;

    __shared__ float sB[KTILE];

    const int threads = blockDim.x * blockDim.y;             // RB * 64
    const int tid     = warp * blockDim.x + lane;            // 0..threads-1

    for (int k0 = 0; k0 < K; k0 += KTILE) {
        const int Ktile = min(K - k0, KTILE);

        // --- Cooperative load of B tile into shared memory ---
        // Vectorized part (float4) — requires (k0 % 4 == 0). If not, scalar path still handles it.
        const int vecN = Ktile >> 2; // number of float4s
        for (int j4 = tid; j4 < vecN; j4 += threads) {
            // NOTE: Best performance when (reinterpret_cast<uintptr_t>(B + k0) % 4 == 0).
            reinterpret_cast<float4*>(sB)[j4] =
                *reinterpret_cast<const float4*>(&B[k0 + (j4 << 2)]);
        }
        // Tail (<=3)
        for (int j = (vecN << 2) + tid; j < Ktile; j += threads) {
            sB[j] = B[k0 + j];
        }
        __syncthreads();

        // --- Compute on this tile: one warp computes one row (dot product) ---
        const int vecRow = Ktile >> 2;
        for (int j4 = lane; j4 < vecRow; j4 += warpSize) {
            const int j = j4 << 2;

            // A[row, k0+j..k0+j+3] as two half2
            half2 a0 = *reinterpret_cast<const half2*>(&A_half[a_base + k0 + j + 0]);
            half2 a1 = *reinterpret_cast<const half2*>(&A_half[a_base + k0 + j + 2]);
            float2 af0 = __half22float2(a0);
            float2 af1 = __half22float2(a1);

            // B tile from shared
            float4 bv = *reinterpret_cast<const float4*>(&sB[j]);

            acc = fmaf(af0.x, bv.x, acc);
            acc = fmaf(af0.y, bv.y, acc);
            acc = fmaf(af1.x, bv.z, acc);
            acc = fmaf(af1.y, bv.w, acc);
        }
        // Tail inside tile
        for (int j = (vecRow << 2) + lane; j < Ktile; j += warpSize) {
            float b = sB[j];
            float a = __half2float(A_half[a_base + k0 + j]);
            acc = fmaf(a, b, acc);
        }
        __syncthreads(); // reuse sB for next tile
    }

    // Warp reduce (64 lanes)
    acc = warp_reduce_sum_64(acc);

    if (lane == 0) {
        float badd = bias ? __half2float(bias[row]) : 0.0f;
        C[row] = acc + badd;
    }
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void matvec_batch_gpu(float* xout, float* x_batch, __half* w, __half* bias,
                      int batch_size, int K /*d*/, int M /*n*/) {
    constexpr int RB    = 8;    // rows per block
    constexpr int KTILE = 512;  // K tile

    dim3 block(64, RB, 1);
    dim3 grid(1, ceil_div(M, RB), batch_size);

    hipLaunchKernelGGL(
        (matvec_batch_kernel_tiled<RB, KTILE>),
        grid, block, 0, 0,
        w, x_batch, xout, bias, batch_size, M, K
    );
    CHECK_HIP(hipGetLastError());
}
