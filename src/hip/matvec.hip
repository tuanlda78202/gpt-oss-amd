#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef K_UNROLL
#define K_UNROLL 4
#endif

template <int RB /* rows per block */>
__global__ void matvec_warp_splitk_vec4_u(const __half* __restrict__ A_half,
                                          const float* __restrict__ B, float* __restrict__ C,
                                          const __half* __restrict__ bias, int M, int K) {
    const int lane = threadIdx.x; // 0..63
    const int warp = threadIdx.y; // 0..RB-1
    const int row = blockIdx.y * RB + warp;
    if (row >= M)
        return;

    const int a_base = row * K;
    float acc = 0.0f;

    // Work in units of 4 for vectorization
    const int K4 = K >> 2; // floor(K/4)

    // Each lane walks k4 in steps of warpSize
    int k4 = lane;

    // Prefetch registers
    float4 bv[K_UNROLL];
    half2 a0[K_UNROLL], a1[K_UNROLL];
    float2 af0[K_UNROLL], af1[K_UNROLL];

// Preload first batch
#pragma unroll
    for (int u = 0; u < K_UNROLL; ++u) {
        int k4u = k4 + u * WARP_SIZE;
        if (k4u < K4) {
            const int k = (k4u << 2);
            bv[u] = *reinterpret_cast<const float4*>(&B[k]);
            a0[u] = *reinterpret_cast<const half2*>(&A_half[a_base + k + 0]);
            a1[u] = *reinterpret_cast<const half2*>(&A_half[a_base + k + 2]);
            af0[u] = __half22float2(a0[u]);
            af1[u] = __half22float2(a1[u]);
        }
    }

    for (;;) {
// Compute FMAs on the prefetched batch
#pragma unroll
        for (int u = 0; u < K_UNROLL; ++u) {
            int k4u = k4 + u * WARP_SIZE;
            if (k4u < K4) {
                acc = fmaf(af0[u].x, bv[u].x, acc);
                acc = fmaf(af0[u].y, bv[u].y, acc);
                acc = fmaf(af1[u].x, bv[u].z, acc);
                acc = fmaf(af1[u].y, bv[u].w, acc);
            }
        }

        k4 += K_UNROLL * WARP_SIZE;
        if (k4 >= K4)
            break;

// Prefetch next batch while previous FMAs are in flight
#pragma unroll
        for (int u = 0; u < K_UNROLL; ++u) {
            int k4u = k4 + u * WARP_SIZE;
            if (k4u < K4) {
                const int k = (k4u << 2);
                bv[u] = *reinterpret_cast<const float4*>(&B[k]);
                a0[u] = *reinterpret_cast<const half2*>(&A_half[a_base + k + 0]);
                a1[u] = *reinterpret_cast<const half2*>(&A_half[a_base + k + 2]);
                af0[u] = __half22float2(a0[u]);
                af1[u] = __half22float2(a1[u]);
            }
        }
    }

    // Tail for K % 4
    for (int kt = (K4 << 2) + lane; kt < K; kt += WARP_SIZE) {
        acc = fmaf(__half2float(A_half[a_base + kt]), B[kt], acc);
    }

    // Warp reduce
    acc = warp_reduce_sum_64(acc);

    if (lane == 0) {
        C[row] = acc + (bias ? __half2float(bias[row]) : 0.0f);
    }
}

void matvec_gpu(float* xout, float* x, const __half* w, const __half* bias, int n, int d) {
    constexpr int RB = 4;
    dim3 block(WARP_SIZE, RB, 1);
    dim3 grid(1, (d + RB - 1) / RB, 1);
    hipLaunchKernelGGL((matvec_warp_splitk_vec4_u<RB>), grid, block, 0, 0, w, x, xout, bias, d, n);
    CHECK_HIP(hipGetLastError());
}

// ! Batched
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
    constexpr int KTILE = 1024;  // K tile

    dim3 block(64, RB, 1);
    dim3 grid(1, ceil_div(M, RB), batch_size);

    hipLaunchKernelGGL(
        (matvec_batch_kernel_tiled<RB, KTILE>),
        grid, block, 0, 0,
        w, x_batch, xout, bias, batch_size, M, K
    );
    CHECK_HIP(hipGetLastError());
}
