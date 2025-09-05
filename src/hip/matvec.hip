#include "BLAS.hip"

// 64-wide wave reduce
__device__ __forceinline__ float warp_reduce_sum_64(float v) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_down(v, off);
    return v;
}

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
