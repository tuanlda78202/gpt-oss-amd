#include "BLAS.hip"

#define VEC 4

template <const int BM, const int BK>
__global__ void matvec_logits(const __half* __restrict__ A_half, const float* __restrict__ B,
                              float* __restrict__ C, const __half* __restrict__ bias, int M,
                              int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK];

    int row = blockIdx.y * BM + threadIdx.y;
    float acc = 0.0f;

    const int k_vec = threadIdx.x * VEC;
    const int b_vec_idx = threadIdx.y * VEC;

    for (int t = 0; t < CEIL_DIV(K, BK); ++t) {
        int k_base = t * BK;

        // --- Load A[row][k] ---
        float4 a_vec = make_float4(0, 0, 0, 0);
        int a_offset = row * K + k_base + k_vec;

        if (row < M && (k_base + k_vec + 3) < K) {
            half2 h0 = *reinterpret_cast<const half2*>(&A_half[a_offset + 0]);
            half2 h1 = *reinterpret_cast<const half2*>(&A_half[a_offset + 2]);
            float2 f0 = __half22float2(h0);
            float2 f1 = __half22float2(h1);
            a_vec = make_float4(f0.x, f0.y, f1.x, f1.y);
        }

#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            int k = k_vec + i;
            if ((k < BK) && (k_base + k < K)) {
                As[threadIdx.y][k] = reinterpret_cast<float*>(&a_vec)[i];
            }
        }

        // --- Load B[k] ---
        float4 b_vec = make_float4(0, 0, 0, 0);
        int b_offset = k_base + b_vec_idx;

        if ((b_offset + 3) < K) {
            b_vec = *reinterpret_cast<const float4*>(&B[b_offset]);
        }

#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            int k = b_vec_idx + i;
            if ((k < BK) && (k_base + k < K)) {
                Bs[k] = reinterpret_cast<float*>(&b_vec)[i];
            }
        }

        __syncthreads();

        // Compute dot product
        for (int i = 0; i < BK; ++i) {
            acc += As[threadIdx.y][i] * Bs[i];
        }

        __syncthreads();
    }

    if (row < M) {
        float b = bias ? __half2float(bias[row]) : 0.0f;
        C[row] = acc + b;
    }
}

// Each warp computes one row; block has RB warps (RB rows) → 64*RB threads.
// Good defaults: RB = 8 (512 threads/block) or RB = 4 (256).
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
