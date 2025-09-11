#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

template <int RB, int KTILE>
__global__ void matvec_kernel(const __half* __restrict__ A_half,
                                          const float* __restrict__ B_batch,
                                          float* __restrict__ C_batch,
                                          const __half* __restrict__ bias,
                                          int batch_size, int M, int K) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row  = blockIdx.y * RB + warp;
    const int bidx = blockIdx.z;

    if (row >= M || bidx >= batch_size) return;

    const float* __restrict__ B = B_batch + bidx * K;
    float*       __restrict__ C = C_batch + bidx * M;

    const int a_base = row * K;
    float acc = 0.0f;

    __shared__ float sB[KTILE];

    const int threads = blockDim.x * blockDim.y;
    const int tid     = warp * blockDim.x + lane;

    for (int k0 = 0; k0 < K; k0 += KTILE) {
        const int Ktile = min(K - k0, KTILE);

        const int vecN = Ktile >> 2;
        for (int j4 = tid; j4 < vecN; j4 += threads) {
            reinterpret_cast<float4*>(sB)[j4] =
                *reinterpret_cast<const float4*>(&B[k0 + (j4 << 2)]);
        }
        for (int j = (vecN << 2) + tid; j < Ktile; j += threads) {
            sB[j] = B[k0 + j];
        }
        __syncthreads();

        const int vecRow = Ktile >> 2;
        for (int j4 = lane; j4 < vecRow; j4 += warpSize) {
            const int j = j4 << 2;

            half2 a0 = *reinterpret_cast<const half2*>(&A_half[a_base + k0 + j + 0]);
            half2 a1 = *reinterpret_cast<const half2*>(&A_half[a_base + k0 + j + 2]);
            float2 af0 = __half22float2(a0);
            float2 af1 = __half22float2(a1);
            float4 bv = *reinterpret_cast<const float4*>(&sB[j]);

            acc = fmaf(af0.x, bv.x, acc);
            acc = fmaf(af0.y, bv.y, acc);
            acc = fmaf(af1.x, bv.z, acc);
            acc = fmaf(af1.y, bv.w, acc);
        }
        for (int j = (vecRow << 2) + lane; j < Ktile; j += warpSize) {
            float b = sB[j];
            float a = __half2float(A_half[a_base + k0 + j]);
            acc = fmaf(a, b, acc);
        }
        __syncthreads();
    }

    acc = warp_reduce_sum_64(acc);

    if (lane == 0) {
        float badd = bias ? __half2float(bias[row]) : 0.0f;
        C[row] = acc + badd;
    }
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void matvec(float* xout, float* x_batch, __half* w, __half* bias,
                      int batch_size, int K /*d*/, int M /*n*/) {
    constexpr int RB    = 8;     // rows per block
    constexpr int KTILE = 2048;  // K tile

    dim3 block(64, RB, 1);
    dim3 grid(1, ceil_div(M, RB), batch_size);

    hipLaunchKernelGGL(
        (matvec_kernel<RB, KTILE>),
        grid, block, 0, 0,
        w, x_batch, xout, bias, batch_size, M, K
    );
    CHECK_HIP(hipGetLastError());
}
