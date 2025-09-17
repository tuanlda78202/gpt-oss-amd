#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

static inline int ceil_div1(int a, int b) { return (a + b - 1) / b; }
static inline __device__ int imin(int a, int b) { return a < b ? a : b; }

template<int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_kernel( // Compute C^T = A (M×K) * B^T (K×N)
    const __hip_bfloat16* __restrict__ A_half, // [M,K] row-major
    const float*          __restrict__ B_batch, // [N,K] row-major (N = batch_size)
    float*                __restrict__ C_batch, // [N,M] row-major (matches your layout)
    const __hip_bfloat16* __restrict__ bias,    // [M] (optional, BF16)
    int N, int M, int K)                        // N=batch_size
{
    // Block tile origin
    const int n0 = blockIdx.x * BN; // along batch (N)
    const int m0 = blockIdx.y * BM; // along rows of A/C

    // Thread coordinates inside the block
    const int tx = threadIdx.x; // [0..TBX)
    const int ty = threadIdx.y; // [0..TBY)

    constexpr int TBX = 16; // 16x16=256 threads
    constexpr int TBY = 16;

    // Each thread computes a TM×TN register tile
    float acc[TM][TN];
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm)
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn)
        acc[tm][tn] = 0.0f;

    // Shared memory: store A as BF16 (2B) and B as FP32 (4B)
    __shared__ __hip_bfloat16 sA[BM * BK]; // (BM×BK) * 2B
    __shared__ float          sB[BK * BN]; // (BK×BN) * 4B

    // Per-thread starting element in output tile
    const int m_thread_base = ty * TM; // 0..BM-1
    const int n_thread_base = tx * TN; // 0..BN-1

    // Loop across K in tiles
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int kTile = imin(BK, K - k0);

        // --- Cooperative load of A tile: [BM, kTile] from A_half ---
        {
            const int tid = ty * TBX + tx;
            const int num_threads = TBX * TBY;
            const int elements = BM * kTile; // only the valid part
            for (int p = tid; p < elements; p += num_threads) {
                const int i = p / kTile; // 0..BM-1
                const int j = p % kTile; // 0..BK-1 (valid part)
                const int m = m0 + i;
                const int k = k0 + j;
                sA[i * BK + j] = (m < M && k < K)
                    ? A_half[m * K + k]
                    : __float2bfloat16(0.0f);
            }
            // Zero-pad the remainder of the A tile if kTile < BK
            for (int p = tid + elements; p < BM * BK; p += num_threads) {
                sA[p] = __float2bfloat16(0.0f);
            }
        }

        // --- Cooperative load of B tile: [kTile, BN] from B_batch (note B is [N,K] row-major) ---
        {
            const int tid = ty * TBX + tx;
            const int num_threads = TBX * TBY;
            const int elements = kTile * BN;
            for (int p = tid; p < elements; p += num_threads) {
                const int i = p / BN; // 0..kTile-1 (k within the tile)
                const int j = p % BN; // 0..BN-1 (n within the tile)
                const int k = k0 + i;
                const int n = n0 + j;
                sB[i * BN + j] = (k < K && n < N)
                    ? B_batch[n * K + k] // row-major per batch row
                    : 0.0f;
            }
            // Zero-pad if kTile < BK
            for (int p = tid + elements; p < BK * BN; p += num_threads) {
                sB[p] = 0.0f;
            }
        }

        __syncthreads();

        // --- Compute on the loaded tiles ---
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // Guard: for the last k tile, kk >= kTile is zero-padded
            // Load a strip of A for our TM rows
            float a_frag[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                const int mi = m_thread_base + tm;
                a_frag[tm] = __bfloat162float(sA[mi * BK + kk]);
            }
            // Multiply-accumulate against TN columns from B
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                const int nj = n_thread_base + tn;
                const float b = sB[kk * BN + nj];
                #pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    acc[tm][tn] = fmaf(a_frag[tm], b, acc[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // --- Write back with bias (bias is per-row m) ---
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        const int m = m0 + m_thread_base + tm;
        if (m >= M) break;

        const float badd = (bias != nullptr) ? __bfloat162float(bias[m]) : 0.0f;

        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            const int n = n0 + n_thread_base + tn;
            if (n < N) {
                // C layout is [N, M] row-major (same as your current code: per-batch row contiguous)
                C_batch[n * M + m] = acc[tm][tn] + badd;
            }
        }
    }
}

void gemm(
    float* xout,
    const float* x_batch,
    const __hip_bfloat16* w,
    const __hip_bfloat16* bias,
    int batch_size, // N
    int K,          // K
    int M)          // M
{
    // Tile config: 128×64×64 with 8×4 per-thread micro-tile, 256 threads/block
    constexpr int BM = 128, BN = 64, BK = 64, TM = 8, TN = 4;

    dim3 block(16, 16, 1); // TBX, TBY
    dim3 grid(ceil_div1(batch_size, BN), ceil_div1(M, BM), 1);

    hipLaunchKernelGGL(
        (gemm_kernel<BM, BN, BK, TM, TN>),
        grid, block, 0, 0,
        w, x_batch, xout, bias,
        batch_size, M, K
    );
    CHECK_HIP(hipGetLastError());
}
