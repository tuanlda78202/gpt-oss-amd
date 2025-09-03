#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef WARPS_PER_BLOCK          // try 8 for large M, 4 for mid, 2 for tiny
#define WARPS_PER_BLOCK 8
#endif

#ifndef TILE_K                   // must be even; 512 is great for large K
#define TILE_K 512
#endif

#ifndef LDS_PAD                  // to mitigate bank conflicts
#define LDS_PAD 32
#endif

#ifndef UNROLL_PAIR
#define UNROLL_PAIR 4
#endif

#ifndef USE_LOADER_WARP          // warp 0 prefetches next x tile while others compute
#define USE_LOADER_WARP 1
#endif

#ifndef USE_NONTEMPORAL_A        // use non-temporal loads for streaming A (AMD/Clang)
#define USE_NONTEMPORAL_A 0
#endif

#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * warpSize)

#if USE_NONTEMPORAL_A
  #if defined(__has_builtin)
    #if __has_builtin(__builtin_nontemporal_load)
      #define NTA_LOAD(TPTR) __builtin_nontemporal_load(TPTR)
    #else
      #define NTA_LOAD(TPTR) (*(TPTR))
    #endif
  #else
    #define NTA_LOAD(TPTR) (*(TPTR))
  #endif
#else
  #define NTA_LOAD(TPTR) (*(TPTR))
#endif

// __device__ __forceinline__ float warp_reduce_sum(float v) {
    // for (int off = warpSize >> 1; off > 0; off >>= 1) {
        // v += __shfl_down(v, off, warpSize);
    // }
    // return v;
// }

extern "C"
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void matvec_hybrid_warp(
    const __half* __restrict__ A_half,
    const float*  __restrict__ x,
    float*        __restrict__ y,
    const __half* __restrict__ bias,
    int M, int K)
{
    // Double-buffered LDS tiles of x
    __shared__ float sX[2][TILE_K + LDS_PAD];

    const int lane = threadIdx.x;    // 0..63 (AMD warpSize = 64)
    const int warp = threadIdx.y;    // 0..WARPS_PER_BLOCK-1
    const int warp_rows_stride = gridDim.y * blockDim.y;

    // Persistent loop: each warp processes multiple rows (row += total warps in the grid)
    for (int row = blockIdx.y * blockDim.y + warp; row < M; row += warp_rows_stride) {

        float acc = 0.0f;

        // --- Preload first tile of x into buffer 0 ---
        int buf = 0;
        int k0  = 0;

        // Loader warp prefetches x tile 0
        if (USE_LOADER_WARP && warp == 0) {
            const float* xptr = x; // k0 == 0
            const bool aligned16 = (((uintptr_t)xptr) & 0xF) == 0;
            int tile = min(TILE_K, K);

            if (aligned16) {
                // vectorized float4 loads
                int i4 = lane; // each lane handles indices in steps of warpSize
                for (int off = i4 * 4; off < tile; off += warpSize * 4) {
                    float4 v = *reinterpret_cast<const float4*>(xptr + off);
                    // store to LDS
                    if (off + 0 < tile) sX[buf][off + 0] = v.x;
                    if (off + 1 < tile) sX[buf][off + 1] = v.y;
                    if (off + 2 < tile) sX[buf][off + 2] = v.z;
                    if (off + 3 < tile) sX[buf][off + 3] = v.w;
                }
            } else {
                // scalar fallback
                for (int i = lane; i < tile; i += warpSize) {
                    sX[buf][i] = xptr[i];
                }
            }
        } else if (!USE_LOADER_WARP) {
            // No dedicated loader: all warps help stage x
            int tile = min(TILE_K, K);
            for (int i = lane; i < tile; i += warpSize) {
                sX[buf][i] = x[i];
            }
        }
        __syncthreads();

        // --- Main tile loop with double-buffered prefetch ---
        while (k0 < K) {
            const int tile = min(TILE_K, K - k0);

            // Next tile prefetch by loader warp while others compute
            const int next_k0  = k0 + tile;
            const int next_buf = buf ^ 1;

            if (USE_LOADER_WARP && warp == 0 && next_k0 < K) {
                const float* xptr = x + next_k0;
                const bool aligned16 = (((uintptr_t)xptr) & 0xF) == 0;

                if (aligned16) {
                    for (int off = lane * 4; off < min(TILE_K, K - next_k0); off += warpSize * 4) {
                        float4 v = *reinterpret_cast<const float4*>(xptr + off);
                        if (off + 0 < TILE_K && next_k0 + off + 0 < K) sX[next_buf][off + 0] = v.x;
                        if (off + 1 < TILE_K && next_k0 + off + 1 < K) sX[next_buf][off + 1] = v.y;
                        if (off + 2 < TILE_K && next_k0 + off + 2 < K) sX[next_buf][off + 2] = v.z;
                        if (off + 3 < TILE_K && next_k0 + off + 3 < K) sX[next_buf][off + 3] = v.w;
                    }
                } else {
                    for (int i = lane; i < min(TILE_K, K - next_k0); i += warpSize) {
                        sX[next_buf][i] = xptr[i];
                    }
                }
            }

            // ----- Compute on current tile (buf) -----
            const __half* a_row = A_half + (size_t)row * K + k0;

            // Align to 4B for half2 vectorized loads
            int start = 0;
            if (((uintptr_t)a_row & 0x2) && tile > 0) {
                if (lane == 0) {
                    float a0 = __half2float(a_row[0]);
                    acc = fmaf(a0, sX[buf][0], acc);
                }
                start = 1;
            }

            const int pairs = (tile - start) >> 1;
            const __half2* __restrict__ a2 = reinterpret_cast<const __half2*>(a_row + start);

            #pragma unroll UNROLL_PAIR
            for (int p = lane; p < pairs; p += warpSize) {
                // optional non-temporal (streaming) load for A
                __half2 h2 = NTA_LOAD(a2 + p);
                float2 af  = __half22float2(h2);
                const int j = (p << 1) + start;
                acc = fmaf(af.x, sX[buf][j],     acc);
                acc = fmaf(af.y, sX[buf][j + 1], acc);
            }

            if (((tile - start) & 1) != 0) {
                const int j = start + (pairs << 1);
                // share the scalar tail
                if (lane == 0) {
                    float a = __half2float(a_row[j]);
                    acc = fmaf(a, sX[buf][j], acc);
                }
            }

            // --- Barrier: ensure all consumers finished before swapping buffers
            __syncthreads();

            // If not using loader warp, cooperatively stage the next tile now
            if (!USE_LOADER_WARP) {
                if (next_k0 < K) {
                    int ntile = min(TILE_K, K - next_k0);
                    for (int i = lane; i < ntile; i += warpSize) {
                        sX[next_buf][i] = x[next_k0 + i];
                    }
                }
            }

            // Ensure the loader finished writing next_buf before we use it next iter
            __syncthreads();

            // Advance
            k0  = next_k0;
            buf = next_buf;
        }

        // Reduce within warp and write out
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            const float b = bias ? __half2float(bias[row]) : 0.0f;
            y[row] = acc + b;
        }
        __syncthreads(); // keep waves roughly in lockstep across rows (helps on some SKUs)
    }
}

// ---- Host launcher (unchanged signature) ----
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

extern "C" void matvec_gpu(float* __restrict__ y,
                           const float* __restrict__ x,
                           const __half* __restrict__ A_half,
                           const __half* __restrict__ bias,
                           int K,  // columns of A (len of x)
                           int M)  // rows of A (len of y)
{
    int wpb = (M >= 2048 ? WARPS_PER_BLOCK :
              (M >=  512 ? max(4, WARPS_PER_BLOCK/2) : max(2, WARPS_PER_BLOCK/4)));

    dim3 block_dim(warpSize, wpb, 1);                 // (64, wpb)
    dim3 grid_dim(1, ceil_div(M, wpb), 1);

    hipLaunchKernelGGL(matvec_hybrid_warp, grid_dim, block_dim, 0, 0,
                       A_half, x, y, bias, M, K);
    CHECK_HIP(hipGetLastError());
}
