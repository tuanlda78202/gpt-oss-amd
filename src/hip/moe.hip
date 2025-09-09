#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Small POD for device->host metadata (active_experts, max_Ne)
struct Int2 { int x; int y; };

// Local 64-lane reduction to avoid symbol conflicts
__device__ __forceinline__ float warp_reduce_sum64_local(float v) {
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 64);
    }
    return v;
}

// Build work queue and compute {active_experts, max_Ne}
__global__ void build_expert_work_queue_kernel(
    const int* __restrict__ expert_offsets, // [E+1]
    int* __restrict__ work_queue,           // [3*E] triplets {expert_id, off, Ne}
    Int2* __restrict__ meta,                // [1] {active_experts, max_Ne}
    int E)
{
    // single thread
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int active = 0;
    int maxNe  = 0;
    for (int e = 0; e < E; ++e) {
        int off = expert_offsets[e];
        int ne  = expert_offsets[e + 1] - off;
        if (ne > 0) {
            int i3 = active * 3;
            work_queue[i3 + 0] = e;
            work_queue[i3 + 1] = off;
            work_queue[i3 + 2] = ne;
            ++active;
            if (ne > maxNe) maxNe = ne;
        }
    }
    meta[0].x = active;
    meta[0].y = maxNe;
}

// -------- Multi-expert MATVEC (stream-aware) --------
template <int RB, int KTILE>
__global__ void multi_expert_matvec_kernel(
    const int* __restrict__ work_queue,   // [work_count * 3]
    int work_start, int work_count,       // slice (per stream)
    const float* __restrict__ x_by_expert,// [sum Ne, K]
    float* __restrict__ out_by_expert,    // [sum Ne, M]
    const __half* __restrict__ weights_base, // expert 0 base
    const __half* __restrict__ bias_base,    // expert 0 base (nullable)
    int K, int M,
    long long expert_weight_stride,
    long long expert_bias_stride)
{
    const int work_idx = blockIdx.z;
    if (work_idx >= work_count) return;

    const int q   = (work_start + work_idx) * 3;
    const int e   = work_queue[q + 0];
    const int off = work_queue[q + 1];
    const int Ne  = work_queue[q + 2];

    const __half* __restrict__ A_half = weights_base + (long long)e * expert_weight_stride;
    const __half* __restrict__ bias   = bias_base ? (bias_base + (long long)e * expert_bias_stride) : nullptr;

    const float* __restrict__ B_batch = x_by_expert + (long long)off * K;
    float*       __restrict__ C_batch = out_by_expert + (long long)off * M;

    const int lane = threadIdx.x;   // 0..63
    const int warp = threadIdx.y;   // 0..RB-1
    const int row  = blockIdx.y * RB + warp;
    const int bidx = blockIdx.x;    // 0..Ne-1

    if (row >= M || bidx >= Ne) return;

    const float* __restrict__ B = B_batch + (long long)bidx * K;
    float*       __restrict__ C = C_batch + (long long)bidx * M;

    const int a_base = row * K;
    float acc = 0.0f;

    __shared__ float sB[KTILE];
    const int threads = blockDim.x * blockDim.y; // RB * 64
    const int tid     = warp * blockDim.x + lane;

    for (int k0 = 0; k0 < K; k0 += KTILE) {
        const int rem   = K - k0;
        const int Ktile = (rem < KTILE) ? rem : KTILE;

        // cooperative load of B tile
        const int vecN = Ktile >> 2; // number of float4s
        for (int j4 = tid; j4 < vecN; j4 += threads) {
            reinterpret_cast<float4*>(sB)[j4] =
                *reinterpret_cast<const float4*>(&B[k0 + (j4 << 2)]);
        }
        for (int j = (vecN << 2) + tid; j < Ktile; j += threads) {
            sB[j] = B[k0 + j];
        }
        __syncthreads();

        const int vecRow = Ktile >> 2;
        for (int j4 = lane; j4 < vecRow; j4 += 64) {
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
        for (int j = (vecRow << 2) + lane; j < Ktile; j += 64) {
            const float b = sB[j];
            const float a = __half2float(A_half[a_base + k0 + j]);
            acc = fmaf(a, b, acc);
        }
        __syncthreads();
    }

    acc = warp_reduce_sum64_local(acc);

    if (lane == 0) {
        float badd = bias ? __half2float(bias[row]) : 0.0f;
        C[row] = acc + badd;
    }
}

__global__ void multi_expert_split_swiglu_fused_kernel(
    const int* __restrict__ work_queue,
    int work_start, int work_count,
    const float* __restrict__ mlp1_by_expert,  // [sum Ne, 2I]
    float* __restrict__ gate_by_expert,        // [sum Ne, I] (output)
    int I, float alpha, float swiglu_limit)
{
    const int work_idx = blockIdx.z;
    if (work_idx >= work_count) return;

    const int q   = (work_start + work_idx) * 3;
    const int off = work_queue[q + 1];
    const int Ne  = work_queue[q + 2];

    const int b = blockIdx.y; // 0..Ne-1
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..I-1
    if (b >= Ne || i >= I) return;

    const float* __restrict__ src = mlp1_by_expert + (long long)off * (2 * I);
    float*       __restrict__ dst = gate_by_expert + (long long)off * I;

    const int mlp_offset = b * (2 * I);
    const int out_offset = b * I + i;

    float gate_val = src[mlp_offset + 2 * i + 0];
    float up_val   = src[mlp_offset + 2 * i + 1];

    // clamp
    gate_val = fminf(gate_val,  swiglu_limit);
    up_val   = fminf(up_val,    swiglu_limit);
    up_val   = fmaxf(up_val,   -swiglu_limit);

    // SiLU
    gate_val *= (1.0f / (1.0f + expf(-alpha * gate_val)));

    // * (up + 1)
    gate_val *= (up_val + 1.0f);

    dst[out_offset] = gate_val;
}

__global__ void multi_expert_scale_scatter_kernel(
    const int* __restrict__ work_queue,
    int work_start, int work_count,
    const float* __restrict__ y_by_expert, // [sum Ne, H]
    const int* __restrict__ tokens_flat,   // [sum Ne]
    const float* __restrict__ weights_flat,// [sum Ne]
    float* __restrict__ e_agg,             // [B, H]
    int H, int B)
{
    const int work_idx = blockIdx.z;
    if (work_idx >= work_count) return;

    const int q   = (work_start + work_idx) * 3;
    const int off = work_queue[q + 1];
    const int Ne  = work_queue[q + 2];

    const int row = blockIdx.y; // 0..Ne-1
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..H-1
    if (row >= Ne || col >= H) return;

    const float* __restrict__ Y = y_by_expert + (long long)off * H;
    const int*   __restrict__ t = tokens_flat   + off;
    const float* __restrict__ w = weights_flat  + off;

    const int b = t[row];
    const float val = Y[row * H + col] * w[row];
    atomicAdd(&e_agg[(long long)b * H + col], val);
}

// ------------- Host wrappers (stream-aware) -------------
inline void multi_expert_matvec_gpu(
    const int* work_queue, int work_start, int work_count,
    const float* x_by_expert, float* out_by_expert,
    const __half* weights_base, const __half* bias_base,
    int K, int M, long long expert_weight_stride, long long expert_bias_stride,
    int maxNe, hipStream_t stream)
{
    constexpr int RB    = 8;
    constexpr int KTILE = 1024;
    dim3 block(64, RB, 1);
    dim3 grid(maxNe, (M + RB - 1) / RB, work_count);
    hipLaunchKernelGGL((multi_expert_matvec_kernel<RB, KTILE>),
        grid, block, 0, stream,
        work_queue, work_start, work_count,
        x_by_expert, out_by_expert,
        weights_base, bias_base,
        K, M, expert_weight_stride, expert_bias_stride);
}

inline void multi_expert_split_swiglu_gpu(
    const int* work_queue, int work_start, int work_count,
    const float* mlp1_by_expert, float* gate_by_expert,
    int I, float alpha, float swiglu_limit, int maxNe,
    hipStream_t stream)
{
    dim3 block(64);
    dim3 grid((I + 63) / 64, maxNe, work_count);
    hipLaunchKernelGGL(multi_expert_split_swiglu_fused_kernel,
        grid, block, 0, stream,
        work_queue, work_start, work_count,
        mlp1_by_expert, gate_by_expert, I, alpha, swiglu_limit);
}

inline void multi_expert_scale_scatter_gpu(
    const int* work_queue, int work_start, int work_count,
    const float* y_by_expert, const int* tokens_flat, const float* weights_flat,
    float* e_agg, int H, int batch_size, int maxNe,
    hipStream_t stream)
{
    dim3 block(256);
    dim3 grid((H + 255) / 256, maxNe, work_count);
    hipLaunchKernelGGL(multi_expert_scale_scatter_kernel,
        grid, block, 0, stream,
        work_queue, work_start, work_count,
        y_by_expert, tokens_flat, weights_flat,
        e_agg, H, batch_size);
}
