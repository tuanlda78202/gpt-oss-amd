#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------
// Device kernels
// ---------------------------

__global__ __launch_bounds__(256)
void convert_embedding_kernel_vec2(const __half* __restrict__ content_row_half,
                                   float* __restrict__ x,
                                   int hidden_dim)
{
    const int t = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (t >= hidden_dim) return;

    const __half2* src2 = reinterpret_cast<const __half2*>(content_row_half + t);
    float2 f2 = __half22float2(*src2);
    reinterpret_cast<float2*>(x + t)[0] = f2;

    if (t + 1 == hidden_dim) {
        x[t] = __half2float(content_row_half[t]);
    }
}

__global__ __launch_bounds__(256)
void convert_embedding_kernel_scalar(const __half* __restrict__ content_row_half,
                                     float* __restrict__ x,
                                     int hidden_dim)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < hidden_dim) {
        x[t] = __half2float(content_row_half[t]);
    }
}

template<int VEC>
__global__ __launch_bounds__(256)
void convert_embedding_batch_kernel_vec(const __half* __restrict__ emb_table,
                                        const int* __restrict__ tokens,
                                        float* __restrict__ x_batch,
                                        int batch_size, int hidden_dim)
{
    const int b = blockIdx.y;  // batch index
    const int t0 = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;

    if (b >= batch_size || t0 >= hidden_dim) return;

    // Broadcast token once per block
    __shared__ int tok_shared;
    if (threadIdx.x == 0) tok_shared = tokens[b];
    __syncthreads();
    const int token = tok_shared;

    const __half* base = emb_table + (size_t)token * hidden_dim + t0;
    float* out       = x_batch   + (size_t)b      * hidden_dim + t0;

    // Process in half2 chunks
    const int pairs = (hidden_dim - t0 >= VEC) ? (VEC >> 1) : ((hidden_dim - t0) >> 1);
#pragma unroll
    for (int i = 0; i < pairs; ++i) {
        const __half2 h2 = reinterpret_cast<const __half2*>(base)[i];
        const float2 f2  = __half22float2(h2);
        reinterpret_cast<float2*>(out)[i] = f2;
    }

    // Tail element if VEC is odd or if hidden_dim isn't divisible by 2
    const int done = pairs << 1;
    if (done < VEC && (t0 + done) < hidden_dim) {
        out[done] = __half2float(base[done]);
    }
}

__global__ __launch_bounds__(256)
void convert_embedding_batch_kernel_scalar(const __half* __restrict__ emb_table,
                                           const int* __restrict__ tokens,
                                           float* __restrict__ x_batch,
                                           int batch_size, int hidden_dim)
{
    const int b   = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || idx >= hidden_dim) return;

    __shared__ int tok_shared;
    if (threadIdx.x == 0) tok_shared = tokens[b];
    __syncthreads();
    const int token = tok_shared;

    const int emb_offset = token * hidden_dim + idx;
    const int out_offset = b     * hidden_dim + idx;
    x_batch[out_offset] = __half2float(emb_table[emb_offset]);
}

// ---------------------------
// Host wrappers
// ---------------------------

void embed_gpu(float* x, __half* content_row_half, int hidden_dim) {
    const int threads = 256;
    if ((hidden_dim & 1) == 0) {
        dim3 grid((hidden_dim/2 + threads - 1) / threads);
        dim3 block(threads);
        hipLaunchKernelGGL(convert_embedding_kernel_vec2,
                           grid, block, 0, 0,
                           content_row_half, x, hidden_dim);
    } else {
        dim3 grid((hidden_dim + threads - 1) / threads);
        dim3 block(threads);
        hipLaunchKernelGGL(convert_embedding_kernel_scalar,
                           grid, block, 0, 0,
                           content_row_half, x, hidden_dim);
    }
    CHECK_HIP(hipGetLastError());
}

void embed_batch_gpu(float* x_batch,
                     __half* emb_table,
                     int* tokens,
                     int batch_size,
                     int hidden_dim)
{
    const int threads = 256;

    // Prefer VEC=8 when aligned; otherwise VEC=2; otherwise scalar
    if ((hidden_dim & 7) == 0) {
        constexpr int VEC = 8;
        dim3 block(threads);
        dim3 grid((hidden_dim / VEC + threads - 1) / threads, batch_size);
        hipLaunchKernelGGL((convert_embedding_batch_kernel_vec<VEC>),
                           grid, block, 0, 0,
                           (const __half*)emb_table, (const int*)tokens, x_batch,
                           batch_size, hidden_dim);
    } else if ((hidden_dim & 1) == 0) {
        constexpr int VEC = 2;
        dim3 block(threads);
        dim3 grid((hidden_dim / VEC + threads - 1) / threads, batch_size);
        hipLaunchKernelGGL((convert_embedding_batch_kernel_vec<VEC>),
                           grid, block, 0, 0,
                           (const __half*)emb_table, (const int*)tokens, x_batch,
                           batch_size, hidden_dim);
    } else {
        dim3 block(threads);
        dim3 grid((hidden_dim + threads - 1) / threads, batch_size);
        hipLaunchKernelGGL(convert_embedding_batch_kernel_scalar,
                           grid, block, 0, 0,
                           (const __half*)emb_table, (const int*)tokens, x_batch,
                           batch_size, hidden_dim);
    }
    CHECK_HIP(hipGetLastError());
}
