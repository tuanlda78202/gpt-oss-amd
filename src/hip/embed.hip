#include "BLAS.hip"

__global__ void convert_embedding_kernel(__half* content_row_half, float* x, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        x[idx] = __half2float(content_row_half[idx]);
    }
}

void embed_gpu(float* x, __half* content_row_half, int hidden_dim) {
    dim3 embed_block(256);
    dim3 embed_grid((hidden_dim + 255) / 256);
    hipLaunchKernelGGL(convert_embedding_kernel, embed_grid, embed_block, 0, 0, content_row_half, x,
                       hidden_dim);
    CHECK_HIP(hipGetLastError());
}
