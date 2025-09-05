#include "BLAS.hip"

__global__ void convert_embedding_kernel(__half* content_row_half, float* x, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        x[idx] = __half2float(content_row_half[idx]);
    }
}

__global__ void convert_embedding_batch_kernel(__half* emb_table, int* tokens, float* x_batch,
                                               int batch_size, int hidden_dim) {
    int b = blockIdx.y;                          // batch index
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // feature index

    if (b < batch_size && idx < hidden_dim) {
        int token = tokens[b];
        int emb_offset = token * hidden_dim + idx;
        int out_offset = b * hidden_dim + idx;
        x_batch[out_offset] = __half2float(emb_table[emb_offset]);
    }
}

void embed_gpu(float* x, __half* content_row_half, int hidden_dim) {
    dim3 embed_block(256);
    dim3 embed_grid((hidden_dim + 255) / 256);
    hipLaunchKernelGGL(convert_embedding_kernel, embed_grid, embed_block, 0, 0, content_row_half, x,
                       hidden_dim);
    CHECK_HIP(hipGetLastError());
}

void embed_batch_gpu(float* x_batch, __half* emb_table, int* tokens, int batch_size, int hidden_dim) {
    dim3 block(256);
    dim3 grid((hidden_dim + 255) / 256, batch_size);
    hipLaunchKernelGGL(convert_embedding_batch_kernel, grid, block, 0, 0, emb_table, tokens, x_batch,
                       batch_size, hidden_dim);
    CHECK_HIP(hipGetLastError());
}
