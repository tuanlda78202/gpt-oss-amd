#include "BLAS.hip"
#include <algorithm>
#include <cstdio>

__global__ void topk_kernel(float* router_scores, float* topk_values, int* topk_indices,
                                  int batch_size, int num_experts, int experts_per_token) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int score_offset = batch_idx * num_experts;
    int result_offset = batch_idx * experts_per_token;

    float* batch_scores = router_scores + score_offset;
    float* batch_topk_values = topk_values + result_offset;
    int* batch_topk_indices = topk_indices + result_offset;

    __shared__ float shared_scores[1024];
    __shared__ int shared_indices[1024];
    int tid = threadIdx.x;

    // Load data into shared memory
    if (tid < num_experts) {
        shared_scores[tid] = batch_scores[tid];
        shared_indices[tid] = tid;
    } else {
        shared_scores[tid] = -3.402e+38f;
        shared_indices[tid] = -1;
    }
    __syncthreads();

    // Only thread 0 performs the sorting
    if (tid == 0) {
        for (int i = 0; i < experts_per_token && i < num_experts; i++) {
            int max_idx = i;
            float max_val = shared_scores[i];
            for (int j = i + 1; j < num_experts; j++) {
                if (shared_scores[j] > max_val) {
                    max_val = shared_scores[j];
                    max_idx = j;
                }
            }
            if (max_idx != i) {
                float temp_score = shared_scores[i];
                shared_scores[i] = shared_scores[max_idx];
                shared_scores[max_idx] = temp_score;

                int temp_index = shared_indices[i];
                shared_indices[i] = shared_indices[max_idx];
                shared_indices[max_idx] = temp_index;
            }
        }
    }
    __syncthreads();

    if (tid < experts_per_token) {
        batch_topk_values[tid] = shared_scores[tid];
        batch_topk_indices[tid] = shared_indices[tid];
    }
}

void topk(float* topk_values, int* topk_indices, float* router_scores,
                    int batch_size, int num_experts, int experts_per_token) {
    if (num_experts > 1024) {
        fprintf(stderr, "TOPK BATCH DEVICE Warning: num_experts=%d exceeds 1024\n", num_experts);
        return;
    }

    dim3 blockDim(1024);
    dim3 gridDim(batch_size);
    hipLaunchKernelGGL(topk_kernel, gridDim, blockDim, 0, 0,
                       router_scores, topk_values, topk_indices,
                       batch_size, num_experts, experts_per_token);
    CHECK_HIP(hipGetLastError());
}
