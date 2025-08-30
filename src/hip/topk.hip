#include "../../include/hip_helper.hpp"
#include "../BLAS.hip"
#include <algorithm>
#include <cstdio>

// ====== SIMPLE AND CORRECT TOP-K KERNEL ======
__global__ void topk_kernel(float* router_score, float* topk_values, int* topk_indices,
                            int num_experts, int experts_per_token) {
    // Use shared memory for the scores and indices
    __shared__ float shared_scores[1024];
    __shared__ int shared_indices[1024];
    int tid = threadIdx.x;
    // Load data into shared memory
    if (tid < num_experts) {
        shared_scores[tid] = router_score[tid];
        shared_indices[tid] = tid;
    } else {
        shared_scores[tid] = -3.402e+38f; // negative infinity for padding
        shared_indices[tid] = -1;
    }
    __syncthreads();
    // Only thread 0 performs the sorting to avoid race conditions
    if (tid == 0) {
        // Simple selection sort for top-k (efficient for small k)
        for (int i = 0; i < experts_per_token && i < num_experts; i++) {
            // Find the maximum value from position i to the end
            int max_idx = i;
            float max_val = shared_scores[i];
            for (int j = i + 1; j < num_experts; j++) {
                if (shared_scores[j] > max_val) {
                    max_val = shared_scores[j];
                    max_idx = j;
                }
            }
            // Swap if necessary
            if (max_idx != i) {
                // Swap scores
                float temp_score = shared_scores[i];
                shared_scores[i] = shared_scores[max_idx];
                shared_scores[max_idx] = temp_score;
                // Swap indices
                int temp_index = shared_indices[i];
                shared_indices[i] = shared_indices[max_idx];
                shared_indices[max_idx] = temp_index;
            }
        }
    }
    __syncthreads();
    // Write results (all threads can participate in writing)
    if (tid < experts_per_token) {
        topk_values[tid] = shared_scores[tid];
        topk_indices[tid] = shared_indices[tid];
    }
}

void topk_hip_device(float* topk_values, int* topk_indices, float* router_score, int num_experts,
                     int experts_per_token) {
    if (num_experts <= 0 || experts_per_token <= 0 || experts_per_token > num_experts) {
        fprintf(stderr, "TOPK DEVICE Invalid parameters: num_experts=%d, experts_per_token=%d\n",
                num_experts, experts_per_token);
        return;
    }
    if (!router_score || !topk_values || !topk_indices) {
        fprintf(stderr, "TOPK DEVICE Null pointer detected\n");
        return;
    }

    if (num_experts > 1024) {
        fprintf(stderr, "TOPK DEVICE Warning: num_experts=%d exceeds 1024\n", num_experts);
        return;
    }

    // Launch kernel directly on device data
    dim3 blockDim(1024);
    dim3 gridDim(1);
    hipLaunchKernelGGL(topk_kernel, gridDim, blockDim, 0, 0, router_score, topk_values,
                       topk_indices, num_experts, experts_per_token);

    CHECK_HIP(hipGetLastError());
}
