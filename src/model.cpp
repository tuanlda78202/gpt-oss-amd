#include "../include/model.hpp"
void copy_large_tensor_streaming(__half** d_ptr, float* h_ptr, size_t total_size,
                                 const char* tensor_name) {
    const size_t chunk_size = 512 * 1024 * 1024; // 512MB chunks

    // Only show progress for large tensors (>1GB)
    bool show_progress = total_size > 1024 * 1024 * 1024;

    if (show_progress) {
        printf("  %s (%.1f GB)...", tensor_name, total_size / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);
    }

    // Allocate temporary conversion buffer
    __half* conversion_buffer = (__half*)malloc(chunk_size);
    if (!conversion_buffer) {
        fprintf(stderr, "Failed to allocate conversion buffer for %s\n", tensor_name);
        exit(EXIT_FAILURE);
    }

    size_t bytes_processed = 0;
    size_t chunks_processed = 0;
    size_t total_elements = total_size / sizeof(__half);

    while (bytes_processed < total_size) {
        size_t current_chunk_bytes = (total_size - bytes_processed < chunk_size)
                                         ? (total_size - bytes_processed)
                                         : chunk_size;
        size_t current_chunk_elements = current_chunk_bytes / sizeof(__half);

        // Convert current chunk from float to half
        size_t element_offset = bytes_processed / sizeof(__half);
        for (size_t i = 0; i < current_chunk_elements; i++) {
            conversion_buffer[i] = __float2half(h_ptr[element_offset + i]);
        }

        // Transfer converted chunk directly to GPU
        CHECK_HIP(hipMemcpy((char*)(*d_ptr) + bytes_processed, conversion_buffer,
                            current_chunk_bytes, hipMemcpyHostToDevice));

        bytes_processed += current_chunk_bytes;
        chunks_processed++;

        if (show_progress && chunks_processed % 8 == 0) {
            printf(".");
            fflush(stdout);
        }
    }

    free(conversion_buffer);

    if (show_progress) {
        printf(" done\n");
    }
}

// ! Hybrid precision (FP16 weights + FP32 activations)
void copy_transformer_to_device_hybrid(OssTransformer* t_fp32, OssTransformerHybrid* t_d) {
    memcpy(&t_d->config, &t_fp32->config, sizeof(OssConfig));

    OssConfig* conf = &t_fp32->config;
    OssTransformerWeights* weights = &t_fp32->weights;
    OssRunState* state = &t_fp32->state;

    int vocab_size = conf->vocab_size;
    int hidden_dim = conf->hidden_dim;
    int n_experts = conf->n_experts;
    int experts_per_token = conf->experts_per_token;
    int intermediate_dim = conf->intermediate_dim;
    int n_layers = conf->n_layers;
    int head_dim = conf->head_dim;
    int n_attn_heads = conf->n_attn_heads;
    int n_kv_heads = conf->n_kv_heads;
    int seq_len = conf->seq_len;

    printf("\nConverting model to hybrid precision (FP16 weights + FP32 activations)...\n");

    // ! GPU Check
    int current_device;
    CHECK_HIP(hipGetDevice(&current_device));

    hipDeviceProp_t deviceProp;
    CHECK_HIP(hipGetDeviceProperties(&deviceProp, current_device));

    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    printf("GPU %d (%s): %.1f GB free / %.1f GB total\n", current_device, deviceProp.name,
           free_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0));

    // ! Allocate FP16 weights on GPU
    size_t token_emb_size = (size_t)vocab_size * hidden_dim * sizeof(__half);
    size_t mlp1_size =
        1ll * n_layers * n_experts * 2 * intermediate_dim * hidden_dim * sizeof(__half);
    size_t mlp2_size = 1ll * n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(__half);

    CHECK_HIP(hipMalloc(&t_d->weights.token_embedding_table, token_emb_size));
    CHECK_HIP(hipMalloc(&t_d->weights.rms_attn_w, (size_t)n_layers * hidden_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->weights.rms_ffn_w, (size_t)n_layers * hidden_dim * sizeof(__half)));

    size_t qkv_size = (size_t)n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) *
                      hidden_dim * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.w_qkv, qkv_size));

    size_t w_o_size = (size_t)n_layers * hidden_dim * head_dim * n_attn_heads * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.w_o, w_o_size));

    size_t b_qkv_size =
        (size_t)n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.b_qkv, b_qkv_size));
    CHECK_HIP(hipMalloc(&t_d->weights.b_o, (size_t)n_layers * hidden_dim * sizeof(__half)));
    CHECK_HIP(
        hipMalloc(&t_d->weights.attn_sinks, (size_t)n_layers * n_attn_heads * sizeof(__half)));

    size_t router_size = (size_t)n_layers * hidden_dim * n_experts * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.w_router, router_size));
    CHECK_HIP(hipMalloc(&t_d->weights.b_router, (size_t)n_layers * n_experts * sizeof(__half)));

    CHECK_HIP(hipMalloc(&t_d->weights.w_mlp1, mlp1_size));
    CHECK_HIP(hipMalloc(&t_d->weights.w_mlp2, mlp2_size));

    size_t b_mlp1_size = (size_t)n_layers * n_experts * 2 * intermediate_dim * sizeof(__half);
    size_t b_mlp2_size = (size_t)n_layers * n_experts * hidden_dim * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.b_mlp1, b_mlp1_size));
    CHECK_HIP(hipMalloc(&t_d->weights.b_mlp2, b_mlp2_size));

    CHECK_HIP(hipMalloc(&t_d->weights.rms_out_w, (size_t)hidden_dim * sizeof(__half)));
    size_t out_size = (size_t)vocab_size * hidden_dim * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.out, out_size));

    // ! Allocate FP32 state on GPU
    CHECK_HIP(hipMalloc(&t_d->state.x, (size_t)hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.t, (size_t)hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.tb, (size_t)head_dim * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.tb2, (size_t)hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.router_score, (size_t)n_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.topk_v, (size_t)experts_per_token * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.topk_i, (size_t)experts_per_token * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.mlp1_out, (size_t)2 * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.gate, (size_t)intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.up, (size_t)intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.gate_up, (size_t)intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.e_agg, (size_t)hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.qkv,
                        (size_t)head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.q, (size_t)n_attn_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.k, (size_t)n_kv_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.v, (size_t)n_kv_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.att, (size_t)n_attn_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.logits, (size_t)vocab_size * sizeof(float)));

    size_t key_cache_size = 1ll * n_layers * seq_len * n_kv_heads * head_dim * sizeof(float);
    size_t value_cache_size = 1ll * n_layers * seq_len * n_kv_heads * head_dim * sizeof(float);
    CHECK_HIP(hipMalloc(&t_d->state.key_cache, key_cache_size));
    CHECK_HIP(hipMalloc(&t_d->state.value_cache, value_cache_size));
    CHECK_HIP(hipMalloc(&t_d->state.mask, (size_t)seq_len * seq_len * sizeof(float)));

    printf("Converting and transferring weights...\n");

    // ! Stream conversion and transfer for weights (FP32 -> FP16)
    copy_large_tensor_streaming(&t_d->weights.token_embedding_table, weights->token_embedding_table,
                                token_emb_size, "token_embedding_table");
    copy_large_tensor_streaming(&t_d->weights.w_qkv, weights->w_qkv, qkv_size, "w_qkv");
    copy_large_tensor_streaming(&t_d->weights.w_o, weights->w_o, w_o_size, "w_o");
    copy_large_tensor_streaming(&t_d->weights.w_router, weights->w_router, router_size, "w_router");
    copy_large_tensor_streaming(&t_d->weights.w_mlp1, weights->w_mlp1, mlp1_size, "w_mlp1");
    copy_large_tensor_streaming(&t_d->weights.w_mlp2, weights->w_mlp2, mlp2_size, "w_mlp2");
    copy_large_tensor_streaming(&t_d->weights.b_mlp1, weights->b_mlp1, b_mlp1_size, "b_mlp1");
    copy_large_tensor_streaming(&t_d->weights.b_mlp2, weights->b_mlp2, b_mlp2_size, "b_mlp2");
    copy_large_tensor_streaming(&t_d->weights.out, weights->out, out_size, "out");

    // Convert small tensors directly
    __half* small_buffer = (__half*)malloc(1024 * 1024 * sizeof(__half));
    if (!small_buffer) {
        fprintf(stderr, "Failed to allocate small conversion buffer\n");
        exit(EXIT_FAILURE);
    }

    // Convert and copy small weights
    size_t rms_size = n_layers * hidden_dim;
    for (size_t i = 0; i < rms_size; i++) {
        small_buffer[i] = __float2half(weights->rms_attn_w[i]);
    }
    CHECK_HIP(hipMemcpy(t_d->weights.rms_attn_w, small_buffer, rms_size * sizeof(__half),
                        hipMemcpyHostToDevice));

    for (size_t i = 0; i < rms_size; i++) {
        small_buffer[i] = __float2half(weights->rms_ffn_w[i]);
    }
    CHECK_HIP(hipMemcpy(t_d->weights.rms_ffn_w, small_buffer, rms_size * sizeof(__half),
                        hipMemcpyHostToDevice));

    size_t b_qkv_elements = n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads);
    for (size_t i = 0; i < b_qkv_elements; i++) {
        small_buffer[i] = __float2half(weights->b_qkv[i]);
    }
    CHECK_HIP(hipMemcpy(t_d->weights.b_qkv, small_buffer, b_qkv_elements * sizeof(__half),
                        hipMemcpyHostToDevice));

    for (size_t i = 0; i < rms_size; i++) {
        small_buffer[i] = __float2half(weights->b_o[i]);
    }
    CHECK_HIP(hipMemcpy(t_d->weights.b_o, small_buffer, rms_size * sizeof(__half),
                        hipMemcpyHostToDevice));

    size_t attn_sinks_elements = n_layers * n_attn_heads;
    for (size_t i = 0; i < attn_sinks_elements; i++) {
        small_buffer[i] = __float2half(weights->attn_sinks[i]);
    }
    CHECK_HIP(hipMemcpy(t_d->weights.attn_sinks, small_buffer, attn_sinks_elements * sizeof(__half),
                        hipMemcpyHostToDevice));

    size_t b_router_elements = n_layers * n_experts;
    for (size_t i = 0; i < b_router_elements; i++) {
        small_buffer[i] = __float2half(weights->b_router[i]);
    }
    CHECK_HIP(hipMemcpy(t_d->weights.b_router, small_buffer, b_router_elements * sizeof(__half),
                        hipMemcpyHostToDevice));

    for (size_t i = 0; i < (size_t)hidden_dim; i++) {
        small_buffer[i] = __float2half(weights->rms_out_w[i]);
    }
    CHECK_HIP(hipMemcpy(t_d->weights.rms_out_w, small_buffer, hidden_dim * sizeof(__half),
                        hipMemcpyHostToDevice));

    free(small_buffer);

    // ! Initialize FP32 state buffers (zero initialization)
    CHECK_HIP(hipMemset(t_d->state.x, 0, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.t, 0, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.tb, 0, head_dim * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.tb2, 0, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.router_score, 0, n_experts * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.topk_v, 0, experts_per_token * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.topk_i, 0, experts_per_token * sizeof(int)));
    CHECK_HIP(hipMemset(t_d->state.mlp1_out, 0, 2 * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.gate, 0, intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.up, 0, intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.gate_up, 0, intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.e_agg, 0, hidden_dim * sizeof(float)));
    CHECK_HIP(
        hipMemset(t_d->state.qkv, 0, head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.q, 0, n_attn_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.k, 0, n_kv_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.v, 0, n_kv_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.att, 0, n_attn_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.logits, 0, vocab_size * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.key_cache, 0, key_cache_size));
    CHECK_HIP(hipMemset(t_d->state.value_cache, 0, value_cache_size));
    if (conf->sliding_window > 0) {
        float* mask_host = (float*)calloc(seq_len * seq_len, sizeof(float));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                if (i - j >= conf->sliding_window) {
                    mask_host[i * seq_len + j] = -INFINITY;
                }
            }
        }
        CHECK_HIP(hipMemcpy(t_d->state.mask, mask_host, seq_len * seq_len * sizeof(float),
                            hipMemcpyHostToDevice));
        free(mask_host);
    } else {
        CHECK_HIP(hipMemset(t_d->state.mask, 0, seq_len * seq_len * sizeof(float)));
    }

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    size_t used_mem = total_mem - free_mem;

    printf("Hybrid precision model loaded: %.1f GB allocated\n",
           used_mem / (1024.0 * 1024.0 * 1024.0));
}

void free_transformer_on_device_hybrid(OssTransformerHybrid* t_d) {
    printf("Freeing hybrid model GPU memory...\n");

    // Free FP16 weights
    CHECK_HIP(hipFree(t_d->weights.token_embedding_table));
    CHECK_HIP(hipFree(t_d->weights.rms_attn_w));
    CHECK_HIP(hipFree(t_d->weights.rms_ffn_w));
    CHECK_HIP(hipFree(t_d->weights.w_qkv));
    CHECK_HIP(hipFree(t_d->weights.w_o));
    CHECK_HIP(hipFree(t_d->weights.b_qkv));
    CHECK_HIP(hipFree(t_d->weights.b_o));
    CHECK_HIP(hipFree(t_d->weights.attn_sinks));
    CHECK_HIP(hipFree(t_d->weights.w_router));
    CHECK_HIP(hipFree(t_d->weights.b_router));
    CHECK_HIP(hipFree(t_d->weights.w_mlp1));
    CHECK_HIP(hipFree(t_d->weights.w_mlp2));
    CHECK_HIP(hipFree(t_d->weights.b_mlp1));
    CHECK_HIP(hipFree(t_d->weights.b_mlp2));
    CHECK_HIP(hipFree(t_d->weights.rms_out_w));
    CHECK_HIP(hipFree(t_d->weights.out));

    // Free FP32 state buffers
    CHECK_HIP(hipFree(t_d->state.x));
    CHECK_HIP(hipFree(t_d->state.t));
    CHECK_HIP(hipFree(t_d->state.tb));
    CHECK_HIP(hipFree(t_d->state.tb2));
    CHECK_HIP(hipFree(t_d->state.router_score));
    CHECK_HIP(hipFree(t_d->state.topk_v));
    CHECK_HIP(hipFree(t_d->state.topk_i));
    CHECK_HIP(hipFree(t_d->state.mlp1_out));
    CHECK_HIP(hipFree(t_d->state.gate));
    CHECK_HIP(hipFree(t_d->state.up));
    CHECK_HIP(hipFree(t_d->state.gate_up));
    CHECK_HIP(hipFree(t_d->state.e_agg));
    CHECK_HIP(hipFree(t_d->state.qkv));
    CHECK_HIP(hipFree(t_d->state.q));
    CHECK_HIP(hipFree(t_d->state.k));
    CHECK_HIP(hipFree(t_d->state.v));
    CHECK_HIP(hipFree(t_d->state.att));
    CHECK_HIP(hipFree(t_d->state.logits));
    CHECK_HIP(hipFree(t_d->state.key_cache));
    CHECK_HIP(hipFree(t_d->state.value_cache));
    CHECK_HIP(hipFree(t_d->state.mask));

    // Verify memory is freed
    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory freed: %.1f GB free / %.1f GB total\n",
           free_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0));
}

// ! 32-bit
void copy_transformer_to_device_full(OssTransformer* t_h, OssTransformer* t_d) {
    // ! Setup
    OssConfig* conf = &t_h->config;
    OssTransformerWeights* weights = &t_h->weights;
    OssRunState* state = &t_h->state;

    int vocab_size = conf->vocab_size;
    int hidden_dim = conf->hidden_dim;
    int n_experts = conf->n_experts;
    int experts_per_token = conf->experts_per_token;
    int intermediate_dim = conf->intermediate_dim;
    int n_layers = conf->n_layers;
    int head_dim = conf->head_dim;
    int n_attn_heads = conf->n_attn_heads;
    int n_kv_heads = conf->n_kv_heads;
    int seq_len = conf->seq_len;

    memcpy(&t_d->config, conf, sizeof(OssConfig));

    // ! GPU Memory Allocation for Weights
    CHECK_HIP(
        hipMalloc(&t_d->weights.token_embedding_table, vocab_size * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.rms_attn_w, n_layers * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.rms_ffn_w, n_layers * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.w_qkv,
                        n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) *
                            hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.w_o,
                        n_layers * hidden_dim * head_dim * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.b_qkv,
                        n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) *
                            sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.b_o, n_layers * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.attn_sinks, n_layers * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.w_router, n_layers * hidden_dim * n_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.b_router, n_layers * n_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.w_mlp1, 1ll * n_layers * n_experts * 2 * intermediate_dim *
                                                  hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.w_mlp2,
                        n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.b_mlp1,
                        n_layers * n_experts * 2 * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.b_mlp2, n_layers * n_experts * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.rms_out_w, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.out, vocab_size * hidden_dim * sizeof(float)));

    // ! Copy Weights to GPU
    CHECK_HIP(hipMemcpy(t_d->weights.token_embedding_table, weights->token_embedding_table,
                        vocab_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.rms_attn_w, weights->rms_attn_w,
                        n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.rms_ffn_w, weights->rms_ffn_w,
                        n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.w_qkv, weights->w_qkv,
                        n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) *
                            hidden_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.w_o, weights->w_o,
                        n_layers * hidden_dim * head_dim * n_attn_heads * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(
        hipMemcpy(t_d->weights.b_qkv, weights->b_qkv,
                  n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(float),
                  hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.b_o, weights->b_o, n_layers * hidden_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.attn_sinks, weights->attn_sinks,
                        n_layers * n_attn_heads * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.w_router, weights->w_router,
                        n_layers * hidden_dim * n_experts * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.b_router, weights->b_router,
                        n_layers * n_experts * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(
        hipMemcpy(t_d->weights.w_mlp1, weights->w_mlp1,
                  1ll * n_layers * n_experts * 2 * intermediate_dim * hidden_dim * sizeof(float),
                  hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.w_mlp2, weights->w_mlp2,
                        n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.b_mlp1, weights->b_mlp1,
                        n_layers * n_experts * 2 * intermediate_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.b_mlp2, weights->b_mlp2,
                        n_layers * n_experts * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.rms_out_w, weights->rms_out_w, hidden_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.out, weights->out, vocab_size * hidden_dim * sizeof(float),
                        hipMemcpyHostToDevice));

    // ! GPU Memory Allocation for State
    CHECK_HIP(hipMalloc(&t_d->state.x, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.t, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.tb, head_dim * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.tb2, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.router_score, n_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.topk_v, experts_per_token * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.topk_i, experts_per_token * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.mlp1_out, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.gate, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.up, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.gate_up, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.e_agg, hidden_dim * sizeof(float)));
    CHECK_HIP(
        hipMalloc(&t_d->state.qkv, head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.q, n_attn_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.k, n_kv_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.v, n_kv_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.att, n_attn_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.logits, vocab_size * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.key_cache, n_layers * seq_len * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.value_cache, n_layers * seq_len * head_dim * sizeof(float)));
    CHECK_HIP(
        hipMalloc(&t_d->state.mask, seq_len * sizeof(float))); // TODO: check if this is correct

    // ! Copy State to GPU
    CHECK_HIP(hipMemcpy(t_d->state.x, state->x, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.t, state->t, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.tb, state->tb, head_dim * n_attn_heads * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(
        hipMemcpy(t_d->state.tb2, state->tb2, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.router_score, state->router_score, n_experts * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.topk_v, state->topk_v, experts_per_token * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.topk_i, state->topk_i, experts_per_token * sizeof(int),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.mlp1_out, state->mlp1_out, hidden_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(
        hipMemcpy(t_d->state.gate, state->gate, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(
        hipMemcpy(t_d->state.up, state->up, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.gate_up, state->gate_up, hidden_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.e_agg, state->e_agg, hidden_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.qkv, state->qkv,
                        head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.q, state->q, n_attn_heads * head_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.k, state->k, n_kv_heads * head_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.v, state->v, n_kv_heads * head_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.att, state->att, n_attn_heads * seq_len * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.logits, state->logits, vocab_size * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.key_cache, state->key_cache,
                        n_layers * seq_len * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.value_cache, state->value_cache,
                        n_layers * seq_len * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(
        hipMemcpy(t_d->state.mask, state->mask, seq_len * sizeof(float), hipMemcpyHostToDevice));
}

void free_transformer_on_device_full(OssTransformer* t_d) {
    CHECK_HIP(hipFree(t_d->weights.token_embedding_table));
    CHECK_HIP(hipFree(t_d->weights.rms_attn_w));
    CHECK_HIP(hipFree(t_d->weights.rms_ffn_w));
    CHECK_HIP(hipFree(t_d->weights.w_qkv));
    CHECK_HIP(hipFree(t_d->weights.w_o));
    CHECK_HIP(hipFree(t_d->weights.b_qkv));
    CHECK_HIP(hipFree(t_d->weights.b_o));
    CHECK_HIP(hipFree(t_d->weights.attn_sinks));
    CHECK_HIP(hipFree(t_d->weights.w_router));
    CHECK_HIP(hipFree(t_d->weights.b_router));
    CHECK_HIP(hipFree(t_d->weights.w_mlp1));
    CHECK_HIP(hipFree(t_d->weights.w_mlp2));
    CHECK_HIP(hipFree(t_d->weights.b_mlp1));
    CHECK_HIP(hipFree(t_d->weights.b_mlp2));
    CHECK_HIP(hipFree(t_d->weights.rms_out_w));
    CHECK_HIP(hipFree(t_d->weights.out));

    CHECK_HIP(hipFree(t_d->state.x));
    CHECK_HIP(hipFree(t_d->state.t));
    CHECK_HIP(hipFree(t_d->state.tb));
    CHECK_HIP(hipFree(t_d->state.tb2));
    CHECK_HIP(hipFree(t_d->state.router_score));
    CHECK_HIP(hipFree(t_d->state.topk_v));
    CHECK_HIP(hipFree(t_d->state.topk_i));
    CHECK_HIP(hipFree(t_d->state.mlp1_out));
    CHECK_HIP(hipFree(t_d->state.gate));
    CHECK_HIP(hipFree(t_d->state.up));
    CHECK_HIP(hipFree(t_d->state.gate_up));
    CHECK_HIP(hipFree(t_d->state.e_agg));
    CHECK_HIP(hipFree(t_d->state.qkv));
    CHECK_HIP(hipFree(t_d->state.q));
    CHECK_HIP(hipFree(t_d->state.k));
    CHECK_HIP(hipFree(t_d->state.v));
    CHECK_HIP(hipFree(t_d->state.att));
    CHECK_HIP(hipFree(t_d->state.logits));
    CHECK_HIP(hipFree(t_d->state.key_cache));
    CHECK_HIP(hipFree(t_d->state.value_cache));
    CHECK_HIP(hipFree(t_d->state.mask));
}
