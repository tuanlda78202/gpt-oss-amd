#include "../include/model.hpp"

// ! 16-bit
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

void copy_transformer_to_device_half(OssTransformer* t_fp32, OssTransformerHalf* t_d) {
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

    printf("\nConverting model to half precision (streaming)...\n");

    // ! GPU Check
    int current_device;
    CHECK_HIP(hipGetDevice(&current_device));

    hipDeviceProp_t deviceProp;
    CHECK_HIP(hipGetDeviceProperties(&deviceProp, current_device));

    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    printf("GPU %d (%s): %.1f GB free / %.1f GB total\n", current_device, deviceProp.name,
           free_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0));

    // ! Calculate sizes
    size_t token_emb_size = (size_t)vocab_size * hidden_dim * sizeof(__half);
    size_t mlp1_size =
        1ll * n_layers * n_experts * 2 * intermediate_dim * hidden_dim * sizeof(__half);
    size_t mlp2_size = 1ll * n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(__half);
    size_t key_cache_size = 1ll * n_layers * seq_len * n_kv_heads * head_dim * sizeof(__half);
    size_t value_cache_size = 1ll * n_layers * seq_len * n_kv_heads * head_dim * sizeof(__half);

    size_t total_expected_allocation =
        token_emb_size + (size_t)n_layers * hidden_dim * sizeof(__half) * 2 + // rms weights
        (size_t)n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * hidden_dim *
            sizeof(__half) +                                                       // w_qkv
        (size_t)n_layers * hidden_dim * head_dim * n_attn_heads * sizeof(__half) + // w_o
        (size_t)n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) *
            sizeof(__half) +                                         // b_qkv
        (size_t)n_layers * hidden_dim * sizeof(__half) +             // b_o
        (size_t)n_layers * n_attn_heads * sizeof(__half) +           // attn_sinks
        (size_t)n_layers * hidden_dim * n_experts * sizeof(__half) + // w_router
        (size_t)n_layers * n_experts * sizeof(__half) +              // b_router
        mlp1_size +
        mlp2_size + (size_t)n_layers * n_experts * 2 * intermediate_dim * sizeof(__half) + // b_mlp1
        (size_t)n_layers * n_experts * hidden_dim * sizeof(__half) +                       // b_mlp2
        (size_t)hidden_dim * sizeof(__half) +              // rms_out_w
        (size_t)vocab_size * hidden_dim * sizeof(__half) + // out
        // State buffers
        (size_t)hidden_dim * sizeof(__half) * 10 + // various state buffers
        (size_t)head_dim * n_attn_heads * sizeof(__half) + (size_t)n_experts * sizeof(__half) +
        (size_t)experts_per_token * sizeof(__half) + (size_t)experts_per_token * sizeof(int) +
        (size_t)head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(__half) +
        (size_t)n_attn_heads * head_dim * sizeof(__half) +
        (size_t)n_kv_heads * head_dim * sizeof(__half) * 2 +
        (size_t)n_attn_heads * seq_len * sizeof(__half) + (size_t)vocab_size * sizeof(__half) +
        key_cache_size + value_cache_size + (size_t)seq_len * sizeof(__half);

    printf("Allocating %.1f GB GPU memory...\n",
           total_expected_allocation / (1024.0 * 1024.0 * 1024.0));

    // ! GPU Memory Allocation for Weights
    size_t allocated_so_far = 0;

    CHECK_HIP(hipMalloc(&t_d->weights.token_embedding_table, token_emb_size));
    allocated_so_far += token_emb_size;

    CHECK_HIP(hipMalloc(&t_d->weights.rms_attn_w, (size_t)n_layers * hidden_dim * sizeof(__half)));
    allocated_so_far += (size_t)n_layers * hidden_dim * sizeof(__half);

    CHECK_HIP(hipMalloc(&t_d->weights.rms_ffn_w, (size_t)n_layers * hidden_dim * sizeof(__half)));
    allocated_so_far += (size_t)n_layers * hidden_dim * sizeof(__half);

    size_t qkv_size = (size_t)n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) *
                      hidden_dim * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.w_qkv, qkv_size));
    allocated_so_far += qkv_size;

    size_t w_o_size = (size_t)n_layers * hidden_dim * head_dim * n_attn_heads * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.w_o, w_o_size));
    allocated_so_far += w_o_size;

    size_t b_qkv_size =
        (size_t)n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.b_qkv, b_qkv_size));
    allocated_so_far += b_qkv_size;

    CHECK_HIP(hipMalloc(&t_d->weights.b_o, (size_t)n_layers * hidden_dim * sizeof(__half)));
    allocated_so_far += (size_t)n_layers * hidden_dim * sizeof(__half);

    CHECK_HIP(
        hipMalloc(&t_d->weights.attn_sinks, (size_t)n_layers * n_attn_heads * sizeof(__half)));
    allocated_so_far += (size_t)n_layers * n_attn_heads * sizeof(__half);

    size_t router_size = (size_t)n_layers * hidden_dim * n_experts * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.w_router, router_size));
    allocated_so_far += router_size;

    CHECK_HIP(hipMalloc(&t_d->weights.b_router, (size_t)n_layers * n_experts * sizeof(__half)));
    allocated_so_far += (size_t)n_layers * n_experts * sizeof(__half);

    // Allocate large MLP weights
    CHECK_HIP(hipMalloc(&t_d->weights.w_mlp1, mlp1_size));
    allocated_so_far += mlp1_size;

    CHECK_HIP(hipMalloc(&t_d->weights.w_mlp2, mlp2_size));
    allocated_so_far += mlp2_size;

    size_t b_mlp1_size = (size_t)n_layers * n_experts * 2 * intermediate_dim * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.b_mlp1, b_mlp1_size));
    allocated_so_far += b_mlp1_size;

    size_t b_mlp2_size = (size_t)n_layers * n_experts * hidden_dim * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.b_mlp2, b_mlp2_size));
    allocated_so_far += b_mlp2_size;

    CHECK_HIP(hipMalloc(&t_d->weights.rms_out_w, (size_t)hidden_dim * sizeof(__half)));
    allocated_so_far += (size_t)hidden_dim * sizeof(__half);

    size_t out_size = (size_t)vocab_size * hidden_dim * sizeof(__half);
    CHECK_HIP(hipMalloc(&t_d->weights.out, out_size));
    allocated_so_far += out_size;

    // ! GPU Memory Allocation for State
    CHECK_HIP(hipMalloc(&t_d->state.x, (size_t)hidden_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.t, (size_t)hidden_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.tb, (size_t)head_dim * n_attn_heads * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.tb2, (size_t)hidden_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.router_score, (size_t)n_experts * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.topk_v, (size_t)experts_per_token * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.topk_i, (size_t)experts_per_token * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.mlp1_out, (size_t)hidden_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.gate, (size_t)hidden_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.up, (size_t)hidden_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.gate_up, (size_t)hidden_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.e_agg, (size_t)hidden_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.qkv,
                        (size_t)head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.q, (size_t)n_attn_heads * head_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.k, (size_t)n_kv_heads * head_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.v, (size_t)n_kv_heads * head_dim * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.att, (size_t)n_attn_heads * seq_len * sizeof(__half)));
    CHECK_HIP(hipMalloc(&t_d->state.logits, (size_t)vocab_size * sizeof(__half)));

    // Allocate KV cache
    CHECK_HIP(hipMalloc(&t_d->state.key_cache, key_cache_size));
    allocated_so_far += key_cache_size;
    CHECK_HIP(hipMalloc(&t_d->state.value_cache, value_cache_size));
    allocated_so_far += value_cache_size;
    CHECK_HIP(hipMalloc(&t_d->state.mask,
                        (size_t)seq_len * sizeof(__half))); // TODO: check if it's correct

    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    size_t used_mem = total_mem - free_mem;
    printf("Allocated %.1f GB, %.1f GB free remaining\n", used_mem / (1024.0 * 1024.0 * 1024.0),
           free_mem / (1024.0 * 1024.0 * 1024.0));

    CHECK_HIP(hipDeviceSynchronize());

    // ! Stream conversion and transfer for weights
    printf("Transferring weights...\n");

    // Allocate small conversion buffer for small weights
    __half* small_buffer = (__half*)malloc(1024 * 1024 * sizeof(__half)); // 1M elements buffer
    if (!small_buffer) {
        fprintf(stderr, "Failed to allocate small conversion buffer\n");
        exit(EXIT_FAILURE);
    }

    // Stream large tensors
    copy_large_tensor_streaming(&t_d->weights.token_embedding_table, weights->token_embedding_table,
                                token_emb_size, "token_embedding_table");

    // Small weights - direct conversion
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

    copy_large_tensor_streaming(&t_d->weights.w_qkv, weights->w_qkv, qkv_size, "w_qkv");
    copy_large_tensor_streaming(&t_d->weights.w_o, weights->w_o, w_o_size, "w_o");

    // Convert bias vectors
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

    copy_large_tensor_streaming(&t_d->weights.w_router, weights->w_router, router_size, "w_router");

    size_t b_router_elements = n_layers * n_experts;
    for (size_t i = 0; i < b_router_elements; i++) {
        small_buffer[i] = __float2half(weights->b_router[i]);
    }
    CHECK_HIP(hipMemcpy(t_d->weights.b_router, small_buffer, b_router_elements * sizeof(__half),
                        hipMemcpyHostToDevice));

    // Stream large MLP weights
    copy_large_tensor_streaming(&t_d->weights.w_mlp1, weights->w_mlp1, mlp1_size, "w_mlp1");
    copy_large_tensor_streaming(&t_d->weights.w_mlp2, weights->w_mlp2, mlp2_size, "w_mlp2");

    copy_large_tensor_streaming(&t_d->weights.b_mlp1, weights->b_mlp1, b_mlp1_size, "b_mlp1");
    copy_large_tensor_streaming(&t_d->weights.b_mlp2, weights->b_mlp2, b_mlp2_size, "b_mlp2");

    // Final weights
    for (size_t i = 0; i < (size_t)hidden_dim; i++) {
        small_buffer[i] = __float2half(weights->rms_out_w[i]);
    }
    CHECK_HIP(hipMemcpy(t_d->weights.rms_out_w, small_buffer, hidden_dim * sizeof(__half),
                        hipMemcpyHostToDevice));

    copy_large_tensor_streaming(&t_d->weights.out, weights->out, out_size, "out");

    // ! Convert and copy state buffers
    printf("Transferring state buffers...\n");

    // Small state buffers
    for (size_t i = 0; i < (size_t)hidden_dim; i++) {
        small_buffer[i] = __float2half(state->x[i]);
    }
    CHECK_HIP(
        hipMemcpy(t_d->state.x, small_buffer, hidden_dim * sizeof(__half), hipMemcpyHostToDevice));

    for (size_t i = 0; i < (size_t)hidden_dim; i++) {
        small_buffer[i] = __float2half(state->t[i]);
    }
    CHECK_HIP(
        hipMemcpy(t_d->state.t, small_buffer, hidden_dim * sizeof(__half), hipMemcpyHostToDevice));

    size_t tb_elements = head_dim * n_attn_heads;
    for (size_t i = 0; i < tb_elements; i++) {
        small_buffer[i] = __float2half(state->tb[i]);
    }
    CHECK_HIP(hipMemcpy(t_d->state.tb, small_buffer, tb_elements * sizeof(__half),
                        hipMemcpyHostToDevice));

    // Large state buffers
    copy_large_tensor_streaming(&t_d->state.key_cache, state->key_cache, key_cache_size,
                                "key_cache");
    copy_large_tensor_streaming(&t_d->state.value_cache, state->value_cache, value_cache_size,
                                "value_cache");

    free(small_buffer);

    // Final verification
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    used_mem = total_mem - free_mem;

    // Test memory access
    __half test_value = __float2half(1.0f);
    CHECK_HIP(hipMemcpy(t_d->weights.token_embedding_table, &test_value, sizeof(__half),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipDeviceSynchronize());

    printf("Half precision model loaded: %.1f GB allocated\n",
           used_mem / (1024.0 * 1024.0 * 1024.0));
}

void free_transformer_on_device_half(OssTransformerHalf* t_d) {
    printf("Freeing GPU memory...\n");

    // Free weights
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

    // Free state buffers
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
