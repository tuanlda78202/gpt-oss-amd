#include "../include/model.hpp"

void copy_large_tensor_streaming(__half** d_ptr, float* h_ptr, size_t total_size,
                                 const char* tensor_name) {
    const size_t chunk_size = 512 * 1024 * 1024;

    bool show_progress = total_size > 1024 * 1024;

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

    printf("\nConverting model to hybrid precision...\n");

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

    // ! Allocate FP32 state on GPU (with batch dimension B)
    int batch_size = conf->batch_size;
    printf("🚒 Allocating GPU buffers for batch size %d...\n", batch_size);

    CHECK_HIP(hipMalloc(&t_d->state.x, (size_t)batch_size * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.t, (size_t)batch_size * hidden_dim * sizeof(float)));
    CHECK_HIP(
        hipMalloc(&t_d->state.tb, (size_t)batch_size * head_dim * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.tb2, (size_t)batch_size * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.router_score, (size_t)batch_size * n_experts * sizeof(float)));
    CHECK_HIP(
        hipMalloc(&t_d->state.topk_v, (size_t)batch_size * experts_per_token * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.topk_i, (size_t)batch_size * experts_per_token * sizeof(int)));
    CHECK_HIP(
        hipMalloc(&t_d->state.mlp1_out, (size_t)batch_size * 2 * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.gate, (size_t)batch_size * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.up, (size_t)batch_size * intermediate_dim * sizeof(float)));
    CHECK_HIP(
        hipMalloc(&t_d->state.gate_up, (size_t)batch_size * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.e_agg, (size_t)batch_size * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.qkv, (size_t)batch_size * head_dim *
                                             (n_attn_heads + 2 * n_kv_heads) * sizeof(float)));
    CHECK_HIP(
        hipMalloc(&t_d->state.q, (size_t)batch_size * n_attn_heads * head_dim * sizeof(float)));
    CHECK_HIP(
        hipMalloc(&t_d->state.att, (size_t)batch_size * n_attn_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.logits, (size_t)batch_size * vocab_size * sizeof(float)));

    // KV cache with batch dimension: (n_layers, batch_size, seq_len, kv_dim)
    size_t key_cache_size =
        1ll * n_layers * batch_size * seq_len * n_kv_heads * head_dim * sizeof(float);
    size_t value_cache_size =
        1ll * n_layers * batch_size * seq_len * n_kv_heads * head_dim * sizeof(float);
    CHECK_HIP(hipMalloc(&t_d->state.key_cache, key_cache_size));
    CHECK_HIP(hipMalloc(&t_d->state.value_cache, value_cache_size));
    CHECK_HIP(hipMalloc(&t_d->state.mask, (size_t)seq_len * seq_len * sizeof(float)));

    CHECK_HIP(hipMalloc(&t_d->state.d_batch_indices, (size_t)batch_size * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.d_tokens, (size_t)batch_size * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.d_pos_per_token, (size_t)batch_size * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.cos_vals, (size_t)(head_dim / 2) * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.sin_vals, (size_t)(head_dim / 2) * sizeof(float)));

    // MoE expert-batching scratch buffers
    int BK_max = batch_size * experts_per_token;
    CHECK_HIP(hipMalloc(&t_d->state.assign_expert, (size_t)BK_max * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.assign_token, (size_t)BK_max * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.assign_weight, (size_t)BK_max * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.expert_counts, (size_t)n_experts * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.expert_offsets, (size_t)(n_experts + 1) * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.tokens_flat, (size_t)BK_max * sizeof(int)));
    CHECK_HIP(hipMalloc(&t_d->state.weights_flat, (size_t)BK_max * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.x_by_expert, (size_t)BK_max * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.mlp1_by_expert,
                        (size_t)BK_max * 2 * intermediate_dim * sizeof(float)));
    CHECK_HIP(
        hipMalloc(&t_d->state.gate_by_expert, (size_t)BK_max * intermediate_dim * sizeof(float)));
    CHECK_HIP(
        hipMalloc(&t_d->state.up_by_expert, (size_t)BK_max * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.y_by_expert, (size_t)BK_max * hidden_dim * sizeof(float)));

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

    // ! Initialize FP32 state buffers (zero initialization) with batch dimension
    CHECK_HIP(hipMemset(t_d->state.x, 0, batch_size * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.t, 0, batch_size * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.tb, 0, batch_size * head_dim * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.tb2, 0, batch_size * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.router_score, 0, batch_size * n_experts * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.topk_v, 0, batch_size * experts_per_token * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.topk_i, 0, batch_size * experts_per_token * sizeof(int)));
    CHECK_HIP(hipMemset(t_d->state.mlp1_out, 0, batch_size * 2 * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.gate, 0, batch_size * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.up, 0, batch_size * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.gate_up, 0, batch_size * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.e_agg, 0, batch_size * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.qkv, 0,
                        batch_size * head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.q, 0, batch_size * n_attn_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.att, 0, batch_size * n_attn_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMemset(t_d->state.logits, 0, batch_size * vocab_size * sizeof(float)));
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
    printf("✅ Hybrid precision model loaded: %.1f GB allocated\n",
           used_mem / (1024.0 * 1024.0 * 1024.0));
}

void free_transformer_on_device_hybrid(OssTransformerHybrid* t_d) {
    printf("\033[1;92m==================================================================\033["
           "0m\n\033[1;92m♻️  FREE GPU MEMORY...\033[0m\n");

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
    CHECK_HIP(hipFree(t_d->state.att));
    CHECK_HIP(hipFree(t_d->state.logits));
    CHECK_HIP(hipFree(t_d->state.key_cache));
    CHECK_HIP(hipFree(t_d->state.value_cache));
    CHECK_HIP(hipFree(t_d->state.mask));
    CHECK_HIP(hipFree(t_d->state.d_batch_indices));
    CHECK_HIP(hipFree(t_d->state.d_tokens));
    CHECK_HIP(hipFree(t_d->state.d_pos_per_token));
    CHECK_HIP(hipFree(t_d->state.cos_vals));
    CHECK_HIP(hipFree(t_d->state.sin_vals));

    CHECK_HIP(hipFree(t_d->state.assign_expert));
    CHECK_HIP(hipFree(t_d->state.assign_token));
    CHECK_HIP(hipFree(t_d->state.assign_weight));
    CHECK_HIP(hipFree(t_d->state.expert_counts));
    CHECK_HIP(hipFree(t_d->state.expert_offsets));
    CHECK_HIP(hipFree(t_d->state.tokens_flat));
    CHECK_HIP(hipFree(t_d->state.weights_flat));
    CHECK_HIP(hipFree(t_d->state.x_by_expert));
    CHECK_HIP(hipFree(t_d->state.mlp1_by_expert));
    CHECK_HIP(hipFree(t_d->state.gate_by_expert));
    CHECK_HIP(hipFree(t_d->state.up_by_expert));
    CHECK_HIP(hipFree(t_d->state.y_by_expert));

    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory: %.1f GB free / %.1f GB total\n", free_mem / (1024.0 * 1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0 * 1024.0));
}
