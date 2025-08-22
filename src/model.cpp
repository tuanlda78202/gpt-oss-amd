#include "../include/model.hpp"

void copy_transformer_to_device(OssTransformer* t_h, OssTransformer* t_d) {
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

    // Copy config to device
    t_d = (OssTransformer*)malloc(sizeof(OssTransformer));
    memcpy(&t_d->config, conf, sizeof(OssConfig));

    CHECK_HIP(hipMalloc(&t_d->weights.token_embedding_table, vocab_size * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.rms_attn_w, n_layers * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.rms_ffn_w, n_layers * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.w_qkv, n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.w_o, n_layers * hidden_dim * head_dim * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.b_qkv, n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.b_o, n_layers * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.attn_sinks, n_layers * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.w_router, n_layers * hidden_dim * n_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.b_router, n_layers * n_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.w_mlp1, n_layers * n_experts * 2 * intermediate_dim * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.w_mlp2, n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.b_mlp1, n_layers * n_experts * 2 * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.b_mlp2, n_layers * n_experts * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.rms_out_w, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->weights.out, vocab_size * hidden_dim * sizeof(float)));

    CHECK_HIP(hipMemcpy(t_d->weights.token_embedding_table, weights->token_embedding_table, vocab_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.rms_attn_w, weights->rms_attn_w, n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.rms_ffn_w, weights->rms_ffn_w, n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.w_qkv, weights->w_qkv, n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.w_o, weights->w_o, n_layers * hidden_dim * head_dim * n_attn_heads * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.b_qkv, weights->b_qkv, n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.b_o, weights->b_o, n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.attn_sinks, weights->attn_sinks, n_layers * n_attn_heads * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.w_router, weights->w_router, n_layers * hidden_dim * n_experts * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.b_router, weights->b_router, n_layers * n_experts * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.w_mlp1, weights->w_mlp1, n_layers * n_experts * 2 * intermediate_dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.w_mlp2, weights->w_mlp2, n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.b_mlp1, weights->b_mlp1, n_layers * n_experts * 2 * intermediate_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.b_mlp2, weights->b_mlp2, n_layers * n_experts * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.rms_out_w, weights->rms_out_w, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->weights.out, weights->out, vocab_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));

    // Copy state to device
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
    CHECK_HIP(hipMalloc(&t_d->state.qkv, head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.q, n_attn_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.k, n_kv_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.v, n_kv_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.att, n_attn_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.logits, vocab_size * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.key_cache, n_layers * seq_len * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.value_cache, n_layers * seq_len * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&t_d->state.mask, seq_len * sizeof(float)));  // TODO: check if this is correct

    CHECK_HIP(hipMemcpy(t_d->state.x, state->x, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.t, state->t, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.tb, state->tb, head_dim * n_attn_heads * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.tb2, state->tb2, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.router_score, state->router_score, n_experts * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.topk_v, state->topk_v, experts_per_token * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.topk_i, state->topk_i, experts_per_token * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.mlp1_out, state->mlp1_out, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.gate, state->gate, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.up, state->up, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.gate_up, state->gate_up, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.e_agg, state->e_agg, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.qkv, state->qkv, head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.q, state->q, n_attn_heads * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.k, state->k, n_kv_heads * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.v, state->v, n_kv_heads * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.att, state->att, n_attn_heads * seq_len * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.logits, state->logits, vocab_size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.key_cache, state->key_cache, n_layers * seq_len * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.value_cache, state->value_cache, n_layers * seq_len * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(t_d->state.mask, state->mask, seq_len * sizeof(float), hipMemcpyHostToDevice));
}

void copy_weights_to_device(OssTransformer* t_h, OssTransformerWeights* w_d) {
    OssConfig* conf = &t_h->config;
    OssTransformerWeights* weights = &t_h->weights;

    int vocab_size = conf->vocab_size;
    int hidden_dim = conf->hidden_dim;

    int n_experts = conf->n_experts;
    int n_layers = conf->n_layers;

    int head_dim = conf->head_dim;
    int n_attn_heads = conf->n_attn_heads;
    int n_kv_heads = conf->n_kv_heads;

    int intermediate_dim = conf->intermediate_dim;

    CHECK_HIP(hipMalloc(&w_d->token_embedding_table, vocab_size * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->rms_attn_w, n_layers * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->rms_ffn_w, n_layers * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->w_qkv, n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->w_o, n_layers * hidden_dim * head_dim * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->b_qkv, n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->b_o, n_layers * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->attn_sinks, n_layers * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->w_router, n_layers * hidden_dim * n_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->b_router, n_layers * n_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->w_mlp1, n_layers * n_experts * 2 * intermediate_dim * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->w_mlp2, n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->b_mlp1, n_layers * n_experts * 2 * intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->b_mlp2, n_layers * n_experts * hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->rms_out_w, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d->out, vocab_size * hidden_dim * sizeof(float)));

    CHECK_HIP(hipMemcpy(w_d->token_embedding_table, weights->token_embedding_table, vocab_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->rms_attn_w, weights->rms_attn_w, n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->rms_ffn_w, weights->rms_ffn_w, n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->w_qkv, weights->w_qkv, n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->w_o, weights->w_o, n_layers * hidden_dim * head_dim * n_attn_heads * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->b_qkv, weights->b_qkv, n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->b_o, weights->b_o, n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->attn_sinks, weights->attn_sinks, n_layers * n_attn_heads * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->w_router, weights->w_router, n_layers * hidden_dim * n_experts * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->b_router, weights->b_router, n_layers * n_experts * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->w_mlp1, weights->w_mlp1, n_layers * n_experts * 2 * intermediate_dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->w_mlp2, weights->w_mlp2, n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->b_mlp1, weights->b_mlp1, n_layers * n_experts * 2 * intermediate_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->b_mlp2, weights->b_mlp2, n_layers * n_experts * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->rms_out_w, weights->rms_out_w, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d->out, weights->out, vocab_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
}

void copy_state_to_device(OssTransformer* t_h, OssRunState* s_d) {
    OssConfig* conf = &t_h->config;
    OssRunState* state = &t_h->state;

    int hidden_dim = conf->hidden_dim;
    int n_experts = conf->n_experts;
    int experts_per_token = conf->experts_per_token;
    int n_layers = conf->n_layers;
    int head_dim = conf->head_dim;
    int n_attn_heads = conf->n_attn_heads;
    int n_kv_heads = conf->n_kv_heads;
    int seq_len = conf->seq_len;
    int vocab_size = conf->vocab_size;

    CHECK_HIP(hipMalloc(&s_d->x, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->t, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->tb, head_dim * n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->tb2, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->router_score, n_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->topk_v, experts_per_token * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->topk_i, experts_per_token * sizeof(int)));
    CHECK_HIP(hipMalloc(&s_d->mlp1_out, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->gate, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->up, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->gate_up, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->e_agg, hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->qkv, head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->q, n_attn_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->k, n_kv_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->v, n_kv_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->att, n_attn_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->logits, vocab_size * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->key_cache, n_layers * seq_len * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->value_cache, n_layers * seq_len * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_d->mask, seq_len * sizeof(float)));

    CHECK_HIP(hipMemcpy(s_d->x, state->x, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->t, state->t, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->tb, state->tb, head_dim * n_attn_heads * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->tb2, state->tb2, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->router_score, state->router_score, n_experts * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->topk_v, state->topk_v, experts_per_token * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->topk_i, state->topk_i, experts_per_token * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->mlp1_out, state->mlp1_out, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->gate, state->gate, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->up, state->up, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->gate_up, state->gate_up, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->e_agg, state->e_agg, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->qkv, state->qkv, head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->q, state->q, n_attn_heads * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->k, state->k, n_kv_heads * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->v, state->v, n_kv_heads * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->att, state->att, n_attn_heads * seq_len * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->logits, state->logits, vocab_size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->key_cache, state->key_cache, n_layers * seq_len * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->value_cache, state->value_cache, n_layers * seq_len * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_d->mask, state->mask, seq_len * sizeof(float), hipMemcpyHostToDevice));
}

void free_transformer_on_device(OssTransformer* t_d) {
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