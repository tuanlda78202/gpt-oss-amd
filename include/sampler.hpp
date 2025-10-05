#pragma once

typedef struct {
    float prob;
    int index;
} OssProbIndex;

typedef struct {
    int vocab_size;
    OssProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} OssSampler;

int sample_argmax_oss(float* probabilities, int n);

int sample_mult_oss(float* probabilities, int n, float coin);

int compare_oss(const void* a, const void* b);

int sample_topp_oss(float* probabilities, int n, float topp, OssProbIndex* probindex, float coin);

void build_sampler_oss(OssSampler* sampler, int vocab_size, float temperature, float topp,
                       unsigned long long rng_seed);

void free_sampler_oss(OssSampler* sampler);

int sample_oss(OssSampler* sampler, float* logits);

int sample_oss_gpu(OssSampler* sampler, float* logits_d);
