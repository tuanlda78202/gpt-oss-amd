#pragma once
#include "model.hpp"

// ! -----------------------------------Memory Management-----------------------------------------
void malloc_run_state(RunState* s, Config* p);

void free_run_state(RunState* s);

void memory_map_weights(TransformerWeights* w, Config* cfg, float* ptr);

void load_checkpoint(char* ckpt, Config* config, TransformerWeights* weights, int* fd, float** data,
                     ssize_t* file_size);

void build_transformer(Transformer* t, char* ckpt_path);

void free_transformer(Transformer* t);


// ! --------------------------------------------------------------------------------------
// utilities: time

long time_in_ms();