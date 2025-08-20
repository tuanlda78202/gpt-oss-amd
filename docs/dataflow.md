# Data Flow

## Entry point

```
main() → argument parsing → build_transformer() → read_tokenizer() → build_sampler()
    ↓
Mode selection: generate|chat|getp
    ↓
getp() [if batch mode]
```

## Batch Processing

```
getp() → read_inputfile() → build_requests()
    ↓
warm_up() [⚠️ EMPTY - OPTIMIZE HERE]
    ↓
inference() → for each request: simple_getp_generate()
    ↓                              ↓
finish() [⚠️ EMPTY]               encode() → generation loop → forward() → sample()
    ↓                              ↓
write_outputfile()                 store output tokens
```

## Forward

```
forward() → token embedding lookup
    ↓
for each layer:
    ↓
    rmsnorm() → matmul(QKV) → apply_rotary_emb() → multi_head_attention()
    ↓                ↓                     ↓              ↓
    [norm]      [🔥 EXPENSIVE]      [position]     [🔥 VERY EXPENSIVE]
    ↓
    attention_output() → residual_connection()
    ↓
    rmsnorm() → matmul(router) → topk() → expert_processing()
    ↓              ↓              ↓           ↓
    [norm]    [🔥 EXPENSIVE]   [selection]  [🔥 VERY EXPENSIVE]
    ↓
    expert_aggregation() → residual_connection()
    ↓
[repeat for all layers]
    ↓
final_rmsnorm() → matmul(output) → logits
                       ↓
                 [🔥 MOST EXPENSIVE]
```

## Memory Usage

```
Model Weights: ~20GB-120GB (memory-mapped, read-only)
KV Cache: n_layers × seq_len × kv_dim × sizeof(float) × batch_size
    Example: 32 layers × 4096 seq × 128 kv_dim × 4 bytes × 8 batch = 512MB per request
Activations: ~hidden_dim × multiple buffers = ~10-100MB per request
```

## Computational Hotspots

1. Matrix Multiplications (80% of time):

- QKV projections: hidden_dim × (n_heads × head_dim)
- Attention outputs: (n_heads × head_dim) × hidden_dim
- MoE projections: hidden_dim × (2 × intermediate_dim) per expert
- Final output: hidden_dim × vocab_size

2. Attention Computation (15% of time):

- Q·K^T: seq_len × seq_len × n_heads × head_dim
- Attention×V: seq_len × n_heads × head_dim

3. Memory Bandwidth (5% of time):

- KV cache access
- Weight loading
- Activation transfers
