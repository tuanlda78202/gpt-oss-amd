# Data Flow

- [Data Flow](#data-flow)
  - [Entry point](#entry-point)
  - [Batch Processing](#batch-processing)
  - [Forward](#forward)
  - [Memory Usage](#memory-usage)
  - [Computational Hotspots](#computational-hotspots)
  - [Visualization](#visualization)

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

- QKV projections: `hidden_dim × (n_heads × head_dim)`
- Attention outputs: `(n_heads × head_dim) × hidden_dim`
- MoE projections: `hidden_dim × (2 × intermediate_dim)` per expert
- Final output: `hidden_dim × vocab_size`

2. Attention Computation (15% of time):

- Q·K^T: `seq_len × seq_len × n_heads × head_dim`
- Attention×V: `seq_len × n_heads × head_dim`

3. Memory Bandwidth (5% of time):

- KV cache access
- Weight loading
- Activation transfers

## Visualization

1. General flow

    ```mermaid
    graph TD
        A["`**main() in run.cpp**
        Command Line Parsing
        -m getp -i input.txt -o output.txt`"] --> B["`**Build Core Components**
        1. build_transformer() - Load model weights
        2. read_tokenizer() - Load tokenizer
        3. build_sampler() - Init sampling`"]
        
        B --> C{"`**Mode Selection**`"}
        C -->|generate| D["`**generate()**
        Single prompt → completion`"]
        C -->|chat| E["`**chat()**
        Interactive conversation`"]
        C -->|getp| F["`**getp() in getp_eval.cpp**
        Batch processing for throughput`"]
        
        F --> G["`**read_inputfile()**
        Load batch of input prompts
        → Requests struct`"]
        
        G --> H["`**Requests Structure**
        • num_reqs: Number of prompts
        • str_reqs: Raw text prompts
        • tok_gens: Output token buffers
        • max_seq_len: Sequence limit`"]
        
        H --> I["`**🔥 WARM-UP PHASE 🔥**
        warm_up(transformer, tokenizer)
        ⚠️ CURRENTLY EMPTY - OPTIMIZE HERE!
        • Pre-allocate GPU memory
        • Load model to GPU
        • Initialize kernels`"]
        
        I --> J["`**⚡ INFERENCE PHASE ⚡**
        inference(transformer, tokenizer, sampler, requests)
        📊 TIMED FOR THROUGHPUT MEASUREMENT`"]
        
        J --> K["`**Current Sequential Processing**
        for each request:
          simple_getp_generate()`"]
        
        K --> L["`**simple_getp_generate() Flow**
        1. encode(prompt) → token_ids
        2. for each position:
           • forward(transformer, token, pos)
           • sample(logits) → next_token
        3. Store output tokens`"]
        
        L --> M["`**forward() - Core Transformer**
        🧠 HEAVY COMPUTATION HERE
        • Token embedding lookup
        • For each layer:
          - RMSNorm → Attention
          - MoE routing → Expert selection
          - Residual connections
        • Output logits`"]
        
        M --> N["`**🔄 Back to Sequential Loop**
        Process next request...
        ❌ NO BATCHING
        ❌ NO GPU USAGE
        ❌ NO PARALLELISM`"]
        
        N --> O["`**write_outputfile()**
        Save generated token sequences
        to output file`"]
        
        O --> P["`**🧹 FINISH PHASE 🧹**
        finish(transformer, tokenizer)
        ⚠️ CURRENTLY EMPTY - CLEANUP HERE!
        • Free GPU memory
        • Unload kernels`"]
        
        P --> Q["`**Performance Report**
        📈 Throughput: tokens/sec
        ⏱️ Elapsed time`"]
    
        style I fill:#ff9999,stroke:#ff0000,stroke-width:3px
        style J fill:#ffeb99,stroke:#ff6600,stroke-width:3px
        style K fill:#ff9999,stroke:#ff0000,stroke-width:2px
        style L fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style M fill:#cc99ff,stroke:#6600cc,stroke-width:3px
        style N fill:#ff9999,stroke:#ff0000,stroke-width:2px
        style P fill:#ff9999,stroke:#ff0000,stroke-width:3px
    ```

2. Pipeline flow

    ```mermaid
    graph LR
        subgraph "`**INPUT DATA FLOW**`"
            A["`**input.txt**
            N
            prompt1
            prompt2
            ...
            promptN`"] --> B["`**Requests Structure**
            📦 Container for batch data`"]
            
            B --> C["`**str_reqs**
            Raw text prompts
            [prompt1][prompt2]...[promptN]`"]
            
            B --> D["`**tok_gens**
            Output token buffers
            [tokens1][tokens2]...[tokensN]`"]
        end
        
        subgraph "`**CORE MODEL COMPONENTS**`"
            E["`**Transformer**
            🧠 Model weights & config
            • Config: vocab_size, n_layers, etc.
            • Weights: embeddings, attention, MLP
            • State: activations, KV cache`"]
            
            F["`**Tokenizer**
            🔤 Text ↔ Token conversion
            • encode(): text → token_ids
            • decode(): token_ids → text`"]
            
            G["`**Sampler**
            🎲 Token selection
            • Temperature, top-p
            • Random number generation`"]
        end
        
        subgraph "`**PROCESSING PIPELINE**`"
            H["`**FOR EACH REQUEST:**
            1. Get str_req[i]`"] --> I["`**encode()**
            text → token_ids
            [101, 2023, 345, ...]`"]
            
            I --> J["`**Generation Loop**
            pos = 0 to max_seq_len`"]
            
            J --> K["`**forward()**
            🔥 MOST EXPENSIVE
            token, pos → logits
            [0.1, 0.05, 0.8, ...]`"]
            
            K --> L["`**sample()**
            logits → next_token
            select based on temperature/top-p`"]
            
            L --> M["`**Store in tok_gens[i]**
            Accumulate output tokens`"]
            
            M --> N{"`More positions?`"}
            N -->|Yes| J
            N -->|No| O["`**Next Request**`"]
            O --> H
        end
        
        subgraph "`**OPTIMIZATION TARGETS**`"
            P["`**❌ CURRENT BOTTLENECKS**
            • Sequential request processing
            • CPU-only computation
            • No memory reuse
            • Single-threaded forward()`"]
            
            Q["`**✅ OPTIMIZATION OPPORTUNITIES**
            • Batch multiple requests
            • GPU acceleration (HIP kernels)
            • Multi-GPU parallelism
            • Efficient KV cache management
            • OpenMP threading`"]
        end
        
        subgraph "`**OUTPUT**`"
            R["`**output.txt**
            token_id1 token_id2 ... \\n
            token_id1 token_id2 ... \\n
            ...`"] --> S["`**decode (separate tool)**
            Convert token IDs back to text
            for human reading`"]
        end
        
        C --> H
        D --> M
        E --> K
        F --> I
        G --> L
        M --> R
        
        style P fill:#ff9999,stroke:#ff0000,stroke-width:2px
        style Q fill:#99ff99,stroke:#00cc00,stroke-width:2px
        style K fill:#ffcc99,stroke:#ff6600,stroke-width:3px
    ```

3. Model architecture

    ```mermaid
    graph TD
        A["`**Input Token**
        token_id (e.g., 1337)`"] --> B["`**Token Embedding**
        embedding_table[token_id]
        → hidden_dim vector`"]
        
        B --> C["`**Layer Loop**
        for l = 0 to n_layers-1`"]
        
        C --> D["`**RMSNorm (Attention)**
        Normalize activations`"]
        
        D --> E["`**QKV Projection**
        🔥 EXPENSIVE: matmul + bias
        hidden_dim → (q,k,v) heads`"]
        
        E --> F["`**RoPE (Rotary Embedding)**
        Apply position encoding
        with YaRN scaling`"]
        
        F --> G["`**Multi-Head Attention**
        🔥 VERY EXPENSIVE
        • Q·K^T attention scores
        • Softmax normalization  
        • Weighted V aggregation
        • Grouped Query Attention (GQA)`"]
        
        G --> H["`**Attention Output**
        matmul + bias + residual`"]
        
        H --> I["`**RMSNorm (MLP)**
        Normalize for feed-forward`"]
        
        I --> J["`**MoE Router**
        🔥 EXPENSIVE: matmul + bias
        Select top-k experts`"]
        
        J --> K["`**Expert Processing**
        For each selected expert:
        🔥 VERY EXPENSIVE
        • Gate/Up projection (2×intermediate_dim)
        • SwiGLU activation
        • Down projection`"]
        
        K --> L["`**Expert Aggregation**
        Weighted sum of expert outputs
        + residual connection`"]
        
        L --> M{"`More layers?`"}
        M -->|Yes| C
        M -->|No| N["`**Final RMSNorm**
        Normalize final activations`"]
        
        N --> O["`**Output Projection**
        🔥 EXPENSIVE: matmul
        hidden_dim → vocab_size logits`"]
        
        O --> P["`**Sampling**
        • Apply temperature
        • Softmax → probabilities
        • Top-p or argmax selection`"]
        
        subgraph "`**Memory Systems**`"
            Q["`**KV Cache**
        key_cache[layer][pos][kv_dim]
        value_cache[layer][pos][kv_dim]
        🔥 HUGE MEMORY USAGE`"]
            
            R["`**Activations**
            • x: current activations
            • qkv: attention projections  
            • expert buffers
            🔥 TEMPORARY MEMORY`"]
        end
        
        subgraph "`**Computational Hotspots**`"
            S["`**🔥 Matrix Multiplications**
            • QKV projections
            • Attention output
            • MoE gate/up/down
            • Final output
            ⚡ GPU ACCELERATION TARGET`"]
            
            T["`**🔥 Attention Computation**
            • Q·K^T (sequence × heads)
            • Attention × V
            ⚡ MEMORY BANDWIDTH BOUND`"]
            
            U["`**🔥 Expert Selection**
            • Router computation
            • Top-k sorting
            • Conditional execution
            ⚡ BATCHING OPPORTUNITY`"]
        end
        
        F --> Q
        G --> Q
        E --> R
        K --> R
        
        style S fill:#ff6666,stroke:#cc0000,stroke-width:3px
        style T fill:#ff6666,stroke:#cc0000,stroke-width:3px  
        style U fill:#ff6666,stroke:#cc0000,stroke-width:3px
        style Q fill:#ffcc99,stroke:#ff6600,stroke-width:2px
        style R fill:#ffcc99,stroke:#ff6600,stroke-width:2px
    ```
