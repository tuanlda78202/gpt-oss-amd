<div align="center">

# GPT-OSS from Scratch on AMD GPUs

 <p>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/github/actions/workflow/status/tuanlda78202/gpt-oss-amd/ci.yaml?branch=main&label=CI&logo=github" alt="CI Status">
    <img src="https://img.shields.io/github/last-commit/tuanlda78202/gpt-oss-amd?&label=commit" alt="Last Commit">
 </p>

[Abstract](#abstract) | [Build & Run](#build-and-run) | [Experiments](#experiments) | [Acknowledgements](#acknowledgments) | [Contact](#contact)

<img width="1709" height="963" alt="image" src="https://github.com/user-attachments/assets/ef0a6e0a-acd0-4a8c-8916-4b28d0207822" />

</div>

## Abstract

After six years-the first time since GPT-2, OpenAI has released new open-weight LLMs, `gpt-oss-20b` and `gpt-oss-120b`. From day one, many inference engines such as llama.cpp, vLLM, and SGLang have supported these models; however, most focus on maximizing throughput using CUDA for NVIDIA GPUs, offering limited support for AMD GPUs. Moreover, their library-oriented implementations are often complex and difficult to adapt for personal/experimental use cases.

To address these limitations, we present `gpt-oss-amd`, a pure C++ implementation of OpenAI’s GPT-OSS models designed to **maximize inference throughput on AMD GPUs without relying on external libraries**. Our goal is to explore end-to-end LLM optimization, from kernel-level improvements to system-level design, providing insights for researchers and developers interested in high-performance computing and model-level optimization.

Inspired by [llama2.c](https://github.com/karpathy/llama2.c), our implementation uses HIP (an AMD equivalent to CUDA) and avoids dependencies such as rocBLAS, hipBLAS, RCCL, and MPI. We employ a range of optimization techniques for both the 20B and 120B models, including efficient model loading, batching, multi-streaming, multi-GPUs communication, optimized CPU–GPU–SRAM memory access, FlashAttention, matrix-core–based GEMM, and load balancing in MoE routing. Experiments on a single node with 8× AMD MI250 GPUs show that our implementation achieves over 30k TPS on the 20B model and nearly 10k TPS on the 120B model in custom benchmarks, demonstrating the effectiveness of our optimizations and the strong potential of AMD GPUs for large-scale LLM inference.

---

## Roadmap

* [x] Release codebase
* [ ] Publish worklog blog post

## Build and Run

### Code Structure

```plain
gpt-oss-amd/
   ├── include/              # Header files
   ├── src/
   │   ├── getp/             # Request serving and runtime logic
   │   ├── hip/              # Custom HIP kernels for AMD GPUs
   │   ├── forward.cpp       # Model forward pass implementation
   ├── tests/                # Evaluation scripts
   ├── tools/                # Model/tokenizer conversion and HF inference utilities
   └── run.sh                # Build and run script
```

### Resources

* Download GPT-OSS 20/120B model `safetensors` files from [here](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) and convert them to `bin` using the provided script in `tools/model_export` to can use with the C++ inference runtime.

* Tokenizer compatible with OpenAI `o200k_harmony`  (via `tiktoken`).

* GCC/Clang with OpenMP and HIP/ROCm installed.

### Setup Env

```bash
uv sync
source .venv/bin/activate
pre-commit install
chmod +x run.sh
```

### Build

```bash
./run build [default|fast|omp]
```

### Run

* Chat

  ```bash
  # interactive turn-based generation, optional system prompt
  ./run run -m chat -i "How do I tune top-p?" -y "You are a concise assistant." -T 0.7
  ```

* Single-Prompt

  ```bash
  # single prompt → completion
  ./run run -m generate -i "Write a haiku about parallelism." -T 0.8 -p 0.95
  ```

* Batch

  ```bash
  # multi-prompt batch
  ./run run                          # default 20B, 1 GPU, uses tests/data/{input,output}.txt
  ./run run -m 120 -g 8 --kv16       # 120B, 8 GPUs, KV 16-bit
  ```

### Help

```bash
# full, colorized usage summary
./run.sh -h
```

## Experiments

| Model | Mode | Num Requests | Num GPUs  | Warm-up (s) | Throughput (TPS) | METEOR | BERTScore |
|-------|------|--------------|-----------|-------------|------------------|--------|-----------|
| `gpt-oss-20b` | `getp` | 7120 | 8x AMD MI250 | 20 | 30086 | 0.52 | 0.98 |
| `gpt-oss-120b` | `getp` | 6144 | 8x AMD MI250 | 46 | 9993 | 0.30 | 0.99 |

## Acknowledgments

This project was part of the GPU Engineer Training Program, a collaboration between [Moreh](https://www.linkedin.com/company/moreh-vietnam/) and [THUNDER Research Group](http://snuvm.snu.ac.kr/) (Seoul National University). We thank them for their HPC expertise and generous AMD GPU support, without which this work would not have been possible.

## Contact

If you have any questions, please feel free to open an issue or email the [authors.](mailto:tuanleducanh78202@gmail.com)
