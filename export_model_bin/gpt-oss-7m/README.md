# Export GPT-OSS 7m Model

This directory contains scripts to prepare the **7m parameter GPT-OSS model** for use with the C++ runtime.
The process converts the Hugging Face checkpoint into a single `.bin` file that can be directly loaded.

1. **Convert state dict**
   Run the key conversion script on the downloaded Hugging Face weights. This step renames tensors’ keys into the expected format.

```bash
python "${GPT_OSS_REPO_ROOT}/export_model_bin/gpt-oss-7m/convert_state_dict.py" \
  --input  "${MODELS_ROOT}/gpt-oss-7m/model.safetensors" \
  --output "${GPT_OSS_REPO_ROOT}/export_model_bin/gpt-oss-7m/model.safetensors"
```

2. **Generate binary**
   Use the conversion output and `config.json` to build the `.bin` file.

```bash
python "${GPT_OSS_REPO_ROOT}/export_model_bin/gpt-oss-7m/generate_bin.py" \
  --config "${GPT_OSS_REPO_ROOT}/export_model_bin/gpt-oss-7m/config.json" \
  --input  "${GPT_OSS_REPO_ROOT}/export_model_bin/gpt-oss-7m/model.safetensors" \
  --output "${GPT_OSS_REPO_ROOT}/gpt-oss-7m.bin"
```

The final binary is written to: gpt-oss-7m.bin

This file is the one consumed by the C++ inference runtime (`./run`).
