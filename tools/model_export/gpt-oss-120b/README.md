# Commands

```bash
python "${GPT_OSS_REPO_ROOT}/tools/model_export/gpt-oss-120b/export_model_bin.py" \
  --input  "${MODELS_ROOT}/gpt-oss-120b/original" \
  --config "${GPT_OSS_REPO_ROOT}/tools/model_export/gpt-oss-120b/config.json" \
  --output "${GPT_OSS_REPO_ROOT}/gpt-oss-120b.bin"
```

Notes

- The 120B checkpoint is sharded; the script detects `model.safetensors.index.json` inside the directory and streams tensors shard-by-shard.
- FP4 `.blocks` + `.scales` pairs are dequantized on the fly; other tensors are passed through.
