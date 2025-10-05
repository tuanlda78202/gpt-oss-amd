# Model weights

- openai/gpt-oss-120b (120b): [Hugging Face](https://huggingface.co/openai/gpt-oss-120b)
- openai/gpt-oss-20b (20b): [Hugging Face](https://huggingface.co/openai/gpt-oss-20b)

The downloaded model checkpoints are provided in **safetensors** format.
Before they can be used with the C++ inference runtime, these weights must be **exported into a single `.bin` file**.

This conversion step:

- Dequantizes FP4 weights (if applicable)
- Flattens all tensors into a binary blob
- Produces a `.bin` file compatible with the C++ loader (`./run`)
