import os
import sys
import threading

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextStreamer

# Threads
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_interop_threads(1)

MODEL_PATH = "/home/ubuntu/data/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,  # keep uniform dtype on CPU
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.to(torch.float32)

prompt = "Tell me a joke"
enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
input_ids = enc.input_ids
attention_mask = enc.attention_mask if "attention_mask" in enc else torch.ones_like(
    input_ids)

# Streamer prints tokens to stdout as they are generated
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

with torch.no_grad():
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        do_sample=False,  # greedy
        num_beams=1,
        use_cache=True,
        streamer=streamer,  # <-- streaming happens here
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
# TextStreamer already printed the continuation incrementally.
