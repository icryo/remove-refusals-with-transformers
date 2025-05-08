#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"                # 3.13 is still pre-release; drop to the latest GA
# dependencies = [
#   "accelerate",
#   "bitsandbytes",
#   "einops",
#   "jaxtyping",
#   "torch",
#   "tqdm",
#   "transformers",
# ]
# ///

import sys, subprocess, textwrap
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TextStreamer,
)

# ------------- everything below is your original logic -------------
model_id, prompt, *rest = sys.argv[1:]
max_new, temp, top_p, top_k = (rest+[150,0.7,0.9,40])[:4]
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tok.padding_side = "left"
bnb_conf = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_conf,
    attn_implementation="eager",
)
model.eval()

chat = tok.apply_chat_template(
    [{"role":"user","content":prompt}],
    add_generation_prompt=True,
    tokenize=False,
)
inputs = tok(chat, return_tensors="pt",
             padding=True, return_attention_mask=True).to(model.device)
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
model.generate(**inputs,
               max_new_tokens=int(max_new),
               do_sample=True,
               temperature=float(temp),
               top_k=int(top_k),
               top_p=float(top_p),
               pad_token_id=tok.eos_token_id,
               eos_token_id=tok.eos_token_id,
               use_cache=True,
               streamer=streamer)
