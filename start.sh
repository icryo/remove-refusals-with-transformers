#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <hf_model_id> \"<prompt>\" [max_new_tokens] [temperature] [top_p] [top_k]"
  exit 1
fi

MODEL_ID="$1"; shift
PROMPT="$1";    shift
MAX_NEW_TOKENS="${1:-150}"
TEMPERATURE="${2:-0.7}"
TOP_P="${3:-0.9}"
TOP_K="${4:-40}"

python3 - <<EOF
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)

# 1) Load
tok = AutoTokenizer.from_pretrained("$MODEL_ID", trust_remote_code=True)
tok.padding_side = "left"
bnb_conf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    "$MODEL_ID",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_conf,
    attn_implementation="eager",
)
model.eval()

# 2) Tokenize
chat = tok.apply_chat_template(
    [{"role":"user","content":"$PROMPT"}],
    add_generation_prompt=True,
    tokenize=False,
)
inputs = tok(chat, return_tensors="pt", padding=True, return_attention_mask=True).to(model.device)

# 3) Streamed generation
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
out = model.generate(
    **inputs,
    max_new_tokens=int("$MAX_NEW_TOKENS"),
    do_sample=True,
    temperature=float("$TEMPERATURE"),
    top_k=int("$TOP_K"),
    top_p=float("$TOP_P"),
    pad_token_id=tok.eos_token_id,
    eos_token_id=tok.eos_token_id,
    use_cache=True,
    streamer=streamer,
)
EOF
