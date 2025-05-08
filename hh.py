#!/usr/bin/env -S uv run --python 3.11 --script
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "numpy<2.0",                     # torch 2.2 wheels compiled with NumPy 1.x
#   "setuptools>=68",                # triton runtime needs it to import
#   "torch",                         # 2.2.x CUDA-12 wheel works on a 4090
#   "transformers>=4.40",
#   "bitsandbytes>=0.43,<0.44",
#   "triton==2.1.0",
#   "accelerate",
#   "einops",
#   "jaxtyping",
#   "tqdm",
# ]
# ///

"""
Householder / refusal-direction edit
-----------------------------------
Example:
  ./hh.py \
     --model_id  Qwen/Qwen3-30B-A3B \
     --output_path  ./qwen3_30b_bleak \
     --harmful  harmful.txt \
     --harmless harmless.txt
"""
import argparse, random, gc, torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# --------- hyper-parameters --------- #
NUM_DIR_SAMPLES = 64
LAYER_FRACTION  = 0.6
SCALE_FACTOR    = 1.0
SKIP_ATTN = False
SKIP_MLP  = False
SKIP_BEGIN, SKIP_END = 0, 0
# ------------------------------------ #


def load_quant(model_id: str):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        attn_implementation="eager",
    )


def build_direction(model, tok, harmful, harmless, layer_idx):
    def to_enc(lines):
        return [
            tok.apply_chat_template(
                [{"role": "user", "content": ln.strip()}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            for ln in lines
        ]

    harm_enc, less_enc = to_enc(harmful), to_enc(harmless)
    bar = tqdm(total=len(harm_enc) + len(less_enc), desc="Dir samples")

    def hid(enc):
        out = model.generate(
            enc.to(model.device),
            max_new_tokens=1,
            use_cache=False,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        bar.update()
        return out.hidden_states[0][layer_idx][:, -1, :].cpu()

    h_hid = [hid(e) for e in harm_enc]
    l_hid = [hid(e) for e in less_enc]
    bar.close()
    vec = torch.stack(h_hid).mean(0) - torch.stack(l_hid).mean(0)
    return (vec / vec.norm()).float()


def reflect(W, v, scale=SCALE_FACTOR):
    v = v.view(-1)
    H = torch.eye(v.size(0)) - 2 * scale * torch.outer(v, v)
    return torch.nn.Parameter((H @ W.float()).bfloat16())


def main(model_id, output_path, harmful, harmless):
    quant = load_quant(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.padding_side = "left"

    layer_idx = int(len(quant.model.layers) * LAYER_FRACTION)
    print(f"Using layer {layer_idx}/{len(quant.model.layers)} for direction")

    harmful_prompts  = random.sample(Path(harmful).read_text().splitlines(),  NUM_DIR_SAMPLES)
    harmless_prompts = random.sample(Path(harmless).read_text().splitlines(), NUM_DIR_SAMPLES)
    direction = build_direction(quant, tok, harmful_prompts, harmless_prompts, layer_idx)

    del quant; gc.collect(); torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    layers = model.model.layers
    v32 = direction

    targets_per = int(not SKIP_ATTN) + int(not SKIP_MLP)
    bar = tqdm(total=(len(layers) - SKIP_BEGIN - SKIP_END) * targets_per, desc="Edit")

    for i, layer in enumerate(layers):
        if i < SKIP_BEGIN or i >= len(layers) - SKIP_END:
            continue

        if not SKIP_ATTN and hasattr(layer.self_attn, "o_proj"):
            layer.self_attn.o_proj.weight = reflect(layer.self_attn.o_proj.weight.data, v32)
            bar.update()

        if not SKIP_MLP:
            edited = False
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                layer.mlp.down_proj.weight = reflect(layer.mlp.down_proj.weight.data, v32)
                edited = True
            if hasattr(layer, "moe"):
                for exp in getattr(layer.moe, "experts", []):
                    if hasattr(exp.ffn, "down_proj"):
                        exp.ffn.down_proj.weight = reflect(exp.ffn.down_proj.weight.data, v32)
                        edited = True
                if hasattr(layer.moe, "shared_mlp") and hasattr(layer.moe.shared_mlp, "down_proj"):
                    layer.moe.shared_mlp.down_proj.weight = reflect(
                        layer.moe.shared_mlp.down_proj.weight.data, v32
                    )
                    edited = True
            if edited:
                bar.update()
    bar.close()

    model.save_pretrained(output_path)
    tok.save_pretrained(output_path)
    print("✅ edited model saved to", output_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--harmful",   default="harmful.txt")
    ap.add_argument("--harmless",  default="harmless.txt")
    main(**vars(ap.parse_args()))
