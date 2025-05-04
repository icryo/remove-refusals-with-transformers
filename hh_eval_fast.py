#!/usr/bin/env python3
"""
Fast A/B evaluator for Householder edits (longer chains).
Batches prompts, uses kv-cache, and generates more tokens so you see the full <think> chain.

Example:
  python hh_eval_fast.py \
    --base_id Qwen/Qwen3-30B-A3B \
    --edited   ./qwen3_30b_bleak \
    --harmful  harmful.txt \
    --harmless harmless.txt \
    --n 20 \
    --batch_size 8 \
    --max_new_tokens 64
"""

import argparse, random, json
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)

def load_4bit(model_id):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        ),
        attn_implementation="eager",
    )

def complete_batch(model, tok, prompts, max_new_tokens, gen_conf):
    # Build batched inputs with left-padding
    txts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in prompts
    ]
    enc = tok(
        txts,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    outs = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        generation_config=gen_conf,
    )

    # Strip off prompt tokens, decode the rest
    input_len = enc["input_ids"].shape[1]
    completions = []
    for seq in outs:
        gen_tokens = seq[input_len:]
        completions.append(tok.decode(gen_tokens, skip_special_tokens=True))
    return completions

def batched(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

def main(base_id, edited, harmful, harmless, n, batch_size, max_new_tokens):
    # Load tokenizer & fix padding
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    tok.padding_side = "left"

    # Load both models in 4-bit for speed
    baseline = load_4bit(base_id)
    edited_model = load_4bit(edited)

    # Sample prompts
    harm_prompts  = random.sample(Path(harmful).read_text().splitlines(), n)
    less_prompts  = random.sample(Path(harmless).read_text().splitlines(), n)

    # Generation config
    gen_conf = GenerationConfig(
        do_sample=True,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        pad_token_id=tok.eos_token_id,
    )

    flips = 0
    results = []
    total = 2 * n
    bar = tqdm(total=total, desc="Evaluating")

    # Harmful prompts: expect edited to be *shorter* (suppression)
    for chunk in batched(harm_prompts, batch_size):
        base_outs = complete_batch(baseline, tok, chunk, max_new_tokens, gen_conf)
        edit_outs = complete_batch(edited_model, tok, chunk, max_new_tokens, gen_conf)
        for p, btxt, etxt in zip(chunk, base_outs, edit_outs):
            if len(etxt) < len(btxt):
                flips += 1
            results.append({"type":"harmful","prompt":p,"base":btxt,"edited":etxt})
        bar.update(len(chunk))

    # Harmless prompts: expect edited to be *longer* or equal
    for chunk in batched(less_prompts, batch_size):
        base_outs = complete_batch(baseline, tok, chunk, max_new_tokens, gen_conf)
        edit_outs = complete_batch(edited_model, tok, chunk, max_new_tokens, gen_conf)
        for p, btxt, etxt in zip(chunk, base_outs, edit_outs):
            if len(etxt) > len(btxt):
                flips += 1
            results.append({"type":"harmless","prompt":p,"base":btxt,"edited":etxt})
        bar.update(len(chunk))

    bar.close()
    print(f"\nFlipped {flips} / {total} = {flips/total:.2%} of prompts.")

    # Print 3 random side-by-side samples
    for sample in random.sample(results, min(3, len(results))):
        print("\n---", sample["type"].upper(), "PROMPT ---")
        print(sample["prompt"])
        print("BASE:\n", sample["base"], "\n")
        print("EDIT:\n", sample["edited"], "\n")

    # Save full results
    with open("hh_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved full outputs to hh_eval_results.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_id",      required=True, help="Baseline model ID")
    p.add_argument("--edited",       required=True, help="Edited model directory")
    p.add_argument("--harmful",      default="harmful.txt", help="Harmful prompt file")
    p.add_argument("--harmless",     default="harmless.txt", help="Harmless prompt file")
    p.add_argument("--n",            type=int, default=20, help="Number of samples per class")
    p.add_argument("--batch_size",   type=int, default=8, help="Batch size for generation")
    p.add_argument("--max_new_tokens", type=int, default=64, help="How many tokens to generate per prompt")
    args = p.parse_args()
    main(
        args.base_id,
        args.edited,
        args.harmful,
        args.harmless,
        args.n,
        args.batch_size,
        args.max_new_tokens,
    )
