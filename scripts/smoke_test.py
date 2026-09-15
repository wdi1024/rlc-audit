#!/usr/bin/env python3
"""MPS smoke test for Qwen3.5-2B and Gemma-4-E2B.

Validates each model loads, generates, and produces sensible output on MPS.
"""
import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "gemma-4-e2b": "google/gemma-4-E2B-it",
}

PROMPT = "What is the capital of France? Think step by step."


def test_model(name, model_id, device, dtype):
    print(f"\n{'='*60}\n{name} ({model_id})\n{'='*60}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  tokenizer loaded ({time.time()-t0:.1f}s)")

    t0 = time.time()
    attn_impl = "eager" if device == "mps" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()
    print(f"  model loaded ({time.time()-t0:.1f}s)")

    messages = [{"role": "user", "content": PROMPT}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=80,
            do_sample=True, temperature=0.7, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"  generated 80 tokens ({time.time()-t0:.1f}s)")
    print(f"  output (first 200 chars): {response[:200]!r}")
    print(f"  RSS approx: {torch.mps.current_allocated_memory() / 1e9:.2f}GB" if device == "mps" else "")

    del model
    if device == "mps":
        torch.mps.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--only", help="comma-separated model keys to test")
    args = parser.parse_args()

    keys = list(MODELS.keys())
    if args.only:
        keys = [k for k in keys if k in args.only.split(",")]

    dtype = torch.float16 if args.device == "cuda" else torch.float32

    results = {}
    for k in keys:
        try:
            test_model(k, MODELS[k], args.device, dtype)
            results[k] = "OK"
        except Exception as e:
            results[k] = f"FAIL: {type(e).__name__}: {e}"
            print(f"\n  ERROR: {results[k]}")

    print(f"\n{'='*60}\nSMOKE TEST SUMMARY\n{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
