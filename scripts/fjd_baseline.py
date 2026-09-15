#!/usr/bin/env python3
"""FJD (Free Jailbreak Detection) baseline replication.

Replicates the core mechanism of arXiv 2509.14558:
    1. Prepend an affirmative prefix in the assistant turn.
    2. Compute first-token softmax distribution at that position.
    3. Score = max softmax probability (with temperature scaling).

Lower confidence (lower max prob) on the first generated token is hypothesized
to indicate prompt difficulty — torn between refusing and complying with the
affirmative prefix.

We score each XSTest prompt with the FJD signal under each base SLM
(Qwen3.5-2B, Gemma-4-E2B-it) and save per-prompt scores.

Comparison with our cross-SLM disagreement signal is performed in
compare_baselines.py.
"""
import argparse
import gc
import json
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "gemma-4-e2b": "google/gemma-4-E2B-it",
}

AFFIRMATIVE_PREFIX = "Sure, I'll help you with that. "


def load_model(name, device, dtype):
    model_id = MODELS[name]
    print(f"[{name}] loading {model_id}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    attn_impl = "eager" if device == "mps" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()
    print(f"[{name}] loaded in {time.time()-t0:.0f}s", flush=True)
    return tokenizer, model


def make_input_with_affirmative(tokenizer, prompt, prefix):
    """Build chat-template prompt and append affirmative prefix in assistant turn.

    FJD targets standard (non-thinking) chat completions — we explicitly disable
    thinking-mode placeholders so the prefix lands directly in the response slot,
    not inside a <think> block. The model's NEXT token logits at the last
    position give the FJD signal.
    """
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        base = tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        base = tokenizer.apply_chat_template(messages, **kwargs)
    return base + prefix


def fjd_score_one(tokenizer, model, prompt, device, temperature=0.7):
    """Compute FJD score = max softmax probability at first generation position."""
    text = make_input_with_affirmative(tokenizer, prompt, AFFIRMATIVE_PREFIX)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    # logits at last input position -> distribution over next token
    logits = out.logits[0, -1, :].float()
    if temperature is not None and temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    max_prob = float(probs.max().item())
    # Also record entropy for completeness
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())
    return {"max_prob": max_prob, "entropy": entropy}


def run_fjd_for_model(name, prompts, prompt_ids, device, dtype, save_path, save_every=50):
    tokenizer, model = load_model(name, device, dtype)
    out = []
    t0 = time.time()
    for i, (pid, prompt) in enumerate(zip(prompt_ids, prompts)):
        try:
            scores = fjd_score_one(tokenizer, model, prompt, device)
            rec = {"id": pid, "prompt": prompt, **scores}
        except Exception as e:
            rec = {"id": pid, "prompt": prompt, "error": f"{type(e).__name__}: {e}"}
        out.append(rec)
        if (i + 1) % save_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(prompts) - i - 1) / rate
            print(f"  [{name}] {i+1}/{len(prompts)} ({rate:.2f}/s, ~{remaining/60:.0f}min left)",
                  flush=True)
            with open(save_path, "w") as f:
                json.dump({
                    "model": name, "model_id": MODELS[name],
                    "prefix": AFFIRMATIVE_PREFIX, "records": out,
                }, f, ensure_ascii=False)
    with open(save_path, "w") as f:
        json.dump({
            "model": name, "model_id": MODELS[name],
            "prefix": AFFIRMATIVE_PREFIX, "records": out,
        }, f, ensure_ascii=False)
    elapsed = time.time() - t0
    print(f"[{name}] done: {len(out)} prompts in {elapsed/60:.1f}min", flush=True)

    del model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return out


def load_xstest(data_dir, max_n):
    path = os.path.join(data_dir, "xstest.jsonl")
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                # Derive proper safety label from XSTest type field:
                # contrast_* types are the unsafe prompts; others are safe.
                t = d.get("type", "")
                d["label_safety"] = "unsafe" if t.startswith("contrast_") else "safe"
                out.append(d)
                if max_n > 0 and len(out) >= max_n:
                    break
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/")
    p.add_argument("--output_dir", default="results/disagree_routing/")
    p.add_argument("--device", default="mps")
    p.add_argument("--max_xstest", type=int, default=450)
    p.add_argument("--models", default="qwen3.5-2b,gemma-4-e2b")
    p.add_argument("--phase", default="phase3_xstest_full_fjd")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    models = args.models.split(",")

    print(f"Loading XSTest (max={args.max_xstest})...", flush=True)
    xs = load_xstest(args.data_dir, args.max_xstest)
    prompts = [d["prompt"] for d in xs]
    prompt_ids = [f"xs_{i}" for i in range(len(xs))]
    print(f"Total prompts: {len(prompts)}", flush=True)

    # Save meta with safety labels (derived from type)
    meta_path = os.path.join(args.output_dir, f"{args.phase}_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "prompts": [{
                "id": f"xs_{i}",
                "task": "xstest",
                "prompt": d["prompt"],
                "type": d.get("type"),
                "label_safety": d["label_safety"],
            } for i, d in enumerate(xs)]
        }, f, ensure_ascii=False, indent=2)
    print(f"Meta saved: {meta_path}", flush=True)

    for name in models:
        save_path = os.path.join(args.output_dir, f"{args.phase}_scores_{name}.json")
        if os.path.exists(save_path):
            with open(save_path) as f:
                existing = json.load(f)
            if len(existing.get("records", [])) >= len(prompts):
                print(f"[{name}] already complete: {save_path}", flush=True)
                continue
        run_fjd_for_model(name, prompts, prompt_ids, args.device, dtype, save_path)

    print("\nFJD BASELINE DONE", flush=True)


if __name__ == "__main__":
    main()
