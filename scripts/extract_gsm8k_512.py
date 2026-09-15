#!/usr/bin/env python3
"""Extract Qwen + Gemma traces on GSM8K 250 at 512 tokens.

Phase 11 — Second task-generalization point (math reasoning correctness).

Complement to HotpotQA (multi-hop factual QA, Phase 10): if HotpotQA
showed framework correctly *fail-predicts* (sep_d ≈ 0, AUC ≈ 0.5), GSM8K
gives a second non-safety task to test whether the prediction generalizes
across different correctness-disagreement structures.

Outputs:
  results/disagree_routing/phase11_gsm8k_traces_qwen3.5-2b.json
  results/disagree_routing/phase11_gsm8k_traces_gemma-4-e2b.json
  results/disagree_routing/phase11_gsm8k_meta.json
"""
import gc
import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = "results/disagree_routing"
META_OUT = f"{OUT_DIR}/phase11_gsm8k_meta.json"

MODELS = {
    "qwen3.5-2b":  "Qwen/Qwen3.5-2B",
    "gemma-4-e2b": "google/gemma-4-E2B-it",
}

PROMPT_TEMPLATE = "Solve the following math word problem. Show your reasoning step by step and give the final numeric answer.\n\nProblem: {question}\n\nAnswer:"


def extract_gold(answer_text):
    """GSM8K gold answer: number after '####' marker."""
    m = re.search(r'####\s*([\-\d,\.]+)', answer_text)
    if m:
        return m.group(1).replace(',', '').strip()
    return ""


def make_input(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def extract_one(tokenizer, model, prompt, device, max_new=512):
    text = make_input(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            do_sample=True, temperature=0.7, top_p=0.95, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    print("[main] loading GSM8K test split...", flush=True)
    ds = load_dataset("gsm8k", "main", split="test")
    # Take first 250 for consistency with HotpotQA setup
    SAMPLES = 250
    items = []
    for i, row in enumerate(ds):
        if i >= SAMPLES:
            break
        gold = extract_gold(row["answer"])
        prompt = PROMPT_TEMPLATE.format(question=row["question"])
        items.append({"id": f"gsm_{i}", "prompt": prompt,
                      "question": row["question"], "answer": gold})
    print(f"[main] prepared {len(items)} GSM8K prompts", flush=True)

    json.dump({"prompts": items}, open(META_OUT, "w"), ensure_ascii=False)
    print(f"[main] saved meta: {META_OUT}", flush=True)

    prompt_ids = [p["id"] for p in items]
    prompts = [p["prompt"] for p in items]

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype = torch.float32
    print(f"[main] device={device}, dtype={dtype}, max_new=512", flush=True)

    for name, model_id in MODELS.items():
        save_path = f"{OUT_DIR}/phase11_gsm8k_traces_{name}.json"
        if os.path.exists(save_path):
            try:
                done = json.load(open(save_path)).get("records", [])
            except Exception:
                done = []
            if len(done) >= len(prompts):
                print(f"[{name}] already complete ({len(done)} traces)", flush=True)
                continue
            done_ids = {r["id"] for r in done}
            out = list(done)
            print(f"[{name}] resuming with {len(done_ids)} done", flush=True)
        else:
            done_ids = set()
            out = []

        print(f"\n=== Loading {name} ===", flush=True)
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
        model.eval()
        print(f"[{name}] loaded in {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        baseline_done = len(done_ids)
        for pid, prompt in zip(prompt_ids, prompts):
            if pid in done_ids:
                continue
            try:
                trace = extract_one(tokenizer, model, prompt, device, max_new=512)
            except Exception as e:
                trace = f"__ERROR__: {type(e).__name__}: {e}"
            out.append({"id": pid, "prompt": prompt, "trace": trace})

            done_now = len(out) - baseline_done
            if done_now % 25 == 0 and done_now > 0:
                elapsed = time.time() - t0
                rate = done_now / max(elapsed, 1)
                rem = (len(prompts) - len(out)) / max(rate, 0.001)
                print(f"  [{name}] {len(out)}/{len(prompts)} ({rate:.3f}/s, ~{rem/60:.1f}min left)", flush=True)
                json.dump({"model": name, "records": out}, open(save_path, "w"), ensure_ascii=False)

        json.dump({"model": name, "records": out}, open(save_path, "w"), ensure_ascii=False)
        print(f"[{name}] done: {len(out)} traces in {(time.time()-t0)/60:.1f}min", flush=True)

        del model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    print("\n=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
