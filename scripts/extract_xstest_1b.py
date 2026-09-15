#!/usr/bin/env python3
"""Extract Llama-3.2-1B + Qwen3-1.7B traces on XSTest 450 (Phase 12).

E2: SLM-scale ablation — does the gate's admit decision change at 1B scale?

Reuses XSTest 450 prompts from phase7 meta. Output:
  results/disagree_routing/phase12_xstest450_1b_traces_llama-3.2-1b.json
  results/disagree_routing/phase12_xstest450_1b_traces_qwen3-1.7b.json
"""
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = "results/disagree_routing"
META_IN = f"{OUT_DIR}/phase7_xstest450_512tok_meta.json"

MODELS = {
    "llama-3.2-1b": {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "revision": None,  # Gated model; original generation snapshot unavailable.
    },
    "qwen3-1.7b": {
        "id": "Qwen/Qwen3-1.7B",
        "revision": "70d244c",
    },
}


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
    print("[main] loading XSTest 450 meta...", flush=True)
    meta = json.load(open(META_IN))["prompts"]
    print(f"[main] n_prompts={len(meta)}", flush=True)

    prompt_ids = [p["id"] for p in meta]
    prompts = [p["prompt"] for p in meta]

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype = torch.float32
    print(f"[main] device={device}, dtype={dtype}, max_new=512", flush=True)

    for name, info in MODELS.items():
        model_id = info["id"]
        revision = info.get("revision")
        save_path = f"{OUT_DIR}/phase12_xstest450_1b_traces_{name}.json"
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

        print(f"\n=== Loading {name} ({model_id}@{revision or 'default'}) ===", flush=True)
        t0 = time.time()
        hf_kwargs = {"trust_remote_code": True}
        if revision:
            hf_kwargs["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype,
            attn_implementation="eager",
            **hf_kwargs,
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
                json.dump({"model": name, "model_id": model_id,
                           "model_revision": revision,
                           "records": out}, open(save_path, "w"), ensure_ascii=False)

        json.dump({"model": name, "model_id": model_id,
                   "model_revision": revision,
                   "records": out}, open(save_path, "w"), ensure_ascii=False)
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
