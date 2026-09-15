#!/usr/bin/env python3
"""LLM-as-judge for HotpotQA correctness label.

Evaluates whether a model's reasoning trace arrives at the gold answer.
Different from llm_judge_refusal.py which checks for safety-refusal —
this checks for factual-correctness against a gold answer.

Pipeline:
    1. Load Phase 10 traces (Qwen + Gemma)
    2. Load gold answers from phase10_hotpot_meta.json
    3. For each (trace, gold) pair, prompt judge:
       "Does this trace correctly arrive at the answer '{gold}'?"
    4. Save correctness label per trace
    5. Output: phase10_hotpot_correctness_{slm}_{backend}_{model}.json

Usage:
    ANTHROPIC_API_KEY=... python3 llm_judge_correctness_hotpot.py \\
        --backend anthropic \\
        --model claude-haiku-4-5-20251001

Cost (250 prompts × 2 SLMs = 500 calls @ Haiku ~150 input + 30 output):
    ~$0.10
"""
import argparse
import json
import os
import time
from typing import Optional

CORRECTNESS_PROMPT_TEMPLATE = """You are evaluating whether a small language model's reasoning trace correctly identifies an answer to a multi-hop QA question.

Question: {question}
Gold answer: {gold_answer}

Model's trace:
\"\"\"
{trace}
\"\"\"

Does the model's trace correctly arrive at (or clearly identify) the gold answer "{gold_answer}"? Be lenient on phrasing — accept paraphrases, reordering, or partial-match if the core answer entity is correctly identified. But do not accept if the model gives a different answer entity, says "I don't know", or fails to commit to an answer.

Respond with exactly one of:
- "yes" if the trace correctly identifies the gold answer
- "no" if the trace fails to identify the gold answer (wrong answer, hedged, or no answer)

Then on a new line, give one short sentence of rationale."""


def call_anthropic(prompt, model):
    from anthropic import Anthropic
    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_openai(prompt, model, base_url=None):
    from openai import OpenAI
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def parse_response(text):
    """Returns (correct: int|None, rationale: str)."""
    if not text:
        return None, ""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return None, ""
    first = lines[0].lower().strip(' ."*-:')
    rationale = " ".join(lines[1:]) if len(lines) > 1 else ""
    if first.startswith("yes"):
        return 1, rationale
    if first.startswith("no"):
        return 0, rationale
    # fallback: search anywhere
    text_lower = text.strip().lower()
    if text_lower.startswith("yes"):
        return 1, rationale
    if text_lower.startswith("no"):
        return 0, rationale
    return None, rationale


def judge_one(question, trace, gold_answer, backend, model, base_url=None):
    prompt = CORRECTNESS_PROMPT_TEMPLATE.format(
        question=question, gold_answer=gold_answer, trace=trace,
    )
    try:
        if backend == "anthropic":
            text = call_anthropic(prompt, model)
        elif backend == "openai":
            text = call_openai(prompt, model, base_url=base_url)
        else:
            raise ValueError(backend)
        correct, rationale = parse_response(text)
        return dict(correct=correct, rationale=rationale, raw=text)
    except Exception as e:
        return dict(correct=None, rationale=None, raw=None,
                    error=f"{type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/disagree_routing")
    p.add_argument("--backend", choices=["openai", "anthropic"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base_url", default=None)
    p.add_argument("--models", default="qwen3.5-2b,gemma-4-e2b")
    p.add_argument("--max_n", type=int, default=0)
    p.add_argument("--phase", default="phase10_hotpot",
                   help="phase whose traces and meta are judged")
    args = p.parse_args()

    rdir = args.results_dir
    slms = args.models.split(",")

    print(f"Backend: {args.backend} / model: {args.model}")
    if args.base_url:
        print(f"Base URL: {args.base_url}")

    # Load gold answers
    meta = json.load(open(f"{rdir}/{args.phase}_meta.json"))["prompts"]
    gold_map = {p["id"]: {"question": p.get("question", ""), "answer": p.get("answer", "")}
                for p in meta}

    for slm in slms:
        traces_path = f"{rdir}/{args.phase}_traces_{slm}.json"
        if not os.path.exists(traces_path):
            print(f"  [SKIP] {traces_path} missing")
            continue
        traces = json.load(open(traces_path))["records"]
        if args.max_n > 0:
            traces = traces[:args.max_n]

        out_path = f"{rdir}/{args.phase}_correctness_{slm}_{args.backend}_{args.model.replace('/', '_')}.json"

        # Resume support
        existing = []
        done_ids = set()
        if os.path.exists(out_path):
            try:
                existing_data = json.load(open(out_path))
                existing = existing_data.get("records", [])
                done_ids = {r["id"] for r in existing if r.get("correct") is not None}
                print(f"  [{slm}] resume: {len(done_ids)} valid done")
            except Exception:
                existing = []
                done_ids = set()

        out = list(existing)
        out_by_id = {r["id"]: r for r in out}
        flips = 0
        n_done_now = 0
        t0 = time.time()

        print(f"\n[{slm}] judging {len(traces)} traces ({len(traces) - len(done_ids)} pending)")
        for i, r in enumerate(traces):
            pid = r["id"]
            if pid in done_ids:
                continue
            trace = r.get("trace", "") or ""
            gold = gold_map.get(pid, {})
            j = judge_one(gold.get("question", ""), trace, gold.get("answer", ""),
                          args.backend, args.model, args.base_url)
            rec = {
                "id": pid,
                "question": gold.get("question", "")[:200],
                "gold_answer": gold.get("answer", ""),
                "correct_judge": j["correct"],
                "rationale": j["rationale"],
                "raw": j.get("raw"),
                "error": j.get("error"),
            }
            if pid in out_by_id:
                out_by_id[pid].update(rec)
            else:
                out.append(rec)
                out_by_id[pid] = rec

            n_done_now += 1
            if n_done_now % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done_now / max(elapsed, 1)
                rem = (len(traces) - len(out_by_id)) / max(rate, 0.001)
                print(f"  [{slm}] {len(out_by_id)}/{len(traces)} ({rate:.2f}/s, ~{rem/60:.1f}min left)")
                json.dump({
                    "phase": args.phase, "slm": slm,
                    "backend": args.backend, "judge_model": args.model,
                    "records": out,
                }, open(out_path, "w"), ensure_ascii=False)

        json.dump({
            "phase": args.phase, "slm": slm,
            "backend": args.backend, "judge_model": args.model,
            "records": out,
        }, open(out_path, "w"), ensure_ascii=False)
        print(f"\n=== Judge complete ===")
        n_correct = sum(1 for r in out if r.get("correct_judge") == 1)
        n_valid = sum(1 for r in out if r.get("correct_judge") is not None)
        print(f"  [{slm}] {n_correct}/{n_valid} correct = {n_correct/n_valid*100:.1f}% (saved: {out_path})")


if __name__ == "__main__":
    main()
