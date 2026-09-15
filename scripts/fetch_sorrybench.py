#!/usr/bin/env python3
"""Fetch SORRY-Bench human-judgment dataset for judge validation.

SORRY-Bench publishes the human-annotation data across two HF datasets that must
be joined:

- sorry-bench/sorry-bench-202406            : prompts, keyed by (question_id, prompt_style)
- sorry-bench/sorry-bench-human-judgment-202406 : (question_id, prompt_style, model_id) -> response + human_score

Both are gated. Run `huggingface-cli login` first.

Usage:
    python scripts/fetch_sorrybench.py
    python scripts/fetch_sorrybench.py --max_n 200      # smoke
    python scripts/fetch_sorrybench.py --base_only      # only base prompt_style

Output schema (one record per (response, human_label) pair):
    {
      "id": "sb_<idx>",
      "prompt": "<unsafe instruction>",
      "response": "<model response text>",
      "human_label": "refuse"|"fulfill",
      "human_score_raw": 0.0|1.0,
      "category": "<sorry-bench category id>",
      "source_model": "<model id that produced the response>",
      "prompt_style": "<base|...>",
      "question_id": <int>,
    }

Conventions (per SORRY-Bench):
    human_score == 0.0 → response refused (label "refuse")
    human_score == 1.0 → response fulfilled the unsafe request ("fulfill")

The category field in this dataset is an integer id; the human-readable name
needs the SORRY-Bench taxonomy file to decode (printed at the end of the run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "sorrybench_human_judgment.json"

PROMPT_SLUG = "sorry-bench/sorry-bench-202406"
JUDGE_SLUG = "sorry-bench/sorry-bench-human-judgment-202406"


def normalize_label(score) -> str | None:
    if score is None:
        return None
    try:
        v = float(score)
    except (TypeError, ValueError):
        return None
    # SORRY-Bench convention: 0 = refused, 1 = fulfilled.
    if v == 0.0:
        return "refuse"
    if v == 1.0:
        return "fulfill"
    # Some snapshots may average annotators -> use majority.
    if v < 0.5:
        return "refuse"
    return "fulfill"


def extract_response(choices) -> str | None:
    if not choices:
        return None
    first = choices[0]
    turns = first.get("turns") if isinstance(first, dict) else None
    if not turns:
        return None
    # Single-turn responses are most common; concatenate if multi-turn just in case.
    if isinstance(turns, list):
        return "\n\n".join(str(t) for t in turns)
    return str(turns)


def extract_prompt(turns) -> str | None:
    if not turns:
        return None
    if isinstance(turns, list):
        return "\n\n".join(str(t) for t in turns)
    return str(turns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max_n", type=int, default=0,
                   help="Limit final records (smoke test). 0 = all.")
    p.add_argument("--base_only", action="store_true",
                   help="Only keep prompt_style=='base' (skip linguistic variants).")
    p.add_argument("--prompt_revision", default=None)
    p.add_argument("--judge_revision", default=None)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    print(f"[fetch] loading prompts: {PROMPT_SLUG}")
    p_kwargs = {"revision": args.prompt_revision} if args.prompt_revision else {}
    prompts_ds = load_dataset(PROMPT_SLUG, **p_kwargs)
    prompt_rows = []
    for split in prompts_ds.keys():
        prompt_rows.extend(list(prompts_ds[split]))
    print(f"  loaded {len(prompt_rows)} prompt rows")

    # Build (question_id, prompt_style) -> {prompt, category}
    prompt_idx = {}
    for r in prompt_rows:
        key = (int(r["question_id"]), str(r.get("prompt_style", "base")))
        prompt_idx[key] = {
            "prompt": extract_prompt(r.get("turns")),
            "category": r.get("category"),
        }

    print(f"[fetch] loading judgments: {JUDGE_SLUG}")
    j_kwargs = {"revision": args.judge_revision} if args.judge_revision else {}
    judge_ds = load_dataset(JUDGE_SLUG, **j_kwargs)
    judge_rows = []
    for split in judge_ds.keys():
        judge_rows.extend(list(judge_ds[split]))
    print(f"  loaded {len(judge_rows)} judgment rows")

    records = []
    skipped = {"no_prompt": 0, "no_response": 0, "bad_label": 0, "non_base": 0}
    for i, jr in enumerate(judge_rows):
        prompt_style = str(jr.get("prompt_style", "base"))
        if args.base_only and prompt_style != "base":
            skipped["non_base"] += 1
            continue
        key = (int(jr["question_id"]), prompt_style)
        prow = prompt_idx.get(key)
        if not prow or not prow["prompt"]:
            skipped["no_prompt"] += 1
            continue
        response = extract_response(jr.get("choices"))
        if not response:
            skipped["no_response"] += 1
            continue
        label = normalize_label(jr.get("human_score"))
        if label is None:
            skipped["bad_label"] += 1
            continue
        records.append({
            "id": f"sb_{i:06d}",
            "prompt": prow["prompt"],
            "response": response,
            "human_label": label,
            "human_score_raw": float(jr["human_score"]),
            "category": str(prow["category"]),
            "source_model": str(jr.get("model_id", "unknown")),
            "prompt_style": prompt_style,
            "question_id": int(jr["question_id"]),
        })

    if args.max_n > 0:
        records = records[: args.max_n]

    refuse_n = sum(1 for r in records if r["human_label"] == "refuse")
    fulfill_n = sum(1 for r in records if r["human_label"] == "fulfill")

    payload = {
        "prompt_slug": PROMPT_SLUG,
        "judge_slug": JUDGE_SLUG,
        "prompt_revision": args.prompt_revision,
        "judge_revision": args.judge_revision,
        "n": len(records),
        "label_distribution": {"refuse": refuse_n, "fulfill": fulfill_n},
        "skipped": skipped,
        "models_present": sorted({r["source_model"] for r in records}),
        "categories_present": sorted({r["category"] for r in records}),
        "prompt_styles_present": sorted({r["prompt_style"] for r in records}),
        "records": records,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[fetch] wrote {args.out}")
    print(f"  records: {len(records)}  refuse: {refuse_n}  fulfill: {fulfill_n}")
    print(f"  models: {len(payload['models_present'])}  "
          f"categories: {len(payload['categories_present'])}  "
          f"prompt_styles: {len(payload['prompt_styles_present'])}")
    print(f"  skipped: {skipped}")


if __name__ == "__main__":
    main()
