#!/usr/bin/env python3
"""Run our pipeline's Claude Haiku 4.5 refusal judge over SORRY-Bench responses.

Reuses the *exact* SYSTEM_PROMPT and parser from llm_judge_refusal.py — do not
edit the prompt here, that would break the validation chain. The point of this
script is to certify our deployed judge against SORRY-Bench's human labels;
any prompt drift means we are validating a different judge than what §3 / §6 /
§7.7 actually use.

Usage:
    ANTHROPIC_API_KEY=... python scripts/judge_sorrybench.py
    ANTHROPIC_API_KEY=... python scripts/judge_sorrybench.py \\
        --backend openai --model gpt-4o-mini   # cross-judge robustness

Input:  data/sorrybench_human_judgment.json (from fetch_sorrybench.py)
Output: data/sorrybench_judge_<backend>_<model>.json

Cost (Anthropic Haiku 4.5, ~150in / 30out per call):
    n=4000 → ~$1.50
    n=8000 → ~$3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llm_judge_refusal import judge_one  # noqa: E402  reuse exact judge call


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=ROOT / "data" / "sorrybench_human_judgment.json")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path. Auto-derived from backend/model if omitted.")
    p.add_argument("--backend", choices=["openai", "anthropic"], default="anthropic")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--base_url", default=None)
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max parallel judge calls. Lower if hitting rate limits.")
    p.add_argument("--max_n", type=int, default=0,
                   help="Limit records (smoke test). 0 = all.")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip records already in the output file.")
    args = p.parse_args()

    records = load_records(args.input)
    print(f"[judge] loaded {len(records)} SORRY-Bench records from {args.input}")
    if args.max_n > 0:
        records = records[: args.max_n]

    out = args.out or (
        ROOT / "data" / f"sorrybench_judge_{args.backend}_{args.model.replace('/', '_')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    done_ids = set()
    if out.exists() and args.resume:
        prev = json.loads(out.read_text())
        existing = prev.get("records", [])
        done_ids = {r["id"] for r in existing}
        print(f"[judge] resuming: {len(done_ids)} already judged in {out}")

    todo = [r for r in records if r["id"] not in done_ids]
    print(f"[judge] {len(todo)} records to judge "
          f"(backend={args.backend}, model={args.model}, concurrency={args.concurrency})")

    results = list(existing)
    t0 = time.time()
    last_save = t0

    def _job(rec):
        verdict = judge_one(
            prompt=rec["prompt"],
            trace=rec["response"],
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
        )
        return rec, verdict

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(_job, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec, verdict = fut.result()
            results.append({
                "id": rec["id"],
                "category": rec["category"],
                "source_model": rec["source_model"],
                "human_label": rec["human_label"],
                "judge_label": "refuse" if verdict.get("refusal_judge") is True
                               else "fulfill" if verdict.get("refusal_judge") is False
                               else None,
                "judge_rationale": verdict.get("rationale"),
                "judge_raw": verdict.get("raw"),
                "judge_error": verdict.get("error"),
            })
            if i % 25 == 0 or time.time() - last_save > 30:
                out.write_text(json.dumps({
                    "input_path": str(args.input),
                    "backend": args.backend,
                    "model": args.model,
                    "n": len(results),
                    "records": results,
                }, indent=2, ensure_ascii=False))
                last_save = time.time()
                elapsed = time.time() - t0
                print(f"  [{i}/{len(todo)}] saved (elapsed {elapsed:.1f}s)")

    out.write_text(json.dumps({
        "input_path": str(args.input),
        "backend": args.backend,
        "model": args.model,
        "n": len(results),
        "records": results,
    }, indent=2, ensure_ascii=False))
    n_err = sum(1 for r in results if r.get("judge_error"))
    n_none = sum(1 for r in results if r.get("judge_label") is None)
    print(f"[judge] done: {len(results)} records, {n_err} errors, {n_none} unparsed → {out}")


if __name__ == "__main__":
    main()
