#!/usr/bin/env python3
"""Re-adjudicate marker-stripped final spans (reviewer request, 2026-08-13).

`marker_strip_symmetry.py` removes refusal-marker sentences from the final span and
shows that the score's semantic signal largely survives.  That is a claim about the
score.  The complementary claim is about the construct: if the judge still reaches
the same refusal decision after the marker sentences are gone, then the marker is a
*symptom* of the decision rather than the evidence for it, and reading it is not the
same act as reading an opening template.

  y   = original final-channel refusal decision (cached)
  y'  = decision on the same response with marker sentences removed

High agreement(y, y') means the construct does not live in the marker.  Low
agreement means the marker carries the decision, and the paper must say that the
repair absorbs the label rather than approaching it.

Responses that become empty after stripping are reported separately: an empty text
carries no decision, and scoring them as refusals or compliances would manufacture
whichever answer we wanted.

  python3 judge_marker_stripped.py --limit 60      # cost probe first
  python3 judge_marker_stripped.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)

from llm_judge_refusal import judge_one  # noqa: E402  (after sys.path insert)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
JUDGE_MODEL = "claude-haiku-4-5-20251001"


def strip_marker_sentences(text: str) -> str:
    return " ".join(s for s in SENT_SPLIT.split(text or "")
                    if not agcr.is_kw(s)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N pairs")
    ap.add_argument("--model", default=JUDGE_MODEL)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    data_dir = ROOT / "results" / "disagree_routing"
    pair = agcr.PAIR_SPECS[0]
    rec_a = agcr.load_records(data_dir, pair.phase_a, pair.model_a)
    rec_b = agcr.load_records(data_dir, pair.phase_b, pair.model_b)
    ja = agcr.load_final_judge(data_dir, pair.phase_a, pair.model_a, pair.judge_tag)
    jb = agcr.load_final_judge(data_dir, pair.phase_b, pair.model_b, pair.judge_tag)
    pa = agcr.load_prompts(data_dir, pair.phase_a)
    pb = agcr.load_prompts(data_dir, pair.phase_b)
    ids = sorted(set(rec_a) & set(rec_b) & set(ja) & set(jb) & set(pa) & set(pb))
    ids = [i for i in ids if pa[i] == pb[i]
           and agcr.final_text(rec_a[i]).strip() and agcr.final_text(rec_b[i]).strip()]
    if args.limit:
        ids = ids[:args.limit]

    out_path = ROOT / "analysis_results" / "judge_marker_stripped.json"
    cache = json.load(open(out_path))["per_item"] if out_path.exists() else {}
    print(f"{len(ids)} pairs; {len(cache)} already cached")

    per_item = dict(cache)
    for n, i in enumerate(ids, 1):
        if i in per_item:
            continue
        row = {"prompt": pa[i], "orig": {}, "stripped": {}, "empty": {}}
        for side, rec, orig in (("a", rec_a, ja), ("b", rec_b, jb)):
            full = agcr.final_text(rec[i])
            strip = strip_marker_sentences(full)
            row["orig"][side] = bool(orig[i])
            row["empty"][side] = len(strip) == 0
            if not strip:
                row["stripped"][side] = None
                continue
            r = judge_one(pa[i], strip, "anthropic", args.model, None)
            row["stripped"][side] = bool(r["refusal_judge"]) \
                if r.get("refusal_judge") is not None else None
        per_item[i] = row
        if n % 25 == 0 or n == len(ids):
            json.dump({"judge_model": args.model, "n_pairs": len(per_item),
                       "per_item": per_item}, open(out_path, "w"), indent=1)
            print(f"  judged {n}/{len(ids)}")

    # Side-level and pair-level agreement, empties excluded and counted.
    sy, sy2 = [], []
    py, py2 = [], []
    n_empty_side = n_empty_pair = 0
    for i, row in per_item.items():
        for side in ("a", "b"):
            if row["empty"][side] or row["stripped"][side] is None:
                n_empty_side += 1
                continue
            sy.append(int(row["orig"][side]))
            sy2.append(int(row["stripped"][side]))
        if any(row["empty"][s] or row["stripped"][s] is None for s in ("a", "b")):
            n_empty_pair += 1
            continue
        py.append(int(row["orig"]["a"] != row["orig"]["b"]))
        py2.append(int(row["stripped"]["a"] != row["stripped"]["b"]))

    sy, sy2, py, py2 = map(np.array, (sy, sy2, py, py2))
    side_agree = float((sy == sy2).mean())
    side_kappa = float(cohen_kappa_score(sy, sy2)) if len(set(sy)) > 1 else float("nan")
    pair_agree = float((py == py2).mean())
    pair_kappa = float(cohen_kappa_score(py, py2)) if len(set(py)) > 1 else float("nan")

    print(f"\nside-level: n={len(sy)}  agreement {side_agree:.3f}  kappa {side_kappa:.3f}")
    print(f"pair-level: n={len(py)}  agreement {pair_agree:.3f}  kappa {pair_kappa:.3f}")
    print(f"excluded because stripping emptied the response: {n_empty_side} sides, "
          f"{n_empty_pair} pairs")
    print(f"refusal rate before {sy.mean():.3f} -> after {sy2.mean():.3f}; "
          f"pair-disagreement rate before {py.mean():.3f} -> after {py2.mean():.3f}")

    summary = {"judge_model": args.model, "n_pairs": len(per_item),
               "side": {"n": int(len(sy)), "agreement": side_agree, "kappa": side_kappa,
                        "rate_before": float(sy.mean()), "rate_after": float(sy2.mean())},
               "pair": {"n": int(len(py)), "agreement": pair_agree, "kappa": pair_kappa,
                        "rate_before": float(py.mean()), "rate_after": float(py2.mean())},
               "excluded_empty": {"sides": n_empty_side, "pairs": n_empty_pair},
               "per_item": per_item}
    json.dump(summary, open(out_path, "w"), indent=1)
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
