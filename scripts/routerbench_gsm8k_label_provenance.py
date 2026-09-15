#!/usr/bin/env python3
"""What the RouterBench grade-school-math labels describe. (2026-08-20)

Section `sec:routerbench` returns *not instantiable* on this slice: its rows carry
fractional exact-match labels over a single stored response, so no function of the
released evidence can reproduce the released label.  That verdict is an argument
from the label's arithmetic.  This script turns it into a measurement, by
recovering the gold answers, scoring the single stored response for each model, and
comparing that to the stored label.

What it establishes, and what it does not.  It establishes that on a fifth of
RouterBench the released label does not describe the released response, that the
two disagree on essentially every row, and that the disagreement is
model-dependent rather than a constant offset -- so it does not cancel in a
comparison.  It does not overturn RouterBench's leaderboard: aggregate model
rankings are nearly unchanged, and we report that too.

Requires network on first run (the RouterBench dump and GSM8K gold answers).

  python3 routerbench_gsm8k_label_provenance.py
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path("/tmp/rb")
SLICE = "grade-school-math"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def load_routerbench() -> pd.DataFrame:
    pkl = CACHE / "routerbench_0shot.pkl"
    if not pkl.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download("withmartian/routerbench", "routerbench_0shot.pkl",
                        repo_type="dataset", local_dir=str(CACHE))
    return pd.read_pickle(pkl)


def load_gold() -> dict[str, str]:
    path = Path("/tmp/gsm8k_gold.json")
    if path.exists():
        return json.loads(path.read_text())
    from datasets import load_dataset
    gold = {}
    for split in ("test", "train"):
        for r in load_dataset("openai/gsm8k", "main", split=split):
            gold[r["question"].strip()] = r["answer"].split("####")[-1].strip().replace(",", "")
    path.write_text(json.dumps(gold))
    return gold


def as_list(x):
    if isinstance(x, list):
        return x
    try:
        v = ast.literal_eval(str(x))
        return v if isinstance(v, list) else [str(x)]
    except (ValueError, SyntaxError):
        return [str(x)]


def target_question(prompt) -> str:
    tail = str(as_list(prompt)[-1])
    m = re.findall(r"Question:\s*(.+?)\s*(?:Answer:|$)", tail, re.S)
    return (m[-1] if m else tail).strip()


def exact_match(response, gold: str | None) -> int | None:
    if gold is None:
        return None
    parts = as_list(response)
    nums = NUM.findall(str(parts[0] if parts else "").replace("$", ""))
    if not nums:
        return 0
    last = nums[-1].replace(",", "").rstrip(".")
    try:
        return int(abs(float(last) - float(gold)) < 1e-6)
    except ValueError:
        return int(last == gold)


def main() -> None:
    df = load_routerbench()
    gold = load_gold()
    models = [c for c in df.columns if "|" not in c
              and c not in ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to")]
    g = df[df.eval_name == SLICE].copy()
    g["gold"] = g["prompt"].map(target_question).map(gold.get)

    n_matched = int(g["gold"].notna().sum())
    print(f"{SLICE}: {len(g)} rows, {100*len(g)/len(df):.1f}% of RouterBench; "
          f"gold recovered for {n_matched} ({100*n_matched/len(g):.1f}%)")

    # 1. the label cannot be the response's correctness: it is not binary
    lab_all = pd.concat([g[m].astype(float) for m in models])
    frac = float((~lab_all.round(6).isin([0.0, 1.0])).mean())
    resp_counts = g[f"{models[0]}|model_response"].map(lambda x: len(as_list(x))).value_counts()
    print(f"  labels strictly between 0 and 1: {frac:.3f};  "
          f"responses stored per row: {dict(resp_counts)}")

    # 2. how far the label is from what the stored response actually does
    rows = []
    print(f"\n{'model':38s} {'label':>8s} {'stored-response EM':>19s} {'gap':>8s} {'label==EM':>10s}")
    for m in models:
        lab = g[m].astype(float).values
        em = np.array([exact_match(x, gd) for x, gd in
                       zip(g[f"{m}|model_response"], g["gold"])], dtype=float)
        rows.append({"model": m, "mean_label": float(lab.mean()), "mean_em": float(em.mean()),
                     "gap": float(lab.mean() - em.mean()),
                     "label_equals_em": float(np.mean(np.isclose(lab, em)))})
        r = rows[-1]
        print(f"{m:38s} {r['mean_label']:8.4f} {r['mean_em']:19.4f} "
              f"{r['gap']:+8.4f} {r['label_equals_em']:10.4f}")

    agree = float(np.mean([r["label_equals_em"] for r in rows]))
    gaps = [r["gap"] for r in rows]
    print(f"\n  the stored label equals the stored response's correctness on {agree:.4f} of rows")
    print(f"  and the discrepancy is model-dependent: {min(gaps):+.3f} to {max(gaps):+.3f}, "
          f"a spread of {max(gaps)-min(gaps):.3f}")

    # 3. the released oracle never records an unsolvable item on this slice
    def unsolvable(sub):
        return int((sub.oracle_model_to_route_to == "no_model_correct").sum())
    hs = df[df.eval_name == "hellaswag"]
    print(f"\n  rows the released oracle marks unsolvable: {SLICE} {unsolvable(g)}/{len(g)}, "
          f"hellaswag {unsolvable(hs)}/{len(hs)}")

    # 4. what this does and does not change
    by_lab = {r["model"]: i + 1 for i, r in enumerate(sorted(rows, key=lambda r: -r["mean_label"]))}
    by_em = {r["model"]: i + 1 for i, r in enumerate(sorted(rows, key=lambda r: -r["mean_em"]))}
    moved = sum(1 for m in models if by_lab[m] != by_em[m])
    rho_slice, _ = spearmanr([by_lab[m] for m in models], [by_em[m] for m in models])

    full = df[models].astype(float).mean().sort_values(ascending=False)
    nog = df[df.eval_name != SLICE][models].astype(float).mean().sort_values(ascending=False)
    r_full = {m: i + 1 for i, m in enumerate(full.index)}
    r_nog = {m: i + 1 for i, m in enumerate(nog.index)}
    rho_agg, _ = spearmanr([r_full[m] for m in models], [r_nog[m] for m in models])
    oracle_full = df[models].astype(float).max(axis=1).mean()
    oracle_nog = df[df.eval_name != SLICE][models].astype(float).max(axis=1).mean()

    print(f"\n  on this slice, ranking by label vs by response: {moved}/{len(models)} models move "
          f"(Spearman {rho_slice:.3f})")
    print(f"  aggregate RouterBench ranking, slice included vs dropped: Spearman {rho_agg:.3f} "
          f"({sum(1 for m in models if r_full[m]!=r_nog[m])}/{len(models)} move)")
    print(f"  oracle quality: {oracle_full:.4f} with the slice, {oracle_nog:.4f} without")
    print("\n  => the leaderboard is robust; what is not interpretable is any result on this "
          "slice that conditions on the released response.")

    out = {"slice": SLICE, "n_rows": len(g), "share_of_benchmark": len(g) / len(df),
           "fraction_non_binary_labels": frac,
           "responses_stored_per_row": {int(k): int(v) for k, v in resp_counts.items()},
           "label_equals_response_correctness": agree,
           "gap_min": min(gaps), "gap_max": max(gaps),
           "oracle_unsolvable_rows": {SLICE: unsolvable(g), "hellaswag": unsolvable(hs)},
           "slice_rank_moves": moved, "slice_rank_spearman": float(rho_slice),
           "aggregate_rank_spearman": float(rho_agg),
           "oracle_with_slice": float(oracle_full), "oracle_without_slice": float(oracle_nog),
           "per_model": rows}
    path = ROOT / "analysis_results" / "routerbench_gsm8k_label_provenance_measured.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
