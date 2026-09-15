#!/usr/bin/env python3
"""Extract RouterBench pair data for an external RLC-Audit.

Reads the public RouterBench 0-shot dump (withmartian/routerbench) and emits
three artifacts under results/routerbench_audit/:

- mtbench_pair.json: 80 mtbench conversations for the audited pair with
  stored GPT-4-judge scores and full response texts (judged later by Haiku).
- mmlu_law_pair.json: all mmlu-professional-law rows with letter responses
  and stored exact-match correctness labels.
- gsm8k_label_provenance.json: label-value histograms showing that stored
  grade-school-math labels are fractional aggregates while only a single
  response string is stored per model.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

PKL = Path("/tmp/routerbench_audit/routerbench_0shot.pkl")
OUT = Path(__file__).resolve().parents[1] / "results/routerbench_audit"
MODEL_A = "claude-instant-v1"
MODEL_B = "mistralai/mixtral-8x7b-chat"


def parse_listish(x) -> list[str]:
    if isinstance(x, list):
        return [str(t) for t in x]
    s = str(x)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(t) for t in v]
    except (ValueError, SyntaxError):
        pass
    return [s]


def join_turns(x) -> str:
    return "\n\n".join(parse_listish(x))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(PKL)

    mt = df[df.eval_name.str.startswith("mtbench")]
    mt_rows = []
    for _, r in mt.iterrows():
        mt_rows.append({
            "sample_id": r["sample_id"],
            "eval_name": r["eval_name"],
            "prompt_turns": parse_listish(r["prompt"]),
            "score_a": float(r[MODEL_A]),
            "score_b": float(r[MODEL_B]),
            "response_a": join_turns(r[f"{MODEL_A}|model_response"]),
            "response_b": join_turns(r[f"{MODEL_B}|model_response"]),
        })
    (OUT / "mtbench_pair.json").write_text(json.dumps({
        "model_a": MODEL_A, "model_b": MODEL_B, "records": mt_rows,
    }, indent=1))

    law = df[df.eval_name == "mmlu-professional-law"]
    law_rows = []
    for _, r in law.iterrows():
        law_rows.append({
            "sample_id": r["sample_id"],
            "correct_a": float(r[MODEL_A]),
            "correct_b": float(r[MODEL_B]),
            "response_a": join_turns(r[f"{MODEL_A}|model_response"]),
            "response_b": join_turns(r[f"{MODEL_B}|model_response"]),
        })
    (OUT / "mmlu_law_pair.json").write_text(json.dumps({
        "model_a": MODEL_A, "model_b": MODEL_B, "records": law_rows,
    }, indent=1))

    g = df[df.eval_name == "grade-school-math"]
    label_cols = [c for c in df.columns if "|" not in c and c not in ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to")]
    prov = {"n": int(len(g)), "models": {}}
    frac_any = (g[label_cols] % 1 != 0).any(axis=1)
    prov["rows_with_any_fractional_label"] = int(frac_any.sum())
    for m in label_cols:
        counts = g[m].value_counts().sort_index()
        prov["models"][m] = {
            "label_histogram": {str(k): int(v) for k, v in counts.items()},
            "fractional_rows": int((g[m] % 1 != 0).sum()),
            "stored_responses_per_row": 1,
        }
    (OUT / "gsm8k_label_provenance.json").write_text(json.dumps(prov, indent=1))

    print(f"mtbench: {len(mt_rows)} rows; mmlu-professional-law: {len(law_rows)} rows; "
          f"gsm8k fractional rows: {prov['rows_with_any_fractional_label']}/{prov['n']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
