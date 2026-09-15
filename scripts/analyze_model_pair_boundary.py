#!/usr/bin/env python3
"""Summarize existing model-pair boundary audits for RLC.

This script does not run generation or judge calls. It reads packaged analysis
JSON files and writes a compact table that separates three claims:

1. the primary Qwen/Gemma raw-prefix mismatch,
2. controls where the mismatch weakens or disappears, and
3. the 8B/9B pair where prefix-50 clears but raw-full mismatches.

The result is meant as a scope/boundary artifact, not a large-model sweep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis_results"
OUT = RESULTS / "model_pair_boundary_summary.json"


def load_json(name: str) -> Any:
    with (RESULTS / name).open() as f:
        return json.load(f)


def r3(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def add_gap(row: dict[str, Any]) -> dict[str, Any]:
    surf = row.get("surface_auc")
    sem = row.get("semantic_auc")
    row["auc_gap_surface_minus_semantic"] = r3(surf - sem) if surf is not None and sem is not None else None
    return row


def status_for(surface_auc: float | None, semantic_auc: float | None) -> str:
    if surface_auc is None:
        return "NO_SURFACE_TARGET"
    if semantic_auc is None:
        return "INCOMPLETE"
    if surface_auc - semantic_auc >= 0.30:
        return "MISMATCH"
    return "BOUNDARY"


def main() -> None:
    rlc = load_json("rlc_audit_report.json")
    same_family = load_json("same_family_control.json")
    phase12 = load_json("phase12_e2_1b_disagreement_report.json")
    phase14 = load_json("phase14_clean_xstest450_advbench100_8b_regeneration_report.json")
    phase14_full = load_json("phase14_raw_full_mismatch_summary.json")
    three_pairs = load_json("three_pairs_xstest_report.json")

    primary = next(row for row in rlc["rows"] if row["short"] == "xstest_p3")
    qwen_qwen_k50 = same_family["pairs"][0]["rows"][0]

    phase14_prefix = next(row for row in phase14["score_tables"] if row["score"] == "raw_prefix50")
    p14_surface = next(label for label in phase14_prefix["labels"] if "kw_" in label["label"])
    p14_semantic = next(label for label in phase14_prefix["labels"] if "judge_" in label["label"])

    rows = [
        add_gap(
            {
                "pair": "Qwen3.5-2B / Gemma-4-E2B-it",
                "family_relation": "cross-family heterogeneous",
                "setting": "XSTest 450",
                "score_span": "raw prefix-50 TF-IDF",
                "surface_label": "prefix keyword disagreement",
                "semantic_label": "LLM-judge refusal disagreement",
                "n": primary["n"],
                "surface_positive": primary["surface_positive_n"],
                "semantic_positive": primary["semantic_positive_n"],
                "surface_auc": r3(primary["auc_surface"]),
                "semantic_auc": r3(primary["auc_semantic"]),
                "status": "MISMATCH",
                "interpretation": "Primary RLC case: opening-template surface evidence drives the proxy more than the semantic construct.",
            }
        ),
        add_gap(
            {
                "pair": "Qwen3.5-2B seed1 / Qwen3.5-2B seed2",
                "family_relation": "same-family control",
                "setting": "XSTest 450",
                "score_span": "raw prefix-50 TF-IDF",
                "surface_label": "prefix keyword disagreement",
                "semantic_label": "LLM-judge refusal disagreement",
                "n": same_family["pairs"][0]["n"],
                "surface_positive": qwen_qwen_k50["n_dis_kw_at_k"],
                "semantic_positive": qwen_qwen_k50["n_dis_judge"],
                "surface_auc": r3(qwen_qwen_k50["auc_kw_at_k"]),
                "semantic_auc": r3(qwen_qwen_k50["auc_judge"]),
                "status": status_for(qwen_qwen_k50["auc_kw_at_k"], qwen_qwen_k50["auc_judge"]),
                "interpretation": "Same-family control has no prefix keyword positives at k=50, so the primary surface target largely disappears.",
            }
        ),
        add_gap(
            {
                "pair": "Llama-3.2-1B / Qwen3-1.7B",
                "family_relation": "small heterogeneous",
                "setting": "XSTest 450",
                "score_span": "raw prefix-50 TF-IDF",
                "surface_label": "prefix keyword disagreement",
                "semantic_label": "LLM-judge refusal disagreement",
                "n": phase12["n"],
                "surface_positive": phase12["n_disagree_keyword"],
                "semantic_positive": phase12["n_disagree_judge"],
                "surface_auc": r3(phase12["auc_keyword_first50"]),
                "semantic_auc": r3(phase12["auc_judge_first50"]),
                "status": status_for(phase12["auc_keyword_first50"], phase12["auc_judge_first50"]),
                "interpretation": "Heterogeneous pair without the primary surface-over-semantic prefix split; useful boundary evidence.",
            }
        ),
        add_gap(
            {
                "pair": "Qwen3.5-9B / Llama-3.1-8B-Instruct",
                "family_relation": "larger heterogeneous",
                "setting": "XSTest 450 + AdvBench 100",
                "score_span": "raw prefix-50 TF-IDF",
                "surface_label": "prefix keyword disagreement",
                "semantic_label": "LLM-judge refusal disagreement",
                "n": phase14["n_common"],
                "surface_positive": p14_surface["n_pos"],
                "semantic_positive": p14_semantic["n_pos"],
                "surface_auc": r3(p14_surface["auc"]),
                "semantic_auc": r3(p14_semantic["auc"]),
                "status": status_for(p14_surface["auc"], p14_semantic["auc"]),
                "interpretation": "The larger pair clears the prefix-50 contract, showing that heterogeneity alone is not sufficient for RLC.",
            }
        ),
        add_gap(
            {
                "pair": "Qwen3.5-9B / Llama-3.1-8B-Instruct",
                "family_relation": "larger heterogeneous",
                "setting": "XSTest 450 + AdvBench 100",
                "score_span": "raw full-trace TF-IDF",
                "surface_label": "full-trace keyword disagreement",
                "semantic_label": "LLM-judge refusal disagreement",
                "n": phase14_full["n"],
                "surface_positive": phase14_full["surface_positive"],
                "semantic_positive": phase14_full["semantic_positive"],
                "surface_auc": r3(phase14_full["auc_surface"]),
                "semantic_auc": r3(phase14_full["auc_semantic"]),
                "status": phase14_full["status"],
                "semantic_routed_at_55": phase14_full["routed_composition"]["top_55"]["semantic_routed"],
                "interpretation": "The same larger pair mismatches under raw-full scoring, so RLC is a score-label-span contract property.",
            }
        ),
    ]

    secondary = []
    for pair, report in three_pairs.items():
        cells = report["cells"]
        secondary.append(
            {
                "pair": pair,
                "setting": "XSTest 450, 512-token full traces",
                "n": report["n"],
                "keyword_disagreement": report["kw_disagree"],
                "semantic_disagreement": report["judge_disagree"],
                "tfidf_keyword_auc": r3(cells["TF-IDF_x_keyword"]["auc"]),
                "tfidf_semantic_auc": r3(cells["TF-IDF_x_judge"]["auc"]),
                "sentence_keyword_auc": r3(cells["Sentence-L12_x_keyword"]["auc"]),
                "sentence_semantic_auc": r3(cells["Sentence-L12_x_judge"]["auc"]),
                "interpretation": "Secondary full-trace scope check; not used as the raw-prefix RLC headline.",
            }
        )

    summary = {
        "takeaway": (
            "Existing packaged results support a pair/span-dependent RLC claim. "
            "They do not support the stronger claim that every heterogeneous pair mismatches; "
            "instead, mismatch appears when the score span and proxy label share artifact-bearing evidence."
        ),
        "recommended_claim": (
            "RLC risk increases when heterogeneous model openings or templates enter both the score representation "
            "and the validation proxy; the audit must therefore be checked pair by pair and span by span."
        ),
        "main_rows": rows,
        "secondary_full_trace_checks": secondary,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(summary["recommended_claim"])


if __name__ == "__main__":
    main()
