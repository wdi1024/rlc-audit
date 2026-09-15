#!/usr/bin/env python3
"""Analyze the current ICLR human-annotation package.

The annotation package in `human_annotation_iclr/` is blind: annotators should
complete `prefix_annotation_template.csv` before seeing
`full_annotation_template.csv`, while automatic labels and routing scores live
in `keys.csv`. This script accepts one or two completed combined annotation
CSVs with the same schema as `annotation_template.csv`; it also accepts the
prefix-only or full-output schemas for partial status checks.

Typical use:

  python scripts/analyze_iclr_human_annotation.py \
    --ann-a human_annotation_iclr/annotator_a.csv \
    --ann-b human_annotation_iclr/annotator_b.csv

Strict two-pass use:

  python scripts/analyze_iclr_human_annotation.py \
    --prefix-ann-a human_annotation_iclr/prefix_annotator_a.csv \
    --full-ann-a human_annotation_iclr/full_annotator_a.csv \
    --prefix-ann-b human_annotation_iclr/prefix_annotator_b.csv \
    --full-ann-b human_annotation_iclr/full_annotator_b.csv

If two annotators disagree, the script writes
`human_annotation_iclr/disagreements_for_adjudication.csv`. After adjudication,
pass a CSV containing `pair_id` plus any needed final fields, such as
`prefix_model_a_final`, `prefix_model_b_final`, `full_model_a_final`,
`full_model_b_final`, `prefix_context_sufficient_final`,
`full_context_sufficient_final`, `prefix_pair_decision_final`,
`full_pair_decision_final`, and `prefix_opening_template_mismatch_final`, via
`--adjudicated`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "human_annotation_iclr"
PAPER_DIR = ROOT / "paper"

VALID_LABELS = {"REFUSAL", "COMPLIANCE", "UNCLEAR", "INSUFFICIENT_CONTEXT"}
VALID_PAIR_DECISIONS = {
    "BOTH_REFUSE",
    "BOTH_COMPLY",
    "DISAGREEMENT",
    "UNCLEAR",
    "INSUFFICIENT_CONTEXT",
}
VALID_TRIAGE = {"YES", "NO", "UNCLEAR"}

STANDARD_ANNOTATION_COLUMNS = [
    "pair_id",
    "prefix_context_sufficient",
    "prefix_pair_decision",
    "prefix_model_a_refusal",
    "prefix_model_b_refusal",
    "prefix_model_a_confidence",
    "prefix_model_b_confidence",
    "prefix_opening_template_mismatch",
    "prefix_notes",
    "full_context_sufficient",
    "full_pair_decision",
    "full_model_a_refusal",
    "full_model_b_refusal",
    "full_model_a_confidence",
    "full_model_b_confidence",
    "full_notes",
]


def norm_label(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    label = str(value).strip().upper()
    if not label:
        return None
    if label not in VALID_LABELS:
        raise ValueError(f"Unknown refusal label: {value!r}")
    return label


def norm_triage(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    label = str(value).strip().upper()
    if not label:
        return None
    if label not in VALID_TRIAGE:
        raise ValueError(f"Unknown YES/NO/UNCLEAR value: {value!r}")
    return label


def norm_pair_decision(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    label = str(value).strip().upper()
    if not label:
        return None
    if label not in VALID_PAIR_DECISIONS:
        raise ValueError(f"Unknown pair decision: {value!r}")
    return label


def label_bool(label: str | None) -> bool | None:
    if label == "REFUSAL":
        return True
    if label == "COMPLIANCE":
        return False
    return None


def pair_from_labels(a: str | None, b: str | None, context: str | None) -> str | None:
    if a == "REFUSAL" and b == "REFUSAL":
        return "BOTH_REFUSE"
    if a == "COMPLIANCE" and b == "COMPLIANCE":
        return "BOTH_COMPLY"
    if {a, b} == {"REFUSAL", "COMPLIANCE"}:
        return "DISAGREEMENT"
    if context == "NO" or a == "INSUFFICIENT_CONTEXT" or b == "INSUFFICIENT_CONTEXT":
        return "INSUFFICIENT_CONTEXT"
    if a == "UNCLEAR" or b == "UNCLEAR" or context == "UNCLEAR":
        return "UNCLEAR"
    return None


def load_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"pair_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    # New two-pass schema is preferred. Legacy single-pass columns are accepted
    # so older partially filled sheets can still be inspected.
    alias_map = {
        "prefix_model_a_refusal": ["prefix_model_a_refusal", "model_a_refusal"],
        "prefix_model_b_refusal": ["prefix_model_b_refusal", "model_b_refusal"],
        "prefix_model_a_confidence": ["prefix_model_a_confidence", "model_a_confidence"],
        "prefix_model_b_confidence": ["prefix_model_b_confidence", "model_b_confidence"],
        "prefix_opening_template_mismatch": [
            "prefix_opening_template_mismatch",
            "opening_template_mismatch",
        ],
        "prefix_notes": ["prefix_notes", "notes"],
        "full_model_a_refusal": ["full_model_a_refusal"],
        "full_model_b_refusal": ["full_model_b_refusal"],
        "full_model_a_confidence": ["full_model_a_confidence"],
        "full_model_b_confidence": ["full_model_b_confidence"],
        "full_notes": ["full_notes"],
    }

    def col_value(target: str) -> pd.Series:
        if target == "prefix_context_sufficient":
            names = ["prefix_context_sufficient"]
        elif target == "prefix_pair_decision":
            names = ["prefix_pair_decision"]
        elif target == "full_context_sufficient":
            names = ["full_context_sufficient"]
        elif target == "full_pair_decision":
            names = ["full_pair_decision"]
        else:
            names = alias_map[target]
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series([None] * len(df), index=df.index)

    out = pd.DataFrame({"pair_id": df["pair_id"]})
    for col in STANDARD_ANNOTATION_COLUMNS[1:]:
        out[col] = col_value(col)

    out["prefix_context_sufficient"] = out["prefix_context_sufficient"].map(norm_triage)
    out["prefix_opening_template_mismatch"] = out["prefix_opening_template_mismatch"].map(norm_triage)
    out["prefix_model_a_refusal"] = out["prefix_model_a_refusal"].map(norm_label)
    out["prefix_model_b_refusal"] = out["prefix_model_b_refusal"].map(norm_label)
    out["prefix_pair_decision"] = out["prefix_pair_decision"].map(norm_pair_decision)
    out["full_context_sufficient"] = out["full_context_sufficient"].map(norm_triage)
    out["full_model_a_refusal"] = out["full_model_a_refusal"].map(norm_label)
    out["full_model_b_refusal"] = out["full_model_b_refusal"].map(norm_label)
    out["full_pair_decision"] = out["full_pair_decision"].map(norm_pair_decision)

    out["prefix_pair_decision"] = [
        pair_from_labels(a, b, c) if p is None else p
        for p, a, b, c in zip(
            out["prefix_pair_decision"],
            out["prefix_model_a_refusal"],
            out["prefix_model_b_refusal"],
            out["prefix_context_sufficient"],
        )
    ]
    out["full_pair_decision"] = [
        pair_from_labels(a, b, c) if p is None else p
        for p, a, b, c in zip(
            out["full_pair_decision"],
            out["full_model_a_refusal"],
            out["full_model_b_refusal"],
            out["full_context_sufficient"],
        )
    ]
    return out


def coalesce_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.where(left.map(lambda x: not is_missing(x)), right)


def load_annotation_inputs(
    combined_path: str | None,
    prefix_path: str | None,
    full_path: str | None,
) -> pd.DataFrame:
    if combined_path:
        return load_annotations(Path(combined_path))
    if not prefix_path and not full_path:
        raise ValueError("Provide --ann-a/--ann-b or split --prefix-ann-* / --full-ann-* files.")
    frames = []
    if prefix_path:
        frames.append(load_annotations(Path(prefix_path)))
    if full_path:
        frames.append(load_annotations(Path(full_path)))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="pair_id", how="outer", suffixes=("_left", "_right"))
        out = pd.DataFrame({"pair_id": merged["pair_id"]})
        for col in STANDARD_ANNOTATION_COLUMNS[1:]:
            left = merged.get(f"{col}_left", pd.Series([None] * len(merged), index=merged.index))
            right = merged.get(f"{col}_right", pd.Series([None] * len(merged), index=merged.index))
            out[col] = coalesce_series(left, right)
        merged = out
    return merged[STANDARD_ANNOTATION_COLUMNS]


def load_keys(annotation_dir: Path) -> pd.DataFrame:
    keys = pd.read_csv(annotation_dir / "keys.csv")
    bool_cols = [
        "judge_a_refusal",
        "judge_b_refusal",
        "semantic_disagreement_judge",
        "keyword_disagreement_prefix50",
    ]
    for col in bool_cols:
        keys[col] = keys[col].astype(str).str.lower().map({"true": True, "false": False})
    return keys


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return None
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) == 0 or y.sum() == 0:
        return None
    return float(average_precision_score(y, score))


def kappa_or_none(a: list[bool], b: list[bool]) -> float | None:
    if len(a) < 2 or (len(set(a)) <= 1 and len(set(b)) <= 1):
        return None
    return float(cohen_kappa_score(a, b))


def final_choice(first: Any, second: Any, adjudicated: Any) -> Any:
    if is_missing(second):
        return None if is_missing(first) else first
    if is_missing(first):
        return second
    if same_value(first, second):
        return first
    return adjudicated


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def same_value(first: Any, second: Any) -> bool:
    if is_missing(first) and is_missing(second):
        return True
    return first == second


def resolve_two_annotators(
    ann_a: pd.DataFrame,
    ann_b: pd.DataFrame | None,
    adjudicated: pd.DataFrame | None,
    keys: pd.DataFrame,
    annotation_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    value_fields = [
        "prefix_context_sufficient",
        "prefix_pair_decision",
        "prefix_model_a_refusal",
        "prefix_model_b_refusal",
        "prefix_model_a_confidence",
        "prefix_model_b_confidence",
        "prefix_opening_template_mismatch",
        "prefix_notes",
        "full_context_sufficient",
        "full_pair_decision",
        "full_model_a_refusal",
        "full_model_b_refusal",
        "full_model_a_confidence",
        "full_model_b_confidence",
        "full_notes",
    ]

    def with_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return df.rename(columns={field: f"{field}_{suffix}" for field in value_fields})

    a = with_suffix(ann_a, "A")
    merged = keys.merge(a, on="pair_id", how="left")
    if ann_b is not None:
        b = with_suffix(ann_b, "B")
        merged = merged.merge(b, on="pair_id", how="left")
    if adjudicated is not None:
        merged = merged.merge(adjudicated, on="pair_id", how="left")

    summary: dict[str, Any] = {}
    disagreements = []
    finals: dict[str, list[Any]] = {field: [] for field in value_fields}

    final_aliases = {
        "prefix_model_a_refusal": ["prefix_model_a_final"],
        "prefix_model_b_refusal": ["prefix_model_b_final"],
        "full_model_a_refusal": ["full_model_a_final", "model_a_final"],
        "full_model_b_refusal": ["full_model_b_final", "model_b_final"],
        "prefix_context_sufficient": ["prefix_context_sufficient_final"],
        "full_context_sufficient": ["full_context_sufficient_final"],
        "prefix_pair_decision": ["prefix_pair_decision_final"],
        "full_pair_decision": ["full_pair_decision_final"],
        "prefix_opening_template_mismatch": ["prefix_opening_template_mismatch_final"],
    }

    def adjudicated_value(row: pd.Series, field: str) -> Any:
        for name in final_aliases.get(field, [f"{field}_final"]):
            if name in row:
                value = row.get(name)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    return value
        return None

    for _, row in merged.iterrows():
        if ann_b is None:
            for field in value_fields:
                finals[field].append(row.get(f"{field}_A"))
        else:
            any_disagreement = False
            for field in value_fields:
                first = row.get(f"{field}_A")
                second = row.get(f"{field}_B")
                if not same_value(first, second):
                    any_disagreement = True
                finals[field].append(final_choice(first, second, adjudicated_value(row, field)))
            if any_disagreement:
                disagreements.append(row)

    merged["prefix_context_sufficient_final"] = [norm_triage(x) for x in finals["prefix_context_sufficient"]]
    merged["prefix_opening_template_mismatch_final"] = [
        norm_triage(x) for x in finals["prefix_opening_template_mismatch"]
    ]
    merged["prefix_model_a_final_label"] = [norm_label(x) for x in finals["prefix_model_a_refusal"]]
    merged["prefix_model_b_final_label"] = [norm_label(x) for x in finals["prefix_model_b_refusal"]]
    merged["prefix_pair_decision_final"] = [
        norm_pair_decision(x) for x in finals["prefix_pair_decision"]
    ]
    merged["prefix_pair_decision_final"] = [
        pair_from_labels(a, b, c) if p is None else p
        for p, a, b, c in zip(
            merged["prefix_pair_decision_final"],
            merged["prefix_model_a_final_label"],
            merged["prefix_model_b_final_label"],
            merged["prefix_context_sufficient_final"],
        )
    ]
    merged["prefix_model_a_final_bool"] = merged["prefix_model_a_final_label"].map(label_bool)
    merged["prefix_model_b_final_bool"] = merged["prefix_model_b_final_label"].map(label_bool)

    merged["full_context_sufficient_final"] = [norm_triage(x) for x in finals["full_context_sufficient"]]
    merged["full_model_a_final_label"] = [norm_label(x) for x in finals["full_model_a_refusal"]]
    merged["full_model_b_final_label"] = [norm_label(x) for x in finals["full_model_b_refusal"]]
    merged["full_pair_decision_final"] = [norm_pair_decision(x) for x in finals["full_pair_decision"]]
    merged["full_pair_decision_final"] = [
        pair_from_labels(a, b, c) if p is None else p
        for p, a, b, c in zip(
            merged["full_pair_decision_final"],
            merged["full_model_a_final_label"],
            merged["full_model_b_final_label"],
            merged["full_context_sufficient_final"],
        )
    ]
    merged["full_model_a_final_bool"] = merged["full_model_a_final_label"].map(label_bool)
    merged["full_model_b_final_bool"] = merged["full_model_b_final_label"].map(label_bool)

    if ann_b is not None:
        for pass_name in ["prefix", "full"]:
            for model in ["a", "b"]:
                left = [label_bool(x) for x in merged[f"{pass_name}_model_{model}_refusal_A"]]
                right = [label_bool(x) for x in merged[f"{pass_name}_model_{model}_refusal_B"]]
                pairs = [(x, y) for x, y in zip(left, right) if x is not None and y is not None]
                summary[f"annotator_kappa_{pass_name}_model_{model}"] = kappa_or_none(
                    [x for x, _ in pairs], [y for _, y in pairs]
                )
                summary[f"annotator_exact_agreement_{pass_name}_model_{model}"] = (
                    float(np.mean([x == y for x, y in pairs])) if pairs else None
                )

        for field in [
            "prefix_context_sufficient",
            "prefix_pair_decision",
            "prefix_opening_template_mismatch",
            "full_pair_decision",
        ]:
            pairs = [
                (x, y)
                for x, y in zip(merged[f"{field}_A"], merged[f"{field}_B"])
                if x is not None and y is not None
            ]
            summary[f"annotator_exact_agreement_{field}"] = (
                float(np.mean([x == y for x, y in pairs])) if pairs else None
            )

        if disagreements:
            wanted = [
                "pair_id",
                "phase",
                "dataset",
                "sample_stratum",
                "prefix_context_sufficient_A",
                "prefix_context_sufficient_B",
                "prefix_pair_decision_A",
                "prefix_pair_decision_B",
                "prefix_model_a_refusal_A",
                "prefix_model_a_refusal_B",
                "prefix_model_b_refusal_A",
                "prefix_model_b_refusal_B",
                "prefix_opening_template_mismatch_A",
                "prefix_opening_template_mismatch_B",
                "full_pair_decision_A",
                "full_pair_decision_B",
                "full_model_a_refusal_A",
                "full_model_a_refusal_B",
                "full_model_b_refusal_A",
                "full_model_b_refusal_B",
            ]
            out = pd.DataFrame(disagreements)[[c for c in wanted if c in merged.columns]].copy()
            out["prefix_context_sufficient_final"] = ""
            out["prefix_pair_decision_final"] = ""
            out["prefix_model_a_final"] = ""
            out["prefix_model_b_final"] = ""
            out["prefix_opening_template_mismatch_final"] = ""
            out["full_context_sufficient_final"] = ""
            out["full_pair_decision_final"] = ""
            out["full_model_a_final"] = ""
            out["full_model_b_final"] = ""
            out["adjudication_notes"] = ""
            out.to_csv(annotation_dir / "disagreements_for_adjudication.csv", index=False)
        else:
            stale = annotation_dir / "disagreements_for_adjudication.csv"
            if stale.exists():
                stale.unlink()

    return merged, summary


def triage_count(series: pd.Series, value: str) -> int:
    return int((series == value).sum())


def base_context_report(merged: pd.DataFrame) -> dict[str, Any]:
    prefix = merged["prefix_context_sufficient_final"]
    opening = merged["prefix_opening_template_mismatch_final"]
    prefix_pair = merged["prefix_pair_decision_final"]
    full_context = merged["full_context_sufficient_final"]
    full_pair = merged["full_pair_decision_final"]
    inconsistent_forced_labels = (
        (prefix == "NO")
        & merged["prefix_model_a_final_bool"].map(lambda x: isinstance(x, bool))
        & merged["prefix_model_b_final_bool"].map(lambda x: isinstance(x, bool))
    )
    full_resolved = (
        merged["full_model_a_final_bool"].map(lambda x: isinstance(x, bool))
        & merged["full_model_b_final_bool"].map(lambda x: isinstance(x, bool))
        & (full_context == "YES")
    )
    prefix_insufficient = (prefix == "NO") | (prefix_pair == "INSUFFICIENT_CONTEXT")
    prefix_insufficient_full_resolved = prefix_insufficient & full_resolved
    full_disagreement = merged["full_model_a_final_bool"] != merged["full_model_b_final_bool"]
    return {
        "prefix_context_sufficient_yes_n": triage_count(prefix, "YES"),
        "prefix_context_sufficient_no_n": triage_count(prefix, "NO"),
        "prefix_context_sufficient_unclear_n": triage_count(prefix, "UNCLEAR"),
        "prefix_pair_both_refuse_n": triage_count(prefix_pair, "BOTH_REFUSE"),
        "prefix_pair_both_comply_n": triage_count(prefix_pair, "BOTH_COMPLY"),
        "prefix_pair_disagreement_n": triage_count(prefix_pair, "DISAGREEMENT"),
        "prefix_pair_insufficient_context_n": triage_count(prefix_pair, "INSUFFICIENT_CONTEXT"),
        "prefix_pair_unclear_n": triage_count(prefix_pair, "UNCLEAR"),
        "opening_template_mismatch_yes_n": triage_count(opening, "YES"),
        "opening_template_mismatch_no_n": triage_count(opening, "NO"),
        "opening_template_mismatch_unclear_n": triage_count(opening, "UNCLEAR"),
        "context_label_inconsistency_n": int(inconsistent_forced_labels.sum()),
        "prefix_model_a_insufficient_context_n": int(
            (merged["prefix_model_a_final_label"] == "INSUFFICIENT_CONTEXT").sum()
        ),
        "prefix_model_b_insufficient_context_n": int(
            (merged["prefix_model_b_final_label"] == "INSUFFICIENT_CONTEXT").sum()
        ),
        "prefix_any_side_insufficient_context_n": int(
            (
                (merged["prefix_model_a_final_label"] == "INSUFFICIENT_CONTEXT")
                | (merged["prefix_model_b_final_label"] == "INSUFFICIENT_CONTEXT")
            ).sum()
        ),
        "full_context_sufficient_yes_n": triage_count(full_context, "YES"),
        "full_context_sufficient_no_n": triage_count(full_context, "NO"),
        "full_context_sufficient_unclear_n": triage_count(full_context, "UNCLEAR"),
        "full_pair_both_refuse_n": triage_count(full_pair, "BOTH_REFUSE"),
        "full_pair_both_comply_n": triage_count(full_pair, "BOTH_COMPLY"),
        "full_pair_disagreement_n": triage_count(full_pair, "DISAGREEMENT"),
        "full_pair_insufficient_context_n": triage_count(full_pair, "INSUFFICIENT_CONTEXT"),
        "full_pair_unclear_n": triage_count(full_pair, "UNCLEAR"),
        "full_model_a_insufficient_context_n": int(
            (merged["full_model_a_final_label"] == "INSUFFICIENT_CONTEXT").sum()
        ),
        "full_model_b_insufficient_context_n": int(
            (merged["full_model_b_final_label"] == "INSUFFICIENT_CONTEXT").sum()
        ),
        "prefix_insufficient_but_full_resolved_n": int(prefix_insufficient_full_resolved.sum()),
        "prefix_insufficient_but_full_agreement_n": int(
            (prefix_insufficient_full_resolved & ~full_disagreement).sum()
        ),
        "prefix_insufficient_but_full_disagreement_n": int(
            (prefix_insufficient_full_resolved & full_disagreement).sum()
        ),
    }


def analyze(merged: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    context_report = base_context_report(merged)
    prefix_resolved = merged[
        merged["prefix_model_a_final_bool"].map(lambda x: isinstance(x, bool))
        & merged["prefix_model_b_final_bool"].map(lambda x: isinstance(x, bool))
        & (merged["prefix_context_sufficient_final"] == "YES")
    ].copy()
    full_resolved = merged[
        merged["full_model_a_final_bool"].map(lambda x: isinstance(x, bool))
        & merged["full_model_b_final_bool"].map(lambda x: isinstance(x, bool))
        & (merged["full_context_sufficient_final"] == "YES")
    ].copy()
    if full_resolved.empty and prefix_resolved.empty:
        return {
            **summary,
            "status": "waiting_for_completed_annotations",
            "n_total": int(len(merged)),
            "n_prefix_resolved_pairs": 0,
            "n_full_resolved_pairs": 0,
            **context_report,
            "message": "No resolved human labels yet. Fill prefix/full annotation labels, then rerun.",
        }

    prefix_human_dis = np.array([])
    prefix_score = np.array([])
    prefix_kw_dis = np.array([])
    if not prefix_resolved.empty:
        prefix_a = prefix_resolved["prefix_model_a_final_bool"].astype(bool).to_numpy()
        prefix_b = prefix_resolved["prefix_model_b_final_bool"].astype(bool).to_numpy()
        prefix_human_dis = (prefix_a != prefix_b).astype(int)
        prefix_score = prefix_resolved["cosine_distance_prefix50"].astype(float).to_numpy()
        prefix_kw_dis = prefix_resolved["keyword_disagreement_prefix50"].astype(bool).astype(int).to_numpy()

    full_human_a = np.array([])
    full_human_b = np.array([])
    judge_a = np.array([])
    judge_b = np.array([])
    full_human_dis = np.array([])
    judge_dis = np.array([])
    full_score = np.array([])
    full_kw_dis = np.array([])
    if not full_resolved.empty:
        full_human_a = full_resolved["full_model_a_final_bool"].astype(bool).to_numpy()
        full_human_b = full_resolved["full_model_b_final_bool"].astype(bool).to_numpy()
        judge_a = full_resolved["judge_a_refusal"].astype(bool).to_numpy()
        judge_b = full_resolved["judge_b_refusal"].astype(bool).to_numpy()
        full_human_dis = (full_human_a != full_human_b).astype(int)
        judge_dis = (judge_a != judge_b).astype(int)
        full_score = full_resolved["cosine_distance_prefix50"].astype(float).to_numpy()
        full_kw_dis = full_resolved["keyword_disagreement_prefix50"].astype(bool).astype(int).to_numpy()

    report = {
        **summary,
        "n_total": int(len(merged)),
        "n_prefix_resolved_pairs": int(len(prefix_resolved)),
        "n_full_resolved_pairs": int(len(full_resolved)),
        **context_report,
        "prefix_human_disagreement_n": int(prefix_human_dis.sum()) if len(prefix_human_dis) else 0,
        "prefix_keyword_disagreement_n_on_resolved": int(prefix_kw_dis.sum()) if len(prefix_kw_dis) else 0,
        "auc_prefix50_vs_prefix_human_disagreement": safe_auc(prefix_human_dis, prefix_score),
        "ap_prefix50_vs_prefix_human_disagreement": safe_ap(prefix_human_dis, prefix_score),
        "auc_prefix50_vs_prefix_keyword_disagreement_resolved_subset": safe_auc(prefix_kw_dis, prefix_score),
        "ap_prefix50_vs_prefix_keyword_disagreement_resolved_subset": safe_ap(prefix_kw_dis, prefix_score),
        "full_human_disagreement_n": int(full_human_dis.sum()) if len(full_human_dis) else 0,
        "judge_disagreement_n_on_full_resolved": int(judge_dis.sum()) if len(judge_dis) else 0,
        "keyword_disagreement_n_on_full_resolved": int(full_kw_dis.sum()) if len(full_kw_dis) else 0,
        "full_human_vs_judge_kappa_model_a": kappa_or_none(full_human_a.tolist(), judge_a.tolist())
        if len(full_human_a)
        else None,
        "full_human_vs_judge_kappa_model_b": kappa_or_none(full_human_b.tolist(), judge_b.tolist())
        if len(full_human_b)
        else None,
        "full_human_dis_vs_judge_dis_kappa": kappa_or_none(
            full_human_dis.astype(bool).tolist(), judge_dis.astype(bool).tolist()
        )
        if len(full_human_dis)
        else None,
        "auc_prefix50_vs_full_human_disagreement": safe_auc(full_human_dis, full_score),
        "ap_prefix50_vs_full_human_disagreement": safe_ap(full_human_dis, full_score),
        "auc_prefix50_vs_judge_disagreement_full_resolved_subset": safe_auc(judge_dis, full_score),
        "ap_prefix50_vs_judge_disagreement_full_resolved_subset": safe_ap(judge_dis, full_score),
        "auc_prefix50_vs_keyword_disagreement_full_resolved_subset": safe_auc(full_kw_dis, full_score),
        "ap_prefix50_vs_keyword_disagreement_full_resolved_subset": safe_ap(full_kw_dis, full_score),
    }
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    def fmt(x: Any) -> str:
        return "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)

    lines = [
        "# ICLR Human Annotation Validation",
        "",
        "Generated by `scripts/analyze_iclr_human_annotation.py`.",
        "",
        f"Prefix-resolved pairs: {report.get('n_prefix_resolved_pairs', 0)}/{report['n_total']}",
        f"Full-output-resolved pairs: {report.get('n_full_resolved_pairs', 0)}/{report['n_total']}",
        "",
        "| Quantity | Value |",
        "|---|---:|",
    ]
    for key in [
        "annotator_kappa_prefix_model_a",
        "annotator_kappa_prefix_model_b",
        "annotator_kappa_full_model_a",
        "annotator_kappa_full_model_b",
        "full_human_vs_judge_kappa_model_a",
        "full_human_vs_judge_kappa_model_b",
        "full_human_dis_vs_judge_dis_kappa",
        "prefix_context_sufficient_yes_n",
        "prefix_context_sufficient_no_n",
        "prefix_context_sufficient_unclear_n",
        "prefix_pair_insufficient_context_n",
        "prefix_model_a_insufficient_context_n",
        "prefix_model_b_insufficient_context_n",
        "prefix_any_side_insufficient_context_n",
        "opening_template_mismatch_yes_n",
        "prefix_insufficient_but_full_resolved_n",
        "prefix_insufficient_but_full_agreement_n",
        "prefix_insufficient_but_full_disagreement_n",
        "full_context_sufficient_yes_n",
        "full_context_sufficient_no_n",
        "context_label_inconsistency_n",
        "prefix_human_disagreement_n",
        "full_human_disagreement_n",
        "judge_disagreement_n_on_full_resolved",
        "keyword_disagreement_n_on_full_resolved",
        "auc_prefix50_vs_prefix_human_disagreement",
        "ap_prefix50_vs_prefix_human_disagreement",
        "auc_prefix50_vs_full_human_disagreement",
        "ap_prefix50_vs_full_human_disagreement",
        "auc_prefix50_vs_judge_disagreement_full_resolved_subset",
        "auc_prefix50_vs_keyword_disagreement_full_resolved_subset",
    ]:
        if key in report:
            lines.append(f"| `{key}` | {fmt(report[key])} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--ann-a", default=None, help="Completed combined CSV from annotator A.")
    parser.add_argument("--ann-b", default=None, help="Completed combined CSV from annotator B.")
    parser.add_argument("--prefix-ann-a", default=None, help="Completed prefix-only CSV from annotator A.")
    parser.add_argument("--prefix-ann-b", default=None, help="Completed prefix-only CSV from annotator B.")
    parser.add_argument("--full-ann-a", default=None, help="Completed full-output CSV from annotator A.")
    parser.add_argument("--full-ann-b", default=None, help="Completed full-output CSV from annotator B.")
    parser.add_argument("--adjudicated", default=None, help="Optional CSV with pair_id, model_a_final, model_b_final.")
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir)
    keys = load_keys(annotation_dir)

    has_a = args.ann_a or args.prefix_ann_a or args.full_ann_a
    has_b = args.ann_b or args.prefix_ann_b or args.full_ann_b

    if not has_a:
        report = {
            "status": "waiting_for_annotations",
            "n_total": int(len(keys)),
            "required_columns": [
                "prefix_context_sufficient",
                "prefix_pair_decision",
                "prefix_model_a_refusal",
                "prefix_model_b_refusal",
                "prefix_model_a_confidence",
                "prefix_model_b_confidence",
                "prefix_opening_template_mismatch",
                "prefix_notes",
                "full_context_sufficient",
                "full_pair_decision",
                "full_model_a_refusal",
                "full_model_b_refusal",
                "full_model_a_confidence",
                "full_model_b_confidence",
                "full_notes",
            ],
            "legacy_required_columns": [
                "prefix_context_sufficient",
                "model_a_refusal",
                "model_b_refusal",
                "model_a_confidence",
                "model_b_confidence",
                "opening_template_mismatch",
                "notes",
            ],
            "label_values": sorted(VALID_LABELS),
            "pair_decision_values": sorted(VALID_PAIR_DECISIONS),
            "prefix_context_sufficient_values": sorted(VALID_TRIAGE),
        }
        out = annotation_dir / "analysis_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Waiting for completed annotation CSVs. Wrote package status to {out}")
        return

    ann_a = load_annotation_inputs(args.ann_a, args.prefix_ann_a, args.full_ann_a)
    ann_b = (
        load_annotation_inputs(args.ann_b, args.prefix_ann_b, args.full_ann_b)
        if has_b
        else None
    )
    adjudicated = pd.read_csv(args.adjudicated) if args.adjudicated else None
    if adjudicated is not None:
        for col in [
            "model_a_final",
            "model_b_final",
            "prefix_model_a_final",
            "prefix_model_b_final",
            "full_model_a_final",
            "full_model_b_final",
        ]:
            if col in adjudicated.columns:
                adjudicated[col] = adjudicated[col].map(norm_label)
        for col in [
            "prefix_context_sufficient_final",
            "full_context_sufficient_final",
            "prefix_opening_template_mismatch_final",
        ]:
            if col in adjudicated.columns:
                adjudicated[col] = adjudicated[col].map(norm_triage)
        for col in ["prefix_pair_decision_final", "full_pair_decision_final"]:
            if col in adjudicated.columns:
                adjudicated[col] = adjudicated[col].map(norm_pair_decision)

    merged, summary = resolve_two_annotators(ann_a, ann_b, adjudicated, keys, annotation_dir)
    report = analyze(merged, summary)
    out_json = annotation_dir / "analysis_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("status") == "waiting_for_completed_annotations":
        print(f"Waiting for completed annotation CSVs. Wrote package status to {out_json}")
        return
    out_md = PAPER_DIR / "appendix_iclr_human_annotation.md"
    write_markdown(report, out_md)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    if ann_b is not None and (annotation_dir / "disagreements_for_adjudication.csv").exists():
        print(f"Saved: {annotation_dir / 'disagreements_for_adjudication.csv'}")


if __name__ == "__main__":
    main()
