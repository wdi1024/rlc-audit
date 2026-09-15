#!/usr/bin/env python3
"""Reusable RLC-Audit CLI.

This command turns precomputed paired outputs, surface labels, semantic labels,
and scores into an RLC-Audit report. It is intentionally data-format-light:
inputs may be JSONL, a JSON array, or CSV, as long as each row has a stable
example id.

The score is interpreted as "higher means more disagreement" by default. Use
--score-direction lower when the supplied score is a similarity or confidence
value where lower scores should route first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score


PAIR_DISAGREEMENT_FIELDS = (
    "disagreement",
    "surface_disagreement",
    "semantic_disagreement",
    "label",
    "y",
)
SIDE_A_FIELDS = ("label_a", "a_label", "surface_a", "semantic_a", "refusal_a", "model_a_label")
SIDE_B_FIELDS = ("label_b", "b_label", "surface_b", "semantic_b", "refusal_b", "model_b_label")
SCORE_FIELDS = ("score", "disagreement_score", "route_score", "distance", "similarity")


@dataclass(frozen=True)
class LabelBundle:
    pair: np.ndarray
    side_a: np.ndarray | None
    side_b: np.ndarray | None
    source_fields: dict[str, str]


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("records", "rows", "items", "data"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"{path} must be a JSON array or contain records/rows/items/data")
    if suffix == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input extension for {path}; use .jsonl, .json, or .csv")


def index_by_id(rows: Iterable[dict[str, Any]], id_field: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if id_field not in row:
            raise ValueError(f"Missing id field {id_field!r} in row: {row}")
        item_id = str(row[id_field])
        if item_id in indexed:
            raise ValueError(f"Duplicate id {item_id!r}")
        indexed[item_id] = row
    return indexed


def first_present(row: dict[str, Any], fields: Iterable[str]) -> str | None:
    for field in fields:
        if field in row and row[field] not in (None, ""):
            return field
    return None


def parse_bool(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            raise ValueError("NaN is not a valid label")
        return int(float(value) != 0.0)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "refusal", "refuse", "refuses", "positive"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "compliance", "comply", "complies", "negative"}:
        return 0
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None and not math.isnan(numeric):
        return int(numeric != 0.0)
    raise ValueError(f"Cannot parse binary label value {value!r}")


def load_labels(rows: dict[str, dict[str, Any]], ids: list[str], explicit_pair: str | None) -> LabelBundle:
    sample = rows[ids[0]]
    pair_field = explicit_pair or first_present(sample, PAIR_DISAGREEMENT_FIELDS)
    side_a_field = first_present(sample, SIDE_A_FIELDS)
    side_b_field = first_present(sample, SIDE_B_FIELDS)

    if pair_field is not None:
        pair = np.array([parse_bool(rows[item_id][pair_field]) for item_id in ids], dtype=int)
        return LabelBundle(pair=pair, side_a=None, side_b=None, source_fields={"pair": pair_field})

    if side_a_field is None or side_b_field is None:
        raise ValueError(
            "Label rows must contain either a pair-level disagreement field or both side labels. "
            "Accepted pair fields include disagreement/surface_disagreement/semantic_disagreement; "
            "accepted side fields include label_a/label_b, surface_a/surface_b, semantic_a/semantic_b."
        )

    side_a = np.array([parse_bool(rows[item_id][side_a_field]) for item_id in ids], dtype=int)
    side_b = np.array([parse_bool(rows[item_id][side_b_field]) for item_id in ids], dtype=int)
    return LabelBundle(
        pair=(side_a != side_b).astype(int),
        side_a=side_a,
        side_b=side_b,
        source_fields={"side_a": side_a_field, "side_b": side_b_field},
    )


def load_scores(rows: dict[str, dict[str, Any]], ids: list[str], score_field: str | None) -> tuple[np.ndarray, str]:
    sample = rows[ids[0]]
    field = score_field or first_present(sample, SCORE_FIELDS)
    if field is None:
        raise ValueError(f"Score rows must contain one of {SCORE_FIELDS} or pass --score-field")
    return np.array([float(rows[item_id][field]) for item_id in ids], dtype=float), field


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if int(y.sum()) == 0 or int(y.sum()) == len(y):
        return None
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    if int(y.sum()) == 0:
        return None
    return float(average_precision_score(y, score))


def precision_at(y: np.ndarray, score: np.ndarray, k: int) -> float | None:
    if k <= 0 or len(y) == 0:
        return None
    k = min(k, len(y))
    order = np.argsort(-score)
    return float(np.mean(y[order[:k]]))


def kappa(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    if len(set(a.tolist())) <= 1 and len(set(b.tolist())) <= 1:
        return None
    return float(cohen_kappa_score(a, b))


def metric_bundle(y: np.ndarray, score: np.ndarray, budgets: list[int]) -> dict[str, Any]:
    n_pos = int(y.sum())
    prevalence = float(n_pos / max(len(y), 1))
    return {
        "n": int(len(y)),
        "n_pos": n_pos,
        "prevalence": prevalence,
        "auc": safe_auc(y, score),
        "average_precision": safe_ap(y, score),
        "ap_lift_over_prevalence": (
            None if prevalence <= 0 or safe_ap(y, score) is None else float(safe_ap(y, score) / prevalence)
        ),
        "precision_at_positive_count": precision_at(y, score, n_pos) if n_pos else None,
        "precision_at_budget": {str(k): precision_at(y, score, k) for k in budgets},
    }


def resolve_budgets(spec: str, n: int, default_positive_count: int) -> list[int]:
    out: list[int] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        value = float(part)
        if 0 < value <= 1:
            k = max(1, int(round(value * n)))
        else:
            k = int(value)
        out.append(min(max(k, 1), n))
    if default_positive_count > 0:
        out.append(min(default_positive_count, n))
    return sorted(set(out))


def composition_for_order(
    ids: list[str],
    order: np.ndarray,
    raw_score: np.ndarray,
    metric_score: np.ndarray,
    surface: LabelBundle,
    semantic: LabelBundle,
    k: int,
    name: str,
) -> dict[str, Any]:
    selected = order[: min(k, len(ids))]
    sem_dis = semantic.pair[selected]
    surf_dis = surface.pair[selected]
    row: dict[str, Any] = {
        "name": name,
        "routed_n": int(len(selected)),
        "semantic_disagreement_n": int(sem_dis.sum()),
        "semantic_disagreement_rate": float(sem_dis.mean()) if len(selected) else None,
        "surface_disagreement_n": int(surf_dis.sum()),
        "surface_disagreement_rate": float(surf_dis.mean()) if len(selected) else None,
        "mean_raw_score": float(raw_score[selected].mean()) if len(selected) else None,
        "mean_metric_score": float(metric_score[selected].mean()) if len(selected) else None,
    }
    if semantic.side_a is not None and semantic.side_b is not None:
        a = semantic.side_a[selected]
        b = semantic.side_b[selected]
        both_refuse = (a == 1) & (b == 1)
        both_comply = (a == 0) & (b == 0)
        row.update(
            {
                "both_refuse_n": int(both_refuse.sum()),
                "both_refuse_rate": float(both_refuse.mean()) if len(selected) else None,
                "both_comply_n": int(both_comply.sum()),
                "both_comply_rate": float(both_comply.mean()) if len(selected) else None,
            }
        )
    return row


def screening_status(
    auc_surface: float | None,
    auc_semantic: float | None,
    ap_surface: float | None,
    ap_semantic: float | None,
    mean_kappa: float | None,
    fail_auc_gap: float,
    warn_auc_gap: float,
    warn_ap_gap: float,
    low_kappa: float,
) -> dict[str, Any]:
    auc_gap = None if auc_surface is None or auc_semantic is None else auc_surface - auc_semantic
    ap_gap = None if ap_surface is None or ap_semantic is None else ap_surface - ap_semantic
    low_construct_agreement = mean_kappa is not None and mean_kappa <= low_kappa
    if auc_gap is not None and auc_gap >= fail_auc_gap and low_construct_agreement:
        status = "FAIL"
        reason = "surface AUC exceeds semantic AUC under low surface/semantic construct agreement"
    elif (auc_gap is not None and auc_gap >= warn_auc_gap) or (ap_gap is not None and ap_gap >= warn_ap_gap):
        status = "WARN"
        reason = "surface/semantic performance gap warrants mechanism inspection"
    else:
        status = "PASS"
        reason = "no large surface-over-semantic gap detected by this diagnostic"
    return {
        "status": status,
        "reason": reason,
        "auc_gap_surface_minus_semantic": auc_gap,
        "ap_gap_surface_minus_semantic": ap_gap,
        "thresholds": {
            "fail_auc_gap": fail_auc_gap,
            "warn_auc_gap": warn_auc_gap,
            "warn_ap_gap": warn_ap_gap,
            "low_kappa": low_kappa,
        },
        "caveat": "FAIL/WARN/PASS is a descriptive screen, not a theoretical guarantee or deployment certificate.",
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# RLC-Audit Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Screening status: **{report['screening']['status']}** - {report['screening']['reason']}.",
        "",
        "| Target | n pos | prevalence | AUC | AP | P@#pos |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Surface label | {metrics['surface']['n_pos']} | {fmt(metrics['surface']['prevalence'])} | "
            f"{fmt(metrics['surface']['auc'])} | {fmt(metrics['surface']['average_precision'])} | "
            f"{fmt(metrics['surface']['precision_at_positive_count'])} |"
        ),
        (
            f"| Semantic label | {metrics['semantic']['n_pos']} | {fmt(metrics['semantic']['prevalence'])} | "
            f"{fmt(metrics['semantic']['auc'])} | {fmt(metrics['semantic']['average_precision'])} | "
            f"{fmt(metrics['semantic']['precision_at_positive_count'])} |"
        ),
        "",
        f"Surface-vs-semantic pair-label kappa: {fmt(report['construct_agreement']['pair_kappa'])}.",
    ]
    if report["construct_agreement"].get("mean_side_kappa") is not None:
        lines.append(f"Mean side-label kappa: {fmt(report['construct_agreement']['mean_side_kappa'])}.")
    lines.extend(
        [
            "",
            "## Routed Composition",
            "",
            "| Budget | routed | semantic dis | surface dis | both refuse | both comply |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["routed_composition"]:
        lines.append(
            f"| {row['name']} | {row['routed_n']} | {row['semantic_disagreement_n']} | "
            f"{row['surface_disagreement_n']} | {fmt(row.get('both_refuse_n'))} | "
            f"{fmt(row.get('both_comply_n'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    paired = index_by_id(read_rows(args.paired_outputs), args.id_field)
    surface_rows = index_by_id(read_rows(args.surface_labels), args.id_field)
    semantic_rows = index_by_id(read_rows(args.semantic_labels), args.id_field)
    score_rows = index_by_id(read_rows(args.scores), args.id_field)
    ids = sorted(set(paired) & set(surface_rows) & set(semantic_rows) & set(score_rows))
    if not ids:
        raise ValueError("No common ids across paired outputs, labels, and scores")

    surface = load_labels(surface_rows, ids, args.surface_pair_field)
    semantic = load_labels(semantic_rows, ids, args.semantic_pair_field)
    raw_score, score_field = load_scores(score_rows, ids, args.score_field)
    metric_score = -raw_score if args.score_direction == "lower" else raw_score
    budgets = resolve_budgets(args.budgets, len(ids), int(semantic.pair.sum()))

    surface_metrics = metric_bundle(surface.pair, metric_score, budgets)
    semantic_metrics = metric_bundle(semantic.pair, metric_score, budgets)
    pair_kappa = kappa(surface.pair, semantic.pair)
    side_kappas = [kappa(surface.side_a, semantic.side_a), kappa(surface.side_b, semantic.side_b)]
    valid_side_kappas = [x for x in side_kappas if x is not None]
    mean_side_kappa = float(np.mean(valid_side_kappas)) if valid_side_kappas else None

    order = np.argsort(-metric_score)
    routed = [
        composition_for_order(ids, order, raw_score, metric_score, surface, semantic, k, f"top_{k}")
        for k in budgets
    ]

    report = {
        "audit": "RLC-Audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "paired_outputs": str(args.paired_outputs),
            "surface_labels": str(args.surface_labels),
            "semantic_labels": str(args.semantic_labels),
            "scores": str(args.scores),
            "id_field": args.id_field,
            "score_field": score_field,
            "score_direction": args.score_direction,
            "surface_label_fields": surface.source_fields,
            "semantic_label_fields": semantic.source_fields,
        },
        "n_common": len(ids),
        "metrics": {"surface": surface_metrics, "semantic": semantic_metrics},
        "construct_agreement": {
            "pair_kappa": pair_kappa,
            "side_a_kappa": side_kappas[0],
            "side_b_kappa": side_kappas[1],
            "mean_side_kappa": mean_side_kappa,
        },
        "screening": screening_status(
            surface_metrics["auc"],
            semantic_metrics["auc"],
            surface_metrics["average_precision"],
            semantic_metrics["average_precision"],
            mean_side_kappa if mean_side_kappa is not None else pair_kappa,
            args.fail_auc_gap,
            args.warn_auc_gap,
            args.warn_ap_gap,
            args.low_kappa,
        ),
        "routed_composition": routed,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLC-Audit on reusable input files.")
    parser.add_argument("--paired-outputs", type=Path, required=True)
    parser.add_argument("--surface-labels", type=Path, required=True)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--score-field")
    parser.add_argument("--surface-pair-field")
    parser.add_argument("--semantic-pair-field")
    parser.add_argument("--score-direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--budgets", default="10,0.05", help="Comma-separated counts or fractions plus #semantic positives.")
    parser.add_argument("--fail-auc-gap", type=float, default=0.15)
    parser.add_argument("--warn-auc-gap", type=float, default=0.10)
    parser.add_argument("--warn-ap-gap", type=float, default=0.10)
    parser.add_argument("--low-kappa", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown_report(report))
    print(f"{report['screening']['status']}: {report['screening']['reason']}")
    print(f"Saved {args.out_json}")
    if args.out_md:
        print(f"Saved {args.out_md}")


if __name__ == "__main__":
    main()
