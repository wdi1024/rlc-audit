#!/usr/bin/env python3
"""Screen non-safety construct audits for proxy/semantic mismatch candidates.

This is an experiment-prep script, not a paper-claim generator. It reuses the
cached HotpotQA and GSM8K traces plus correctness labels, then scans multiple
observable proxy labels and score spans under the RLC-Audit contract:

  score representation -> surface proxy disagreement
  score representation -> semantic correctness disagreement

The goal is to identify candidate non-refusal rows worth promoting to full
experiments. A row is a candidate when proxy AUC is much higher than semantic
AUC under low proxy/semantic agreement.

Outputs:
  analysis_results/non_safety_proxy_mismatch_screen.json
  analysis_results/non_safety_proxy_mismatch_screen.md
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "analysis_results"
OUT_JSON = OUT / "non_safety_proxy_mismatch_screen.json"
OUT_MD = OUT / "non_safety_proxy_mismatch_screen.md"
ROUTE_BUDGET = 50


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def norm(text: str | None) -> str:
    return (text or "").strip()


def lower(text: str | None) -> str:
    return norm(text).lower()


def has_answer_text(trace: str, gold: str) -> bool:
    trace_l = lower(trace)
    gold_l = lower(gold)
    if not trace_l or not gold_l:
        return False
    return gold_l in trace_l


def has_answer_number(trace: str, gold: str) -> bool:
    trace_l = lower(trace).replace(",", "")
    gold_s = norm(gold).replace(",", "")
    if not trace_l or not gold_s:
        return False
    gold_s = gold_s.rstrip("0").rstrip(".") or gold_s
    return re.search(r"(?<![\d.])" + re.escape(gold_s) + r"(?![\d.])", trace_l) is not None


def has_answer(task: str, trace: str, gold: str) -> bool:
    if task == "gsm8k":
        return has_answer_number(trace, gold)
    return has_answer_text(trace, gold)


ANSWER_MARKER_RE = re.compile(
    r"\b(final answer|answer is|therefore|thus|so the answer|final output)\b",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"\b(unknown|unclear|not enough|not specified|cannot determine|can't determine|not sure|ambiguous)\b",
    re.IGNORECASE,
)
PASSAGE_RE = re.compile(r"\b(passage|section|paragraph|evidence|provided text)\b", re.IGNORECASE)


def is_incomplete(trace: str) -> bool:
    text = norm(trace)
    if not text:
        return True
    if "__ERROR__" in text:
        return True
    if len(text) < 40:
        return True
    return text[-1] not in ".!?)]}\"'"


def side_proxy_functions(task: str, traces_a: list[str], traces_b: list[str], golds: list[str]):
    all_lengths = np.array([len(t) for t in traces_a + traces_b], dtype=float)
    long_threshold = float(np.median(all_lengths)) if len(all_lengths) else 0.0

    def gold_full(trace: str, gold: str) -> bool:
        return has_answer(task, trace, gold)

    def gold_prefix128(trace: str, gold: str) -> bool:
        return has_answer(task, trace[:128], gold)

    def gold_prefix256(trace: str, gold: str) -> bool:
        return has_answer(task, trace[:256], gold)

    def gold_suffix256(trace: str, gold: str) -> bool:
        return has_answer(task, trace[-256:], gold)

    return {
        "gold_in_full_trace": gold_full,
        "gold_in_prefix128": gold_prefix128,
        "gold_in_prefix256": gold_prefix256,
        "gold_in_suffix256": gold_suffix256,
        "explicit_answer_marker": lambda trace, gold: ANSWER_MARKER_RE.search(trace or "") is not None,
        "uncertainty_marker": lambda trace, gold: UNCERTAINTY_RE.search(trace or "") is not None,
        "passage_or_evidence_marker": lambda trace, gold: PASSAGE_RE.search(trace or "") is not None,
        "incomplete_surface": lambda trace, gold: is_incomplete(trace),
        "long_trace_surface": lambda trace, gold: len(trace or "") > long_threshold,
    }


def tfidf_distance(a_texts: list[str], b_texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(a_texts + b_texts)
    xa = vec.transform(a_texts)
    xb = vec.transform(b_texts)
    sims = np.array([float(cosine_similarity(a, b)[0, 0]) for a, b in zip(xa, xb)])
    return 1.0 - sims


def marker_set(text: str) -> set[str]:
    text_l = lower(text)
    markers = set()
    for name, pattern in {
        "answer": ANSWER_MARKER_RE,
        "uncertain": UNCERTAINTY_RE,
        "passage": PASSAGE_RE,
    }.items():
        if pattern.search(text_l):
            markers.add(name)
    if is_incomplete(text):
        markers.add("incomplete")
    return markers


def jaccard_distance_scores(a_texts: list[str], b_texts: list[str]) -> np.ndarray:
    scores = []
    for a, b in zip(a_texts, b_texts):
        sa = marker_set(a)
        sb = marker_set(b)
        union = sa | sb
        scores.append(0.0 if not union else 1.0 - (len(sa & sb) / len(union)))
    return np.asarray(scores, dtype=float)


def score_sets(traces_a: list[str], traces_b: list[str]) -> dict[str, np.ndarray]:
    prefix128_a = [t[:128] for t in traces_a]
    prefix128_b = [t[:128] for t in traces_b]
    prefix256_a = [t[:256] for t in traces_a]
    prefix256_b = [t[:256] for t in traces_b]
    suffix256_a = [t[-256:] for t in traces_a]
    suffix256_b = [t[-256:] for t in traces_b]
    return {
        "tfidf_full_distance": tfidf_distance(traces_a, traces_b),
        "tfidf_prefix128_distance": tfidf_distance(prefix128_a, prefix128_b),
        "tfidf_prefix256_distance": tfidf_distance(prefix256_a, prefix256_b),
        "tfidf_suffix256_distance": tfidf_distance(suffix256_a, suffix256_b),
        "length_absdiff": np.asarray([abs(len(a) - len(b)) for a, b in zip(traces_a, traces_b)], dtype=float),
        "marker_jaccard_distance": jaccard_distance_scores(traces_a, traces_b),
    }


def metric_or_none(labels: np.ndarray, scores: np.ndarray, metric: Callable) -> float | None:
    if len(set(labels.astype(int).tolist())) < 2:
        return None
    try:
        return float(metric(labels, scores))
    except ValueError:
        return None


def status(delta_auc: float | None, delta_ap: float | None, kappa: float | None) -> str:
    if delta_auc is not None and kappa is not None and delta_auc >= 0.15 and kappa <= 0.20:
        return "MISMATCH_CANDIDATE"
    if (delta_auc is not None and delta_auc >= 0.10) or (delta_ap is not None and delta_ap >= 0.10):
        return "CAUTION"
    return "ALIGNED_OR_LOW_GAP"


def routed_semantic_count(labels: np.ndarray, scores: np.ndarray, budget: int) -> int:
    top = np.argsort(-scores)[: min(budget, len(scores))]
    return int(labels[top].sum())


def audit_rows(task_name: str, traces_a: list[str], traces_b: list[str], golds: list[str], sem_a: np.ndarray, sem_b: np.ndarray):
    y_sem = (sem_a != sem_b).astype(int)
    scores_by_name = score_sets(traces_a, traces_b)
    proxies = side_proxy_functions(task_name, traces_a, traces_b, golds)
    rows = []
    for proxy_name, proxy_fn in proxies.items():
        proxy_a = np.asarray([int(proxy_fn(t, g)) for t, g in zip(traces_a, golds)], dtype=int)
        proxy_b = np.asarray([int(proxy_fn(t, g)) for t, g in zip(traces_b, golds)], dtype=int)
        y_surface = (proxy_a != proxy_b).astype(int)
        if len(set(y_surface.tolist())) < 2:
            continue
        pair_kappa = float(cohen_kappa_score(y_surface, y_sem)) if len(set(y_sem.tolist())) >= 2 else None
        for score_name, score in scores_by_name.items():
            auc_surface = metric_or_none(y_surface, score, roc_auc_score)
            auc_sem = metric_or_none(y_sem, score, roc_auc_score)
            ap_surface = metric_or_none(y_surface, score, average_precision_score)
            ap_sem = metric_or_none(y_sem, score, average_precision_score)
            delta_auc = None if auc_surface is None or auc_sem is None else auc_surface - auc_sem
            delta_ap = None if ap_surface is None or ap_sem is None else ap_surface - ap_sem
            rows.append(
                {
                    "task": task_name,
                    "proxy": proxy_name,
                    "score": score_name,
                    "status": status(delta_auc, delta_ap, pair_kappa),
                    "n": len(y_sem),
                    "surface_positive": int(y_surface.sum()),
                    "semantic_positive": int(y_sem.sum()),
                    "auc_surface": auc_surface,
                    "auc_semantic": auc_sem,
                    "delta_auc": delta_auc,
                    "ap_surface": ap_surface,
                    "ap_semantic": ap_sem,
                    "delta_ap": delta_ap,
                    "kappa_surface_semantic": pair_kappa,
                    "semantic_at_budget": routed_semantic_count(y_sem, score, ROUTE_BUDGET),
                    "surface_at_budget": routed_semantic_count(y_surface, score, ROUTE_BUDGET),
                }
            )
    return rows


def load_task(task: str):
    if task == "hotpotqa":
        phase = "phase10_hotpot"
        a_name = "qwen3.5-2b"
        b_name = "gemma-4-e2b"
        meta = read_json(DATA / f"{phase}_meta.json")["prompts"]
        a_records = read_json(DATA / f"{phase}_traces_{a_name}.json")["records"]
        b_records = read_json(DATA / f"{phase}_traces_{b_name}.json")["records"]
        a_judge = read_json(DATA / f"{phase}_correctness_{a_name}_anthropic_claude-haiku-4-5-20251001.json")["records"]
        b_judge = read_json(DATA / f"{phase}_correctness_{b_name}_anthropic_claude-haiku-4-5-20251001.json")["records"]
    elif task == "gsm8k":
        phase = "phase11_gsm8k"
        a_name = "qwen3.5-2b"
        b_name = "gemma-4-e2b"
        meta = read_json(DATA / f"{phase}_meta.json")["prompts"]
        a_records = read_json(DATA / f"{phase}_traces_{a_name}.json")["records"]
        b_records = read_json(DATA / f"{phase}_traces_{b_name}.json")["records"]
        a_judge = read_json(DATA / f"{phase}_correctness_{a_name}_anthropic_claude-haiku-4-5-20251001.json")["records"]
        b_judge = read_json(DATA / f"{phase}_correctness_{b_name}_anthropic_claude-haiku-4-5-20251001.json")["records"]
    else:
        raise ValueError(task)

    meta_by_id = {r["id"]: r for r in meta}
    a_by_id = {r["id"]: r for r in a_records}
    b_by_id = {r["id"]: r for r in b_records}
    aj_by_id = {r["id"]: r for r in a_judge if r.get("correct_judge") is not None}
    bj_by_id = {r["id"]: r for r in b_judge if r.get("correct_judge") is not None}
    ids = sorted(set(meta_by_id) & set(a_by_id) & set(b_by_id) & set(aj_by_id) & set(bj_by_id))
    return {
        "ids": ids,
        "traces_a": [norm(a_by_id[i].get("trace")) for i in ids],
        "traces_b": [norm(b_by_id[i].get("trace")) for i in ids],
        "golds": [norm(meta_by_id[i].get("answer")) for i in ids],
        "sem_a": np.asarray([int(aj_by_id[i]["correct_judge"]) for i in ids], dtype=int),
        "sem_b": np.asarray([int(bj_by_id[i]["correct_judge"]) for i in ids], dtype=int),
    }


def fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.3f}"


def write_markdown(rows: list[dict]):
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if r["status"] == "MISMATCH_CANDIDATE" else 1 if r["status"] == "CAUTION" else 2,
            -(r["delta_auc"] if r["delta_auc"] is not None else -999),
        ),
    )
    lines = [
        "# Non-Safety Proxy Mismatch Screen",
        "",
        "This is an experiment-prep screen over cached HotpotQA and GSM8K correctness traces.",
        "Rows marked `MISMATCH_CANDIDATE` are candidates for a full non-safety RLC audit; they are not paper claims until inspected and validated.",
        "",
        "| Task | Status | Proxy | Score | Surf+ | Sem+ | AUC surf | AUC sem | Gap | Kappa | Sem@50 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranked[:40]:
        lines.append(
            f"| {r['task']} | {r['status']} | {r['proxy']} | {r['score']} | "
            f"{r['surface_positive']} | {r['semantic_positive']} | {fmt(r['auc_surface'])} | "
            f"{fmt(r['auc_semantic'])} | {fmt(r['delta_auc'])} | "
            f"{fmt(r['kappa_surface_semantic'])} | {r['semantic_at_budget']}/{ROUTE_BUDGET} |"
        )
    lines += [
        "",
        "Next steps:",
        "1. Inspect any `MISMATCH_CANDIDATE` rows for artifact mechanisms.",
        "2. If the mechanism is plausible, freeze the proxy and score definition before adding new data.",
        "3. Re-run with fresh labels or a held-out construct before promoting the row to the paper.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    all_rows = []
    for task in ["hotpotqa", "gsm8k"]:
        bundle = load_task(task)
        rows = audit_rows(
            task,
            bundle["traces_a"],
            bundle["traces_b"],
            bundle["golds"],
            bundle["sem_a"],
            bundle["sem_b"],
        )
        all_rows.extend(rows)
    OUT_JSON.write_text(json.dumps({"rows": all_rows}, indent=2))
    write_markdown(all_rows)
    ranked = sorted(all_rows, key=lambda r: -(r["delta_auc"] if r["delta_auc"] is not None else -999))
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_MD}")
    print("Top candidate rows:")
    for row in ranked[:8]:
        print(
            f"  {row['task']:8s} {row['status']:18s} "
            f"{row['proxy']:24s} {row['score']:24s} "
            f"gap={fmt(row['delta_auc'])} sem_auc={fmt(row['auc_semantic'])}"
        )


if __name__ == "__main__":
    main()
