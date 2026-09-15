#!/usr/bin/env python3
"""Create human-inspection tables for non-safety RLC mismatch candidates.

The screening script identifies candidate proxy/semantic splits. This script
turns a frozen candidate row into inspection artifacts for the routed top-B
cases, so a human can decide whether the score is really selecting an
artifact-bearing cue rather than the claimed semantic construct.

Default candidates:
  gsm8k: incomplete_surface proxy + marker_jaccard_distance score
  hotpotqa: long_trace_surface proxy + length_absdiff score

Outputs per candidate:
  analysis_results/non_safety_inspection/*.csv
  analysis_results/non_safety_inspection/*.jsonl
  analysis_results/non_safety_inspection/*.md
  analysis_results/non_safety_inspection/*_full_traces.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from screen_non_safety_proxy_mismatch import (
    ANSWER_MARKER_RE,
    DATA,
    OUT,
    PASSAGE_RE,
    UNCERTAINTY_RE,
    fmt,
    has_answer,
    is_incomplete,
    load_task,
    marker_set,
    read_json,
    score_sets,
    side_proxy_functions,
)


DEFAULT_CANDIDATES = [
    "gsm8k:incomplete_surface:marker_jaccard_distance",
    "hotpotqa:long_trace_surface:length_absdiff",
]

MODEL_A = "qwen3.5-2b"
MODEL_B = "gemma-4-e2b"


def compact(text: str | None, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def parse_candidate(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Candidate must be task:proxy:score, got {spec!r}")
    return parts[0], parts[1], parts[2]


def phase_for_task(task: str) -> str:
    if task == "hotpotqa":
        return "phase10_hotpot"
    if task == "gsm8k":
        return "phase11_gsm8k"
    raise ValueError(task)


def load_raw_maps(task: str):
    phase = phase_for_task(task)
    meta = read_json(DATA / f"{phase}_meta.json")["prompts"]
    a_records = read_json(DATA / f"{phase}_traces_{MODEL_A}.json")["records"]
    b_records = read_json(DATA / f"{phase}_traces_{MODEL_B}.json")["records"]
    a_judge = read_json(
        DATA / f"{phase}_correctness_{MODEL_A}_anthropic_claude-haiku-4-5-20251001.json"
    )["records"]
    b_judge = read_json(
        DATA / f"{phase}_correctness_{MODEL_B}_anthropic_claude-haiku-4-5-20251001.json"
    )["records"]
    return {
        "meta": {r["id"]: r for r in meta},
        "a_trace": {r["id"]: r for r in a_records},
        "b_trace": {r["id"]: r for r in b_records},
        "a_judge": {r["id"]: r for r in a_judge},
        "b_judge": {r["id"]: r for r in b_judge},
    }


def explain_proxy(task: str, proxy: str, trace: str, gold: str, long_threshold: float) -> str:
    if proxy == "gold_in_full_trace":
        return f"gold_in_full={int(has_answer(task, trace, gold))}"
    if proxy == "gold_in_prefix128":
        return f"gold_in_prefix128={int(has_answer(task, trace[:128], gold))}"
    if proxy == "gold_in_prefix256":
        return f"gold_in_prefix256={int(has_answer(task, trace[:256], gold))}"
    if proxy == "gold_in_suffix256":
        return f"gold_in_suffix256={int(has_answer(task, trace[-256:], gold))}"
    if proxy == "explicit_answer_marker":
        return f"answer_marker={int(ANSWER_MARKER_RE.search(trace or '') is not None)}"
    if proxy == "uncertainty_marker":
        return f"uncertainty_marker={int(UNCERTAINTY_RE.search(trace or '') is not None)}"
    if proxy == "passage_or_evidence_marker":
        return f"evidence_marker={int(PASSAGE_RE.search(trace or '') is not None)}"
    if proxy == "incomplete_surface":
        end = (trace or "")[-1:] or "<empty>"
        return f"incomplete={int(is_incomplete(trace))}; len={len(trace or '')}; end={end!r}"
    if proxy == "long_trace_surface":
        return f"long={int(len(trace or '') > long_threshold)}; len={len(trace or '')}; median={long_threshold:.1f}"
    return "unknown_proxy"


def artifact_hint(proxy: str, proxy_a: int, proxy_b: int, reason_a: str, reason_b: str, trace_a: str, trace_b: str) -> str:
    cue_relation = "differs" if proxy_a != proxy_b else "same"
    if proxy == "incomplete_surface":
        return f"incomplete cue {cue_relation}: A[{reason_a}] B[{reason_b}]"
    if proxy == "long_trace_surface":
        return f"length cue {cue_relation}: A[{reason_a}] B[{reason_b}]"
    if "marker" in proxy:
        return f"marker cue {cue_relation}: A[{reason_a}] B[{reason_b}]"
    if proxy.startswith("gold_in"):
        return f"gold-string cue {cue_relation}: A[{reason_a}] B[{reason_b}]"
    markers_a = ",".join(sorted(marker_set(trace_a))) or "none"
    markers_b = ",".join(sorted(marker_set(trace_b))) or "none"
    return f"markers A={markers_a}; B={markers_b}"


def screen_metric(task: str, proxy: str, score: str) -> dict:
    path = OUT / "non_safety_proxy_mismatch_screen.json"
    if not path.exists():
        return {}
    rows = read_json(path).get("rows", [])
    for row in rows:
        if row.get("task") == task and row.get("proxy") == proxy and row.get("score") == score:
            return row
    return {}


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [k for k in rows[0].keys() if not k.endswith("_full")] if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, task: str, proxy: str, score: str, rows: list[dict], metric: dict, budget: int, snippets: int) -> None:
    route_counts = Counter((int(r["surface_disagreement"]), int(r["semantic_disagreement"])) for r in rows)
    lines = [
        f"# Inspection: {task} / {proxy} / {score}",
        "",
        "This table is for manual inspection. It should not be cited as a paper claim until the candidate is frozen and rerun on held-out or fresh data.",
        "",
        "## Candidate Metrics",
        "",
        f"- status: `{metric.get('status', 'NA')}`",
        f"- n: {metric.get('n', 'NA')}",
        f"- surface positives: {metric.get('surface_positive', 'NA')}",
        f"- semantic positives: {metric.get('semantic_positive', 'NA')}",
        f"- surface AUC: {fmt(metric.get('auc_surface')) if metric else 'NA'}",
        f"- semantic AUC: {fmt(metric.get('auc_semantic')) if metric else 'NA'}",
        f"- gap: {fmt(metric.get('delta_auc')) if metric else 'NA'}",
        f"- kappa: {fmt(metric.get('kappa_surface_semantic')) if metric else 'NA'}",
        f"- semantic@{budget}: {sum(int(r['semantic_disagreement']) for r in rows)}/{budget}",
        "",
        "## Routed Composition",
        "",
        "| surface disagreement | semantic disagreement | count |",
        "|---:|---:|---:|",
    ]
    for key in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        lines.append(f"| {key[0]} | {key[1]} | {route_counts.get(key, 0)} |")

    lines += [
        "",
        "## Top Routed Cases",
        "",
        "| Rank | ID | Score | Surface | Semantic | Proxy A/B | Correct A/B | Artifact Cue | Question | Gold |",
        "|---:|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['rank']} | `{r['id']}` | {float(r['score']):.4f} | "
            f"{r['surface_disagreement']} | {r['semantic_disagreement']} | "
            f"{r['proxy_a']}/{r['proxy_b']} | {r['correct_a']}/{r['correct_b']} | "
            f"{r['artifact_hint']} | {r['question_short']} | {r['gold']} |"
        )

    lines += [
        "",
        "## Manual Decision Rubric",
        "",
        "Mark each case as one of:",
        "",
        "- `artifact_only`: score/proxy is driven by the surface cue and semantic labels agree.",
        "- `real_semantic`: routed case is a genuine correctness disagreement.",
        "- `label_noise`: judge correctness label or gold matching is suspect.",
        "- `unclear`: needs full-trace inspection.",
        "",
        f"## Trace Snippets: First {min(snippets, len(rows))} Routed Cases",
        "",
    ]
    for r in rows[:snippets]:
        lines += [
            f"### Rank {r['rank']} - {r['id']}",
            "",
            f"- question: {r['question_short']}",
            f"- gold: {r['gold']}",
            f"- score: {float(r['score']):.4f}; surface={r['surface_disagreement']}; semantic={r['semantic_disagreement']}",
            f"- proxy A: {r['proxy_reason_a']}",
            f"- proxy B: {r['proxy_reason_b']}",
            f"- correct A/B: {r['correct_a']}/{r['correct_b']}",
            f"- judge rationale A: {r['judge_rationale_a']}",
            f"- judge rationale B: {r['judge_rationale_b']}",
            "",
            f"Qwen snippet: {r['trace_a_snippet']}",
            "",
            f"Qwen tail: {r['trace_a_tail']}",
            "",
            f"Gemma snippet: {r['trace_b_snippet']}",
            "",
            f"Gemma tail: {r['trace_b_tail']}",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def write_full_trace_markdown(path: Path, task: str, proxy: str, score: str, rows: list[dict]) -> None:
    lines = [
        f"# Full Traces: {task} / {proxy} / {score}",
        "",
        "Use this file when the summary table is too short to see the final answer.",
        "If the full trace itself ends mid-sentence or before an answer, that is part of the candidate artifact and should not be treated as a display bug.",
        "",
    ]
    for r in rows:
        lines += [
            f"## Rank {r['rank']} - {r['id']}",
            "",
            f"- score: {float(r['score']):.4f}",
            f"- surface disagreement: {r['surface_disagreement']}",
            f"- semantic disagreement: {r['semantic_disagreement']}",
            f"- proxy A/B: {r['proxy_a']}/{r['proxy_b']}",
            f"- correct A/B: {r['correct_a']}/{r['correct_b']}",
            f"- proxy reason A: {r['proxy_reason_a']}",
            f"- proxy reason B: {r['proxy_reason_b']}",
            f"- gold: {r['gold']}",
            f"- question: {r['question_short']}",
            f"- judge rationale A: {r['judge_rationale_a']}",
            f"- judge rationale B: {r['judge_rationale_b']}",
            "",
            "### Qwen Full Trace",
            "",
            "```text",
            r["trace_a_full"],
            "```",
            "",
            "### Gemma Full Trace",
            "",
            "```text",
            r["trace_b_full"],
            "```",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def inspect_candidate(task: str, proxy: str, score_name: str, budget: int, out_dir: Path, snippets: int) -> list[Path]:
    bundle = load_task(task)
    raw = load_raw_maps(task)
    ids = bundle["ids"]
    traces_a = bundle["traces_a"]
    traces_b = bundle["traces_b"]
    golds = bundle["golds"]
    sem_a = bundle["sem_a"]
    sem_b = bundle["sem_b"]
    y_sem = (sem_a != sem_b).astype(int)

    proxies = side_proxy_functions(task, traces_a, traces_b, golds)
    if proxy not in proxies:
        raise ValueError(f"Unknown proxy {proxy!r}; choices: {sorted(proxies)}")
    scores_by_name = score_sets(traces_a, traces_b)
    if score_name not in scores_by_name:
        raise ValueError(f"Unknown score {score_name!r}; choices: {sorted(scores_by_name)}")

    proxy_fn = proxies[proxy]
    proxy_a = np.asarray([int(proxy_fn(t, g)) for t, g in zip(traces_a, golds)], dtype=int)
    proxy_b = np.asarray([int(proxy_fn(t, g)) for t, g in zip(traces_b, golds)], dtype=int)
    y_surface = (proxy_a != proxy_b).astype(int)
    scores = scores_by_name[score_name]

    long_threshold = float(np.median(np.array([len(t) for t in traces_a + traces_b], dtype=float)))
    top_idx = np.argsort(-scores)[: min(budget, len(scores))]
    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        case_id = ids[int(idx)]
        meta = raw["meta"][case_id]
        trace_a = traces_a[int(idx)]
        trace_b = traces_b[int(idx)]
        reason_a = explain_proxy(task, proxy, trace_a, golds[int(idx)], long_threshold)
        reason_b = explain_proxy(task, proxy, trace_b, golds[int(idx)], long_threshold)
        rows.append(
            {
                "rank": rank,
                "id": case_id,
                "score": float(scores[int(idx)]),
                "surface_disagreement": int(y_surface[int(idx)]),
                "semantic_disagreement": int(y_sem[int(idx)]),
                "proxy_a": int(proxy_a[int(idx)]),
                "proxy_b": int(proxy_b[int(idx)]),
                "correct_a": int(sem_a[int(idx)]),
                "correct_b": int(sem_b[int(idx)]),
                "proxy_reason_a": reason_a,
                "proxy_reason_b": reason_b,
                "artifact_hint": artifact_hint(
                    proxy,
                    int(proxy_a[int(idx)]),
                    int(proxy_b[int(idx)]),
                    reason_a,
                    reason_b,
                    trace_a,
                    trace_b,
                ),
                "manual_decision": "",
                "manual_notes": "",
                "question": meta.get("question") or meta.get("prompt") or "",
                "question_short": compact(meta.get("question") or meta.get("prompt") or "", 120),
                "gold": golds[int(idx)],
                "trace_a_snippet": compact(trace_a, 360),
                "trace_b_snippet": compact(trace_b, 360),
                "trace_a_tail": compact(trace_a[-1200:], 1200),
                "trace_b_tail": compact(trace_b[-1200:], 1200),
                "trace_a_full": trace_a,
                "trace_b_full": trace_b,
                "judge_rationale_a": compact(raw["a_judge"].get(case_id, {}).get("rationale"), 300),
                "judge_rationale_b": compact(raw["b_judge"].get(case_id, {}).get("rationale"), 300),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{task}__{proxy}__{score_name}__top{budget}"
    csv_path = out_dir / f"{stem}.csv"
    jsonl_path = out_dir / f"{stem}.jsonl"
    md_path = out_dir / f"{stem}.md"
    full_md_path = out_dir / f"{stem}_full_traces.md"
    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)
    write_markdown(md_path, task, proxy, score_name, rows, screen_metric(task, proxy, score_name), budget, snippets)
    write_full_trace_markdown(full_md_path, task, proxy, score_name, rows)
    return [csv_path, jsonl_path, md_path, full_md_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate as task:proxy:score. Can be repeated.",
    )
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=OUT / "non_safety_inspection")
    parser.add_argument("--snippets", type=int, default=12)
    args = parser.parse_args()

    candidates = args.candidate or DEFAULT_CANDIDATES
    written = []
    for spec in candidates:
        task, proxy, score = parse_candidate(spec)
        paths = inspect_candidate(task, proxy, score, args.budget, args.out_dir, args.snippets)
        written.extend(paths)
        print(f"Wrote inspection files for {spec}:")
        for path in paths:
            print(f"  {path}")
    print(f"Done. Generated {len(written)} files.")


if __name__ == "__main__":
    main()
