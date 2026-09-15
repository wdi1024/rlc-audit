#!/usr/bin/env python3
"""Analyze completed human annotations.

Inputs:
  results/disagree_routing/human_annotation/task.csv (with both annotators' labels filled in)
  results/disagree_routing/human_annotation/keys.csv (LLM judge + kw labels per case)

Outputs:
  results/disagree_routing/human_annotation/analysis_report.json
  results/disagree_routing/human_annotation/disagreements_for_adjudication.csv

Steps:
  1. Load both annotators' labels (from columns refusal_*_human_A and refusal_*_human_B)
  2. Compute Cohen's κ between A and B (inter-rater reliability)
  3. Identify A vs B disagreements → write CSV for adjudication
  4. After adjudication (CSV updated with `final_*` columns), compute κ vs LLM judge
  5. Re-run AUC paradox on human-disagreement labels (where A=B; or use final after adjudication)
"""
import csv
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

ANN_DIR = "results/disagree_routing/human_annotation"


def parse_bool(s):
    if s is None:
        return None
    s = str(s).strip().upper()
    if s in {"TRUE", "T", "1", "YES", "Y"}:
        return True
    if s in {"FALSE", "F", "0", "NO", "N"}:
        return False
    return None


def cohen_kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a, b)


def load_task(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_keys(path):
    keys = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            keys[r["id"]] = r
    return keys


def main():
    task_path = f"{ANN_DIR}/task.csv"
    keys_path = f"{ANN_DIR}/keys.csv"
    if not os.path.exists(task_path):
        print(f"ERROR: {task_path} not found. Run build_human_annotation_task.py first.")
        sys.exit(1)
    task = load_task(task_path)
    keys = load_keys(keys_path)

    print(f"Loaded {len(task)} task rows, {len(keys)} keys.")

    # Parse labels
    parsed = []
    for r in task:
        rec = {"id": r["id"], "source": r["source"]}
        for col in ["refusal_qwen_human_A", "refusal_gemma_human_A",
                    "refusal_qwen_human_B", "refusal_gemma_human_B",
                    "refusal_qwen_final", "refusal_gemma_final"]:
            rec[col] = parse_bool(r.get(col))
        rec["judge_qwen"] = parse_bool(keys[r["id"]].get("judge_qwen"))
        rec["judge_gemma"] = parse_bool(keys[r["id"]].get("judge_gemma"))
        parsed.append(rec)

    # === Coverage ===
    n_total = len(parsed)
    has_A = sum(1 for r in parsed if r["refusal_qwen_human_A"] is not None
                and r["refusal_gemma_human_A"] is not None)
    has_B = sum(1 for r in parsed if r["refusal_qwen_human_B"] is not None
                and r["refusal_gemma_human_B"] is not None)
    print(f"\nCoverage: {has_A}/{n_total} fully labeled by A, {has_B}/{n_total} by B")

    if has_A < 2 or has_B < 2:
        print("\nWaiting for annotations. Re-run after both annotators complete.")
        return

    # === Inter-rater reliability ===
    print("\n=== Inter-rater reliability (Cohen's κ between annotators A and B) ===")
    pairs_qwen = [(r["refusal_qwen_human_A"], r["refusal_qwen_human_B"]) for r in parsed
                  if r["refusal_qwen_human_A"] is not None and r["refusal_qwen_human_B"] is not None]
    pairs_gemma = [(r["refusal_gemma_human_A"], r["refusal_gemma_human_B"]) for r in parsed
                   if r["refusal_gemma_human_A"] is not None and r["refusal_gemma_human_B"] is not None]
    if pairs_qwen:
        ka_q = cohen_kappa([p[0] for p in pairs_qwen], [p[1] for p in pairs_qwen])
        agree_q = np.mean([p[0] == p[1] for p in pairs_qwen])
        print(f"  Qwen labels:  κ = {ka_q:.3f},  exact agree = {agree_q*100:.1f}%  (n={len(pairs_qwen)})")
    if pairs_gemma:
        ka_g = cohen_kappa([p[0] for p in pairs_gemma], [p[1] for p in pairs_gemma])
        agree_g = np.mean([p[0] == p[1] for p in pairs_gemma])
        print(f"  Gemma labels: κ = {ka_g:.3f},  exact agree = {agree_g*100:.1f}%  (n={len(pairs_gemma)})")

    # === Disagreements for adjudication ===
    disagreements = [r for r in parsed if (
        (r["refusal_qwen_human_A"] is not None and r["refusal_qwen_human_B"] is not None
         and r["refusal_qwen_human_A"] != r["refusal_qwen_human_B"])
        or
        (r["refusal_gemma_human_A"] is not None and r["refusal_gemma_human_B"] is not None
         and r["refusal_gemma_human_A"] != r["refusal_gemma_human_B"])
    )]
    print(f"\n  A vs B disagreements: {len(disagreements)}/{n_total}")

    if disagreements:
        adj_path = f"{ANN_DIR}/disagreements_for_adjudication.csv"
        with open(adj_path, "w", newline="", encoding="utf-8") as f:
            fields = ["id", "source",
                      "refusal_qwen_human_A", "refusal_qwen_human_B", "refusal_qwen_final",
                      "refusal_gemma_human_A", "refusal_gemma_human_B", "refusal_gemma_final",
                      "adjudication_notes"]
            w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            w.writeheader()
            for r in disagreements:
                w.writerow({
                    "id": r["id"], "source": r["source"],
                    "refusal_qwen_human_A": r["refusal_qwen_human_A"],
                    "refusal_qwen_human_B": r["refusal_qwen_human_B"],
                    "refusal_qwen_final": "",
                    "refusal_gemma_human_A": r["refusal_gemma_human_A"],
                    "refusal_gemma_human_B": r["refusal_gemma_human_B"],
                    "refusal_gemma_final": "",
                    "adjudication_notes": "",
                })
        print(f"  wrote {adj_path} for adjudicator review")

    # === Final labels: A=B → A; else require final from adjudicator ===
    final_q = []
    final_g = []
    judge_q_arr = []
    judge_g_arr = []
    final_set_q, final_set_g = 0, 0
    needs_adjud_q, needs_adjud_g = 0, 0
    for r in parsed:
        # Qwen
        if r["refusal_qwen_human_A"] is not None and r["refusal_qwen_human_B"] is not None:
            if r["refusal_qwen_human_A"] == r["refusal_qwen_human_B"]:
                final_q.append(r["refusal_qwen_human_A"])
                judge_q_arr.append(r["judge_qwen"])
                final_set_q += 1
            elif r["refusal_qwen_final"] is not None:
                final_q.append(r["refusal_qwen_final"])
                judge_q_arr.append(r["judge_qwen"])
                final_set_q += 1
            else:
                needs_adjud_q += 1
        # Gemma
        if r["refusal_gemma_human_A"] is not None and r["refusal_gemma_human_B"] is not None:
            if r["refusal_gemma_human_A"] == r["refusal_gemma_human_B"]:
                final_g.append(r["refusal_gemma_human_A"])
                judge_g_arr.append(r["judge_gemma"])
                final_set_g += 1
            elif r["refusal_gemma_final"] is not None:
                final_g.append(r["refusal_gemma_final"])
                judge_g_arr.append(r["judge_gemma"])
                final_set_g += 1
            else:
                needs_adjud_g += 1

    print(f"\n  Final-label resolution: Qwen {final_set_q}/{n_total} resolved "
          f"({needs_adjud_q} need adjudication)")
    print(f"  Final-label resolution: Gemma {final_set_g}/{n_total} resolved "
          f"({needs_adjud_g} need adjudication)")

    # === Human (final) vs Judge κ ===
    print(f"\n=== Human (final) vs LLM Judge (Claude Haiku 4.5) ===")
    if final_q and all(j is not None for j in judge_q_arr):
        kj_q = cohen_kappa(final_q, judge_q_arr)
        agree_j_q = np.mean([f == j for f, j in zip(final_q, judge_q_arr)])
        print(f"  Qwen:  κ = {kj_q:.3f},  exact agree = {agree_j_q*100:.1f}%  (n={len(final_q)})")
    if final_g and all(j is not None for j in judge_g_arr):
        kj_g = cohen_kappa(final_g, judge_g_arr)
        agree_j_g = np.mean([f == j for f, j in zip(final_g, judge_g_arr)])
        print(f"  Gemma: κ = {kj_g:.3f},  exact agree = {agree_j_g*100:.1f}%  (n={len(final_g)})")

    # === Disagreement-rate comparison: human vs judge ===
    # Build per-id pairs
    pair_data = {}
    for r in parsed:
        if r["refusal_qwen_human_A"] is not None and r["refusal_qwen_human_B"] is not None:
            if r["refusal_qwen_human_A"] == r["refusal_qwen_human_B"]:
                fq = r["refusal_qwen_human_A"]
            else:
                fq = r["refusal_qwen_final"]
        else:
            fq = None
        if r["refusal_gemma_human_A"] is not None and r["refusal_gemma_human_B"] is not None:
            if r["refusal_gemma_human_A"] == r["refusal_gemma_human_B"]:
                fg = r["refusal_gemma_human_A"]
            else:
                fg = r["refusal_gemma_final"]
        else:
            fg = None
        if fq is not None and fg is not None:
            pair_data[r["id"]] = {"human_q": fq, "human_g": fg,
                                  "judge_q": r["judge_qwen"], "judge_g": r["judge_gemma"]}

    if pair_data:
        n = len(pair_data)
        human_dis = sum(1 for v in pair_data.values() if v["human_q"] != v["human_g"])
        judge_dis = sum(1 for v in pair_data.values()
                        if v["judge_q"] is not None and v["judge_g"] is not None
                        and v["judge_q"] != v["judge_g"])
        print(f"\n=== Disagreement rates (n={n} fully resolved cases) ===")
        print(f"  Human-disagreement: {human_dis} ({human_dis/n*100:.1f}%)")
        print(f"  Judge-disagreement: {judge_dis} ({judge_dis/n*100:.1f}%)")

        # κ between human-disagreement and judge-disagreement labels
        ids = list(pair_data.keys())
        h_dis_arr = [int(pair_data[i]["human_q"] != pair_data[i]["human_g"]) for i in ids]
        j_dis_arr = []
        h_dis_paired = []
        for i in ids:
            v = pair_data[i]
            if v["judge_q"] is not None and v["judge_g"] is not None:
                j_dis_arr.append(int(v["judge_q"] != v["judge_g"]))
                h_dis_paired.append(int(v["human_q"] != v["human_g"]))
        if j_dis_arr:
            kk = cohen_kappa(h_dis_paired, j_dis_arr)
            print(f"  κ(human-dis, judge-dis) = {kk:.3f}  (n={len(j_dis_arr)})")

    # === Save report ===
    report = {
        "n_total": n_total,
        "n_resolved_qwen": final_set_q,
        "n_resolved_gemma": final_set_g,
        "kappa_AB_qwen": float(ka_q) if pairs_qwen else None,
        "kappa_AB_gemma": float(ka_g) if pairs_gemma else None,
        "n_disagreements_for_adjudication": len(disagreements),
    }
    if final_q and all(j is not None for j in judge_q_arr):
        report["kappa_human_vs_judge_qwen"] = float(kj_q)
    if final_g and all(j is not None for j in judge_g_arr):
        report["kappa_human_vs_judge_gemma"] = float(kj_g)

    save = f"{ANN_DIR}/analysis_report.json"
    with open(save, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
