#!/usr/bin/env python3
"""Analyze disagreement signal: trace similarity vs label disagreement.

For Phase 1 GO/NO-GO decision:
- AUC > 0.65 (trace similarity predicts label disagreement) → GO
- Pearson |r| > 0.3 between similarity and disagreement → bonus signal
"""
import argparse
import json
import os
import statistics

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


def normalize_answer(s):
    import re, string
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def hotpot_correct(pred_text, gold):
    """Simple correctness: gold normalized form appears in pred normalized form."""
    if not pred_text or not gold:
        return False
    p = normalize_answer(pred_text)
    g = normalize_answer(gold)
    if not g:
        return False
    return g in p or p in g


def extract_predicted_answer(trace):
    """Try to extract the final answer from a trace.
    Heuristic: take last non-empty line, or after 'Answer:'.
    """
    if not trace:
        return ""
    for marker in ["Answer:", "answer:", "Final answer:", "ANSWER:"]:
        if marker in trace:
            return trace.rsplit(marker, 1)[1].strip().split("\n")[0].strip()
    lines = [ln.strip() for ln in trace.strip().split("\n") if ln.strip()]
    if lines:
        return lines[-1]
    return trace.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/disagree_routing/")
    parser.add_argument("--phase", default="phase1")
    parser.add_argument("--models", default="qwen3.5-2b,gemma-4-e2b")
    parser.add_argument("--auc_threshold", type=float, default=0.65)
    parser.add_argument("--pearson_threshold", type=float, default=0.3)
    args = parser.parse_args()

    models = args.models.split(",")
    assert len(models) == 2, "exactly 2 models supported for binary disagreement"

    # Load metadata
    with open(os.path.join(args.results_dir, f"{args.phase}_meta.json")) as f:
        meta = json.load(f)["prompts"]

    # Load traces per model
    traces = {}
    for m in models:
        with open(os.path.join(args.results_dir, f"{args.phase}_traces_{m}.json")) as f:
            d = json.load(f)
        traces[m] = {r["id"]: r for r in d["records"]}

    # Match prompts present in both
    common = set(traces[models[0]].keys()) & set(traces[models[1]].keys())
    common = [pid for pid in (m["id"] for m in meta) if pid in common]
    print(f"Common prompts in both models: {len(common)}")

    # Build tables
    rows = []
    for pid in common:
        m_meta = next((m for m in meta if m["id"] == pid), {})
        r0 = traces[models[0]][pid]
        r1 = traces[models[1]][pid]
        row = {
            "id": pid,
            "task": m_meta.get("task"),
            "prompt": m_meta.get("prompt", "")[:200],
            "trace_0": r0["trace"],
            "trace_1": r1["trace"],
            "refusal_0": r0.get("is_refusal", False),
            "refusal_1": r1.get("is_refusal", False),
        }
        if m_meta.get("task") == "hotpotqa":
            gold = m_meta.get("answer", "")
            ans0 = extract_predicted_answer(r0["trace"])
            ans1 = extract_predicted_answer(r1["trace"])
            row["pred_0"] = ans0
            row["pred_1"] = ans1
            row["gold"] = gold
            row["correct_0"] = hotpot_correct(ans0, gold)
            row["correct_1"] = hotpot_correct(ans1, gold)
        rows.append(row)

    # === TF-IDF cosine similarity per prompt ===
    all_traces_for_fit = [r["trace_0"] for r in rows] + [r["trace_1"] for r in rows]
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    vectorizer.fit(all_traces_for_fit)

    similarities = []
    for r in rows:
        v0 = vectorizer.transform([r["trace_0"]])
        v1 = vectorizer.transform([r["trace_1"]])
        sim = float(cosine_similarity(v0, v1)[0, 0])
        r["trace_similarity"] = sim
        similarities.append(sim)

    # === Compute label disagreement per task ===
    # XSTest: refusal disagreement
    # HotpotQA: correctness disagreement
    xs_rows = [r for r in rows if r["task"] == "xstest"]
    hp_rows = [r for r in rows if r["task"] == "hotpotqa"]

    xs_disagree = [int(r["refusal_0"] != r["refusal_1"]) for r in xs_rows]
    xs_sim = [r["trace_similarity"] for r in xs_rows]

    hp_disagree = [int(r.get("correct_0") != r.get("correct_1")) for r in hp_rows]
    hp_sim = [r["trace_similarity"] for r in hp_rows]

    print()
    print("=" * 60)
    print(f"PHASE 1 DISAGREEMENT ANALYSIS")
    print("=" * 60)

    # Stats per task
    def report(name, sim, disagree):
        if not sim:
            return None
        n = len(sim)
        n_dis = sum(disagree)
        sim = np.array(sim)
        dis = np.array(disagree)
        print(f"\n[{name}] n={n}, disagreement rate={n_dis/n:.1%} ({n_dis}/{n})")
        print(f"  Trace similarity: mean={sim.mean():.3f} median={np.median(sim):.3f} std={sim.std():.3f}")
        if n_dis > 0 and n_dis < n:
            sim_dis = sim[dis == 1].mean()
            sim_agree = sim[dis == 0].mean()
            print(f"  Mean similarity (disagree): {sim_dis:.3f}")
            print(f"  Mean similarity (agree):    {sim_agree:.3f}")
            print(f"  Delta: {sim_agree - sim_dis:+.3f} (positive = signal)")
            try:
                # AUC: lower similarity → predicts disagreement → use 1-sim as score
                auc = roc_auc_score(dis, -sim)
                print(f"  AUC(1-sim → disagree): {auc:.3f}")
            except Exception as e:
                auc = None
                print(f"  AUC failed: {e}")
            try:
                pearson = float(np.corrcoef(sim, dis)[0, 1])
                print(f"  Pearson r(sim, dis): {pearson:.3f}")
            except Exception:
                pearson = None
            return {"n": n, "disagree_rate": n_dis/n, "auc": auc, "pearson": pearson,
                    "sim_mean_disagree": float(sim_dis), "sim_mean_agree": float(sim_agree)}
        else:
            print(f"  (skipping AUC: too few cases for binary classification)")
            return {"n": n, "disagree_rate": n_dis/n, "auc": None, "pearson": None}

    xs_stats = report("XSTest", xs_sim, xs_disagree)
    hp_stats = report("HotpotQA", hp_sim, hp_disagree)

    # === GO/NO-GO ===
    print()
    print("=" * 60)
    print("GO / NO-GO DECISION")
    print("=" * 60)

    def go_signal(stats):
        if not stats or stats.get("auc") is None:
            return False
        return (stats["auc"] >= args.auc_threshold) or \
               (stats.get("pearson") is not None and abs(stats["pearson"]) >= args.pearson_threshold)

    xs_go = go_signal(xs_stats) if xs_stats else False
    hp_go = go_signal(hp_stats) if hp_stats else False
    overall_go = xs_go or hp_go

    print(f"  XSTest signal:    {'GO' if xs_go else 'NO-GO'}")
    print(f"  HotpotQA signal:  {'GO' if hp_go else 'NO-GO'}")
    print(f"  Overall:          {'>>> GO (proceed to Phase 2) <<<' if overall_go else '>>> NO-GO (pivot) <<<'}")

    # Save report
    out = {
        "phase": args.phase,
        "models": models,
        "n_total": len(rows),
        "xstest_stats": xs_stats,
        "hotpot_stats": hp_stats,
        "decision": {
            "xstest_go": xs_go,
            "hotpot_go": hp_go,
            "overall_go": overall_go,
            "auc_threshold": args.auc_threshold,
            "pearson_threshold": args.pearson_threshold,
        },
        "details": rows,
    }
    out_path = os.path.join(args.results_dir, f"{args.phase}_disagreement_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Exit code: 0 if GO, 1 if NO-GO (for orchestrator branching)
    import sys
    sys.exit(0 if overall_go else 1)


if __name__ == "__main__":
    main()
