#!/usr/bin/env python3
"""RLC-Audit of a published routing recipe, not a setup we designed (2026-08-13).

FrugalGPT trains a scorer on the query and the model's answer text and validates it
against correctness defined as a string match on that same answer text.  That is the
representation-label coupling precondition stated in a deployed recipe: score and
proxy read one span, while the semantic construct -- whether the answer is actually
right -- depends on more than whether a gold string appears.

This script instantiates that recipe on public data we already have judged.  It is a
reproduction of the *recipe*, not of the released FrugalGPT system: the scorer here
is a TF-IDF + logistic model rather than DistilBERT, trained out-of-fold on
(question, answer) text to predict the string-match label.  Everything else follows
the recipe: same span for score and proxy, independent judge adjudication as the
construct, 10% escalation budget.

  score s_i  = out-of-fold P(string-match correct | question, answer text)
  proxy z_i  = gold answer string appears in the trace  (the recipe's label)
  construct y_i = judge adjudication of correctness on the same trace

The audit asks whether a scorer validated against z can be read as evidence about y.

  python3 audit_frugalgpt_recipe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "disagree_routing"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
N_BOOT = 2000
RNG = np.random.default_rng(0)

TASKS = [
    ("HotpotQA multi-hop", "phase10_hotpot", ["qwen3.5-2b", "gemma-4-e2b"], "correctness"),
    ("GSM8K reasoning", "phase11_gsm8k", ["qwen3.5-2b", "gemma-4-e2b"], "correctness"),
]


def load_task(phase: str, model: str, kind: str):
    traces = {r["id"]: r for r in
              json.loads((DATA / f"{phase}_traces_{model}.json").read_text())["records"]}
    judged = {r["id"]: r for r in
              json.loads((DATA / f"{phase}_{kind}_{model}_{JUDGE}.json").read_text())["records"]}
    ids = sorted(set(traces) & set(judged))
    rows = []
    for i in ids:
        t = traces[i]
        j = judged[i]
        text = t.get("trace") or ""
        gold = (j.get("gold_answer") or "").strip()
        if not text.strip() or not gold:
            continue
        rows.append({
            "id": i,
            "question": j.get("question") or t.get("prompt", "")[:400],
            "answer": text,
            # the recipe's proxy: is the gold string present in what the model wrote
            "z": int(gold.lower() in text.lower()),
            "y": int(bool(j.get("correct_judge"))),
        })
    return rows


def oof_scores(rows: list[dict], target: str, seed: int = 0) -> np.ndarray:
    """Out-of-fold P(target=1 | question + answer text). No gold answer is shown."""
    X = [f"{r['question']}\n{r['answer']}" for r in rows]
    y = np.array([r[target] for r in rows])
    pred = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y)
    for tr, te in folds:
        clf = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                            max_features=50000),
            LogisticRegression(max_iter=2000, C=2.0))
        clf.fit([X[i] for i in tr], y[tr])
        pred[te] = clf.predict_proba([X[i] for i in te])[:, 1]
    return pred


def paired_boot(y1, s, y2) -> tuple[float, float, float]:
    """Bootstrap CI on AUC(s, y1) - AUC(s, y2) with the same resampled rows."""
    d = []
    n = len(s)
    for _ in range(N_BOOT):
        sel = RNG.integers(0, n, n)
        if len(set(y1[sel])) < 2 or len(set(y2[sel])) < 2:
            continue
        d.append(roc_auc_score(y1[sel], s[sel]) - roc_auc_score(y2[sel], s[sel]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    out = {"recipe": "FrugalGPT-style scorer on (question, answer) validated against "
                     "string-match correctness", "tasks": {}}
    for name, phase, models, kind in TASKS:
        rows = []
        for m in models:
            for r in load_task(phase, m, kind):
                r["model"] = m
                rows.append(r)
        z = np.array([r["z"] for r in rows])
        y = np.array([r["y"] for r in rows])
        if len(set(z)) < 2 or len(set(y)) < 2:
            print(f"{name}: degenerate labels, skipped")
            continue

        # The recipe: the scorer is trained and validated against the proxy.
        s = oof_scores(rows, "z")
        # Escalation routes the lowest-scoring (predicted-incorrect) queries.
        defer = -s
        B = max(1, int(round(len(rows) * 0.10)))
        idx = np.argsort(-defer, kind="stable")[:B]

        auc_z = float(roc_auc_score(z, s))
        auc_y = float(roc_auc_score(y, s))
        ap_z = float(average_precision_score(z, s))
        ap_y = float(average_precision_score(y, s))
        k = float(cohen_kappa_score(z, y))
        gap, lo, hi = paired_boot(z, s, y)
        # what the escalation budget actually catches
        wrong_by_judge = int((1 - y)[idx].sum())
        wrong_by_proxy = int((1 - z)[idx].sum())

        print(f"\n{name}  n={len(rows)} ({', '.join(models)})")
        print(f"  proxy prevalence {z.mean():.3f}   construct prevalence {y.mean():.3f}"
              f"   kappa(z,y)={k:.3f}")
        print(f"  scorer vs proxy      AUC {auc_z:.3f}  AP {ap_z:.3f}")
        print(f"  scorer vs construct  AUC {auc_y:.3f}  AP {ap_y:.3f}")
        print(f"  gap {gap:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"  at a 10% escalation budget (B={B}): {wrong_by_proxy}/{B} fail the "
              f"proxy, {wrong_by_judge}/{B} are judged wrong "
              f"(base rate {1 - y.mean():.3f})")

        out["tasks"][name] = {
            "n": len(rows), "models": models,
            "prevalence_proxy": float(z.mean()), "prevalence_construct": float(y.mean()),
            "kappa_z_y": k, "auc_proxy": auc_z, "auc_construct": auc_y,
            "ap_proxy": ap_z, "ap_construct": ap_y,
            "auc_gap": gap, "auc_gap_ci": [lo, hi], "budget": B,
            "routed_proxy_fail": wrong_by_proxy, "routed_judge_wrong": wrong_by_judge,
            "base_rate_wrong": float(1 - y.mean()),
        }

    p = ROOT / "analysis_results" / "frugalgpt_recipe_audit.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
