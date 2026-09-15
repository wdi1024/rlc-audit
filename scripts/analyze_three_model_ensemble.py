#!/usr/bin/env python3
"""3-Model Ensemble (k>2) extension for Diagnostic-Gated Routing (E1).

Tests whether extending from k=2 (Qwen + Gemma) to k=3 (Qwen + Gemma + Llama)
changes the gate's admit/deny decision and AUC magnitude on XSTest 450 × 512-tok.

Aggregations tested:
  - score: mean / min / max over 3 pair-wise sentence-cosine similarities
  - label: "any pair disagrees" (= not all 3 SLMs unanimous), i.e.
           y_3model(x) = 1[max_i refuse_i(x) != min_i refuse_i(x)]

Output: results/disagree_routing/three_model_ensemble_report.json
"""
import json
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

RDIR = "results/disagree_routing"


def load_traces(slm, phase):
    path = f"{RDIR}/{phase}_xstest450_512tok_traces_{slm}.json"
    return {r["id"]: r.get("trace") or "" for r in json.load(open(path))["records"]}


def load_judge(slm, phase):
    path = f"{RDIR}/{phase}_xstest450_512tok_judge_{slm}_anthropic_claude-haiku-4-5-20251001.json"
    data = json.load(open(path))["records"]
    return {r["id"]: r["refusal_judge"] for r in data if r.get("refusal_judge") is not None}


def auc_with_ci(labels, scores, n_boot=2000, seed=42):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(set(labels)) < 2:
        return None, None
    point = roc_auc_score(labels, scores)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if len(set(labels[idx])) < 2:
            continue
        try:
            boots.append(roc_auc_score(labels[idx], scores[idx]))
        except Exception:
            pass
    if not boots:
        return float(point), None
    boots = np.array(boots)
    return float(point), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def cohens_d(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    s_a = s[y == 0]
    s_d = s[y == 1]
    if len(s_d) < 2 or len(s_a) < 2:
        return None
    n_a, n_d = len(s_a), len(s_d)
    var_a, var_d = s_a.var(ddof=1), s_d.var(ddof=1)
    pooled = np.sqrt((var_a * (n_a - 1) + var_d * (n_d - 1)) / (n_a + n_d - 2))
    if pooled == 0:
        return None
    return float((s_a.mean() - s_d.mean()) / pooled)


def main():
    print("=" * 78)
    print("3-Model Ensemble (k>2) Extension on XSTest 450 × 512-tok")
    print("=" * 78)

    # Load all 3 SLMs
    qt = load_traces("qwen3.5-2b", "phase7")
    gt = load_traces("gemma-4-e2b", "phase7")
    lt = load_traces("llama-3.2-3b", "phase9")
    qj = load_judge("qwen3.5-2b", "phase7")
    gj = load_judge("gemma-4-e2b", "phase7")
    lj = load_judge("llama-3.2-3b", "phase9")
    common = sorted(set(qt) & set(gt) & set(lt) & set(qj) & set(gj) & set(lj))
    print(f"\n[1/4] Common ids across 3 SLMs + 3 judges: {len(common)}")

    qts = [qt[i] for i in common]
    gts = [gt[i] for i in common]
    lts = [lt[i] for i in common]
    qjs = np.array([qj[i] for i in common], dtype=int)
    gjs = np.array([gj[i] for i in common], dtype=int)
    ljs = np.array([lj[i] for i in common], dtype=int)

    # Sentence embedding
    print("\n[2/4] Computing sentence embeddings (MiniLM-L12)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    eq = model.encode(qts, show_progress_bar=False, normalize_embeddings=True)
    eg = model.encode(gts, show_progress_bar=False, normalize_embeddings=True)
    el = model.encode(lts, show_progress_bar=False, normalize_embeddings=True)
    sim_qg = (eq * eg).sum(axis=1)
    sim_ql = (eq * el).sum(axis=1)
    sim_gl = (eg * el).sum(axis=1)

    # Pair-wise disagreement labels
    y_qg = (qjs != gjs).astype(int)
    y_ql = (qjs != ljs).astype(int)
    y_gl = (gjs != ljs).astype(int)

    # 3-model ensemble label: not all 3 unanimous
    stack = np.stack([qjs, gjs, ljs], axis=0)
    y_3 = ((stack.max(axis=0) != stack.min(axis=0))).astype(int)

    # Aggregate scores: mean / min / max over 3 pair-wise sims
    score_mean = (sim_qg + sim_ql + sim_gl) / 3
    score_min = np.minimum(np.minimum(sim_qg, sim_ql), sim_gl)
    score_max = np.maximum(np.maximum(sim_qg, sim_ql), sim_gl)

    # Headline cells
    print("\n[3/4] AUC + diagnostic d on aggregations:")
    print(f"\n  {'Configuration':45s} {'n_dis':>6s} {'rate':>7s} {'AUC':>7s} {'95% CI':>22s} {'sep_d':>8s}")
    results = {}
    configs = [
        ("Single-pair Q+G (baseline)", -sim_qg, y_qg),
        ("Single-pair Q+L", -sim_ql, y_ql),
        ("Single-pair G+L", -sim_gl, y_gl),
        ("3-model ensemble (mean score, any-disagree label)", -score_mean, y_3),
        ("3-model ensemble (min score, any-disagree label)", -score_min, y_3),
        ("3-model ensemble (max score, any-disagree label)", -score_max, y_3),
        ("3-model ensemble (mean score, Q+G label)", -score_mean, y_qg),
    ]
    for name, scores, labels in configs:
        auc, ci = auc_with_ci(labels, scores)
        # cohens_d expects raw similarity (not negated)
        # use the original similarity that was negated
        sims = -scores
        d = cohens_d(sims, labels)
        n_dis = int(labels.sum())
        rate = float(labels.mean())
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        d_str = f"{d:.3f}" if d is not None else "—"
        print(f"  {name:45s} {n_dis:>6d} {rate*100:>6.1f}% {auc:>7.3f} {ci_str:>22s} {d_str:>8s}")
        results[name] = dict(n_dis=n_dis, rate=rate, auc=auc, ci=ci, sep_d=d)

    # Save
    out_path = f"{RDIR}/three_model_ensemble_report.json"
    json.dump(dict(
        n_common=len(common),
        per_model_refusal_rates=dict(
            qwen=float(qjs.mean()), gemma=float(gjs.mean()), llama=float(ljs.mean()),
        ),
        configs=results,
    ), open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Verdict
    print("\n[4/4] " + "=" * 70)
    print("VERDICT — Does k=3 extend the gate's admit decision?")
    print("=" * 78)

    base = results["Single-pair Q+G (baseline)"]
    ens_mean = results["3-model ensemble (mean score, any-disagree label)"]
    print(f"\n  Q+G baseline:                AUC {base['auc']:.3f} [CI {base['ci'][0]:.3f}, {base['ci'][1]:.3f}], d={base['sep_d']:.3f}, n_dis={base['n_dis']}")
    print(f"  3-model ensemble (mean):     AUC {ens_mean['auc']:.3f} [CI {ens_mean['ci'][0]:.3f}, {ens_mean['ci'][1]:.3f}], d={ens_mean['sep_d']:.3f}, n_dis={ens_mean['n_dis']}")

    # Gate conditions
    n0, d0 = 30, 0.15
    base_pass = base["n_dis"] >= n0 and base["sep_d"] is not None and base["sep_d"] > d0
    ens_pass = ens_mean["n_dis"] >= n0 and ens_mean["sep_d"] is not None and ens_mean["sep_d"] > d0
    print(f"\n  Gate (n0={n0}, d0={d0}):")
    print(f"    Q+G baseline:    {'ADMIT ✓' if base_pass else 'DENY ✗'}")
    print(f"    3-model mean:    {'ADMIT ✓' if ens_pass else 'DENY ✗'}")

    if ens_pass and base_pass:
        delta_auc = ens_mean['auc'] - base['auc']
        delta_d = ens_mean['sep_d'] - base['sep_d']
        if delta_auc > 0.02:
            print(f"\n→ Verdict: k=3 ensemble IMPROVES the admitted configuration.")
            print(f"  ΔAUC = +{delta_auc:.3f}, Δd = {delta_d:+.3f}. Aggregate score gives stronger signal.")
        elif abs(delta_auc) < 0.02:
            print(f"\n→ Verdict: k=3 ensemble preserves but does not significantly improve.")
            print(f"  ΔAUC = {delta_auc:+.3f}, Δd = {delta_d:+.3f}. Same admitted regime, no synergy.")
        else:
            print(f"\n→ Verdict: k=3 ensemble degrades the signal.")
            print(f"  ΔAUC = {delta_auc:+.3f}, Δd = {delta_d:+.3f}. Single-pair preferable.")
    elif ens_pass and not base_pass:
        print(f"\n→ Verdict: k=3 ensemble RECOVERS gate admission where Q+G failed.")
    elif not ens_pass and base_pass:
        print(f"\n→ Verdict: k=3 ensemble denies gate where Q+G admitted (regression).")
    else:
        print(f"\n→ Verdict: both denied.")


if __name__ == "__main__":
    main()
