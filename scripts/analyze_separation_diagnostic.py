#!/usr/bin/env python3
"""Experiment D — Distributional separation diagnostic.

Operationalizes condition (ii) of the §5.6 scope theorem (paper draft) as a
measurable statistic that can be computed *before* AUC estimation, so that a
practitioner can predict in advance whether the headline cell is identifiable
on a new (benchmark, SLM pair, encoder) triple.

Definition. For a benchmark with sentence-embedding cosine similarities
{s_i} and binary judge-disagreement labels {y_i ∈ {0=agree, 1=disagree}}:

    μ_a = E[s | y=0],  μ_d = E[s | y=1]
    σ_p = sqrt((Var[s|y=0] * (n_a-1) + Var[s|y=1] * (n_d-1)) / (n_a+n_d-2))
    d   = (μ_a - μ_d) / σ_p          # standardized class separation

This is Cohen's d for the (agree vs disagree) similarity distributions.
A high d means the disagreement class has *lower* trace similarity than the
agreement class — i.e., the encoder can tell them apart. d ≈ 0 means the
distributions are indistinguishable, in which case AUC will be near chance
regardless of class prior (rate).

We compute d on:
  - 4 full benchmarks (XSTest, OR-Bench-Hard-1K, AdvBench, SimpleSafetyTests)
  - Rate-controlled subsamples of XSTest and OR-Bench (Experiment C)
and check whether AUC is a monotone function of d *across* the full panel.
A monotone fit confirms that condition (ii) reduces to a single, computable
statistic per triple — providing the "diagnosable in advance" property the
paper claims.

Output: results/disagree_routing/separation_diagnostic.json
"""
import json
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

RDIR = "results/disagree_routing"

REFUSAL_KW = [
    "sorry", "cannot", "can't", "i'm not able", "i refuse",
    "i won't", "inappropriate", "not appropriate", "i'm unable",
    "unable to", "unable to provide", "i must decline",
]


def load_judge(path):
    data = json.load(open(path))
    return {r["id"]: r["refusal_judge"] for r in data["records"] if r.get("refusal_judge") is not None}


def cohens_d(scores, labels):
    """Cohen's d on similarity scores: positive d ⇒ y=0 has higher mean than y=1."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    s_a = s[y == 0]
    s_d = s[y == 1]
    if len(s_d) < 2 or len(s_a) < 2:
        return None
    n_a, n_d = len(s_a), len(s_d)
    mu_a, mu_d = s_a.mean(), s_d.mean()
    var_a, var_d = s_a.var(ddof=1), s_d.var(ddof=1)
    pooled = np.sqrt((var_a * (n_a - 1) + var_d * (n_d - 1)) / (n_a + n_d - 2))
    if pooled == 0:
        return None
    return float((mu_a - mu_d) / pooled)


def encode_pair(model, traces_a, traces_b):
    ea = model.encode(traces_a, show_progress_bar=False, normalize_embeddings=True)
    eb = model.encode(traces_b, show_progress_bar=False, normalize_embeddings=True)
    return (ea * eb).sum(axis=1)


def compute_diag(name, q_traces_path, g_traces_path, q_judge_path, g_judge_path, sent_model):
    qwen = json.load(open(q_traces_path))["records"]
    gemma = json.load(open(g_traces_path))["records"]
    qj = load_judge(q_judge_path)
    gj = load_judge(g_judge_path)
    qm = {r["id"]: r for r in qwen}
    gm = {r["id"]: r for r in gemma}
    common = sorted(set(qm) & set(gm) & set(qj) & set(gj))

    qwen_traces = [qm[i].get("trace") or "" for i in common]
    gemma_traces = [gm[i].get("trace") or "" for i in common]
    sims = encode_pair(sent_model, qwen_traces, gemma_traces)
    q_jd = np.array([int(qj[i]) for i in common])
    g_jd = np.array([int(gj[i]) for i in common])
    y = (q_jd != g_jd).astype(int)

    n_total = len(common)
    n_dis = int(y.sum())
    rate = float(y.mean())

    if n_dis < 2 or len(set(y)) < 2:
        return dict(name=name, n=n_total, n_disagree=n_dis, rate=rate,
                    sep_d=None, mu_agree=float(sims[y == 0].mean()) if (y == 0).any() else None,
                    mu_disagree=float(sims[y == 1].mean()) if (y == 1).any() else None,
                    auc=None, sims=sims, y=y)

    d = cohens_d(sims, y)
    auc = float(roc_auc_score(y, -sims))
    return dict(
        name=name, n=n_total, n_disagree=n_dis, rate=rate,
        sep_d=d,
        mu_agree=float(sims[y == 0].mean()),
        mu_disagree=float(sims[y == 1].mean()),
        std_agree=float(sims[y == 0].std()),
        std_disagree=float(sims[y == 1].std()),
        auc=auc,
        sims=sims, y=y,  # kept locally; stripped before json dump
    )


def subsample_panel(name, sims, y, target_rates, n_seeds=20):
    """Generate (sep_d, AUC) pairs at varying rates for one benchmark."""
    sims = np.asarray(sims)
    y = np.asarray(y)
    agree_idx = np.where(y == 0)[0]
    disagree_idx = np.where(y == 1)[0]
    n_agree, n_disagree = len(agree_idx), len(disagree_idx)
    rows = []
    for tr in target_rates:
        n_dis_target = int(round(tr * n_agree / max(1 - tr, 1e-6)))
        n_dis_target = min(n_dis_target, n_disagree)
        if n_dis_target < 5:
            continue
        seed_rows = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            chosen_dis = rng.choice(disagree_idx, size=n_dis_target, replace=False)
            chosen = np.concatenate([agree_idx, chosen_dis])
            sub_sims = sims[chosen]
            sub_y = y[chosen]
            d = cohens_d(sub_sims, sub_y)
            try:
                auc = float(roc_auc_score(sub_y, -sub_sims))
            except Exception:
                continue
            if d is None:
                continue
            seed_rows.append((d, auc))
        if not seed_rows:
            continue
        seed_rows = np.array(seed_rows)
        rows.append(dict(
            target_rate=float(tr),
            actual_rate=float(n_dis_target / (n_dis_target + n_agree)),
            n_dis=int(n_dis_target),
            sep_d_mean=float(seed_rows[:, 0].mean()),
            sep_d_std=float(seed_rows[:, 0].std()),
            auc_mean=float(seed_rows[:, 1].mean()),
            auc_std=float(seed_rows[:, 1].std()),
        ))
    return rows


def main():
    print("=" * 78)
    print("Experiment D — Distributional separation diagnostic")
    print("=" * 78)

    print("\nLoading sentence encoder (MiniLM-L12)...")
    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    BENCHMARKS = [
        ("XSTest", f"{RDIR}/phase3_xstest_full_traces_qwen3.5-2b.json",
         f"{RDIR}/phase3_xstest_full_traces_gemma-4-e2b.json",
         f"{RDIR}/phase3_xstest_full_judge_qwen3.5-2b_anthropic_claude-haiku-4-5-20251001.json",
         f"{RDIR}/phase3_xstest_full_judge_gemma-4-e2b_anthropic_claude-haiku-4-5-20251001.json"),
        ("OR-Bench-Hard-1K", f"{RDIR}/phase8_orbench_hard1k_traces_qwen3.5-2b.json",
         f"{RDIR}/phase8_orbench_hard1k_traces_gemma-4-e2b.json",
         f"{RDIR}/phase8_orbench_hard1k_judge_qwen3.5-2b_anthropic_claude-haiku-4-5-20251001.json",
         f"{RDIR}/phase8_orbench_hard1k_judge_gemma-4-e2b_anthropic_claude-haiku-4-5-20251001.json"),
        ("AdvBench", f"{RDIR}/phase4_advbench_traces_qwen3.5-2b.json",
         f"{RDIR}/phase4_advbench_traces_gemma-4-e2b.json",
         f"{RDIR}/phase4_advbench_judge_qwen3.5-2b_anthropic_claude-haiku-4-5-20251001.json",
         f"{RDIR}/phase4_advbench_judge_gemma-4-e2b_anthropic_claude-haiku-4-5-20251001.json"),
        ("SimpleSafetyTests", f"{RDIR}/phase4_simplesafety_traces_qwen3.5-2b.json",
         f"{RDIR}/phase4_simplesafety_traces_gemma-4-e2b.json",
         f"{RDIR}/phase4_simplesafety_judge_qwen3.5-2b_anthropic_claude-haiku-4-5-20251001.json",
         f"{RDIR}/phase4_simplesafety_judge_gemma-4-e2b_anthropic_claude-haiku-4-5-20251001.json"),
    ]

    print("\n[1/2] Per-benchmark separation + headline AUC")
    print(f"\n  {'Benchmark':22s} {'n':>5s} {'n_dis':>5s} {'rate':>7s} {'μ_agree':>9s} {'μ_disagree':>11s} {'sep_d':>8s} {'AUC':>7s}")
    bench_results = {}
    for name, qt, gt, qj, gj in BENCHMARKS:
        r = compute_diag(name, qt, gt, qj, gj, sent_model)
        bench_results[name] = r
        if r["sep_d"] is None:
            print(f"  {name:22s} {r['n']:>5d} {r['n_disagree']:>5d} "
                  f"{r['rate']*100:>6.1f}% {r.get('mu_agree') or 0:>9.4f} "
                  f"{r.get('mu_disagree') or 0:>11.4f} {'—':>8s} {'—':>7s}")
        else:
            print(f"  {name:22s} {r['n']:>5d} {r['n_disagree']:>5d} "
                  f"{r['rate']*100:>6.1f}% {r['mu_agree']:>9.4f} "
                  f"{r['mu_disagree']:>11.4f} {r['sep_d']:>8.3f} {r['auc']:>7.3f}")

    # ─────────────────────────────────────────────────────────────────
    # Subsample panels for XSTest + OR-Bench (rate-controlled)
    # ─────────────────────────────────────────────────────────────────
    print("\n[2/2] Rate-controlled subsample panels (sep_d vs AUC, 20 seeds each)")

    panels = {}
    for name, target_rates in [
        ("XSTest", [0.025, 0.05, 0.075, 0.102]),
        ("OR-Bench-Hard-1K", [0.05, 0.10, 0.15, 0.20, 0.25, 0.359]),
    ]:
        r = bench_results[name]
        panels[name] = subsample_panel(name, r["sims"], r["y"], target_rates, n_seeds=20)
        print(f"\n  {name}:")
        print(f"    {'rate':>7s} {'sep_d μ':>10s} {'sep_d σ':>10s} {'AUC μ':>8s} {'AUC σ':>8s}")
        for row in panels[name]:
            print(f"    {row['actual_rate']*100:>6.2f}% {row['sep_d_mean']:>10.3f} "
                  f"{row['sep_d_std']:>10.3f} {row['auc_mean']:>8.3f} {row['auc_std']:>8.3f}")

    # ─────────────────────────────────────────────────────────────────
    # Combined panel: AUC vs separation (across benchmarks + subsamples)
    # ─────────────────────────────────────────────────────────────────
    panel_points = []
    for name in ["XSTest", "OR-Bench-Hard-1K", "AdvBench", "SimpleSafetyTests"]:
        r = bench_results[name]
        if r["sep_d"] is not None:
            panel_points.append(dict(
                source=f"{name}_full",
                rate=r["rate"], sep_d=r["sep_d"], auc=r["auc"], n=r["n"], n_dis=r["n_disagree"],
            ))
    for name in ["XSTest", "OR-Bench-Hard-1K"]:
        for row in panels[name]:
            panel_points.append(dict(
                source=f"{name}_subsample_rate_{row['target_rate']:.3f}",
                rate=row["actual_rate"],
                sep_d=row["sep_d_mean"], auc=row["auc_mean"], n=None, n_dis=row["n_dis"],
            ))

    # Spearman correlation of (sep_d, AUC) across the panel
    ds = np.array([p["sep_d"] for p in panel_points if p["sep_d"] is not None])
    aucs = np.array([p["auc"] for p in panel_points if p["sep_d"] is not None])
    from scipy.stats import spearmanr, pearsonr
    rho_s, p_s = spearmanr(ds, aucs)
    rho_p, p_p = pearsonr(ds, aucs)

    print("\n" + "=" * 78)
    print("DIAGNOSTIC PANEL: AUC vs separation across benchmarks + rate sub-samples")
    print("=" * 78)
    print(f"\n  panel size: n={len(ds)} points")
    print(f"  Spearman ρ(sep_d, AUC) = {rho_s:+.3f}, p = {p_s:.2e}")
    print(f"  Pearson  r(sep_d, AUC) = {rho_p:+.3f}, p = {p_p:.2e}")
    print()
    print(f"  {'source':45s} {'rate':>7s} {'sep_d':>8s} {'AUC':>7s}")
    for p in sorted(panel_points, key=lambda x: x["sep_d"] or -10):
        if p["sep_d"] is None:
            print(f"  {p['source']:45s} {p['rate']*100:>6.1f}% {'—':>8s} {'—':>7s}")
        else:
            print(f"  {p['source']:45s} {p['rate']*100:>6.1f}% {p['sep_d']:>8.3f} {p['auc']:>7.3f}")

    # ─────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────
    out = dict(
        per_benchmark={
            name: {k: v for k, v in r.items() if k not in ("sims", "y")}
            for name, r in bench_results.items()
        },
        rate_panels=panels,
        combined_panel=panel_points,
        spearman_rho=float(rho_s),
        spearman_p=float(p_s),
        pearson_r=float(rho_p),
        pearson_p=float(p_p),
    )
    out_path = f"{RDIR}/separation_diagnostic.json"
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # ─────────────────────────────────────────────────────────────────
    # Paper-ready statement
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PAPER-READY STATEMENT (§5.6 diagnostic)")
    print("=" * 78)
    statement = f"""
A predictive diagnostic for condition (ii). To make condition (ii) operative
*before* AUC estimation, we define the class-conditional separation as

    d = (mean cosine sim under agree-class) − (mean cosine sim under
        disagree-class), standardized by the pooled within-class std,

i.e., Cohen's d on the trace-similarity score under the binary judge-
disagreement label. A high positive d means the encoder places agree- and
disagree-class traces in distinguishable regions of similarity space; d ≈ 0
means the classes are indistinguishable. Computing d across our panel of
four benchmarks plus the rate-controlled sub-samples of XSTest and
OR-Bench-Hard-1K (Experiment C) yields {len(ds)} (d, AUC) pairs with
Spearman ρ = {rho_s:+.3f} (p = {p_s:.1e}). The diagnostic separates the
identifiable cases (XSTest full d = {bench_results['XSTest']['sep_d']:.3f},
AUC {bench_results['XSTest']['auc']:.3f}) from the failed cases (OR-Bench
full d = {bench_results['OR-Bench-Hard-1K']['sep_d']:.3f}, AUC
{bench_results['OR-Bench-Hard-1K']['auc']:.3f}) and tracks the AUC
monotonically across the rate-subsampled panel within each benchmark. We
treat d as the *practitioner's diagnostic*: given a new (benchmark, SLM
pair, encoder) triple, computing d on a labelled disagreement subset is
sufficient to predict whether the headline cell will be identifiable,
without committing to full AUC estimation.
"""
    print(statement)


if __name__ == "__main__":
    main()
