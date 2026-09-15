#!/usr/bin/env python3
"""Experiment C — Rate-controlled subsampling test.

Tests whether disagreement rate is a *sufficient* condition for identifiability,
or merely necessary. Within each benchmark, we keep all agreement events and
sub-sample the disagreement events to target rates, then re-compute the
sentence × judge AUC.

Two scenarios:
  (α) AUC tracks disagreement rate → rate is sufficient (cleanest scope theorem)
  (β) AUC stays roughly constant within a benchmark independent of subsampled
      rate → rate is necessary but not sufficient; the underlying score
      separation is benchmark-specific.

We test on:
  - XSTest         (original rate 10.2%, 46 disagree / 450 total)
    → sub-sample to 2.5%, 5%, 7.5%, 10.2% (max)
  - OR-Bench-Hard-1K (original rate 35.9%, 473 / 1319)
    → sub-sample to 5%, 10%, 15%, 20%, 25%, 35.9% (max)

For each (benchmark, target rate), we run 20 random seeds and report mean ±
std of AUC + bootstrap CI.

Output: results/disagree_routing/rate_controlled_subsample.json
"""
import json
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

RDIR = "results/disagree_routing"

REFUSAL_KW = [
    "sorry", "cannot", "can't", "i'm not able", "i refuse",
    "i won't", "inappropriate", "not appropriate", "i'm unable",
    "unable to", "unable to provide", "i must decline",
]


def is_kw(t):
    if not t:
        return False
    s = t.strip().lower()
    return any(kw in s for kw in REFUSAL_KW)


def load_judge(path):
    data = json.load(open(path))
    return {r["id"]: r["refusal_judge"] for r in data["records"] if r.get("refusal_judge") is not None}


def auc_with_ci(labels, scores, n_boot=1000, seed=42):
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
        return point, None
    boots = np.array(boots)
    ci = np.percentile(boots, [2.5, 97.5])
    return float(point), [float(ci[0]), float(ci[1])]


def precompute_signals(qm, gm, qj, gj, all_ids, sent_model):
    """Compute sentence × judge similarity once per benchmark.

    Returns: dict of id -> (sent_sim, tfidf_sim, q_jd, g_jd, q_kw, g_kw, q_trace, g_trace)
    """
    qwen_traces = [qm[i].get("trace") or "" for i in all_ids]
    gemma_traces = [gm[i].get("trace") or "" for i in all_ids]
    q_kw = np.array([int(is_kw(qm[i].get("trace"))) for i in all_ids])
    g_kw = np.array([int(is_kw(gm[i].get("trace"))) for i in all_ids])
    q_jd = np.array([int(qj[i]) for i in all_ids])
    g_jd = np.array([int(gj[i]) for i in all_ids])

    # Sentence sims
    ea = sent_model.encode(qwen_traces, show_progress_bar=False, normalize_embeddings=True)
    eb = sent_model.encode(gemma_traces, show_progress_bar=False, normalize_embeddings=True)
    sent_sims = (ea * eb).sum(axis=1)

    # TF-IDF sims
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(qwen_traces + gemma_traces)
    XA = vec.transform(qwen_traces)
    XB = vec.transform(gemma_traces)
    tfidf_sims = np.array([float(cosine_similarity(a, b)[0, 0]) for a, b in zip(XA, XB)])

    return dict(
        ids=np.asarray(all_ids),
        q_kw=q_kw, g_kw=g_kw, q_jd=q_jd, g_jd=g_jd,
        sent_sims=sent_sims, tfidf_sims=tfidf_sims,
    )


def auc_at_rate(signals, target_rate, seed):
    """Sub-sample disagreements to hit target rate, return AUC for sent × judge + tfidf × keyword."""
    n_total = len(signals["ids"])
    y_judge_all = (signals["q_jd"] != signals["g_jd"]).astype(int)
    y_kw_all = (signals["q_kw"] != signals["g_kw"]).astype(int)

    agree_idx = np.where(y_judge_all == 0)[0]
    disagree_idx = np.where(y_judge_all == 1)[0]
    n_agree = len(agree_idx)
    n_disagree = len(disagree_idx)

    # We keep all agree, sub-sample disagree to hit target rate
    # rate = n_dis_keep / (n_dis_keep + n_agree)  ⇒  n_dis_keep = rate * n_agree / (1 - rate)
    n_dis_target = int(round(target_rate * n_agree / max(1 - target_rate, 1e-6)))
    n_dis_target = min(n_dis_target, n_disagree)
    if n_dis_target < 5:
        return None

    rng = np.random.default_rng(seed)
    chosen_dis = rng.choice(disagree_idx, size=n_dis_target, replace=False)
    chosen = np.concatenate([agree_idx, chosen_dis])

    actual_rate = n_dis_target / (n_dis_target + n_agree)
    y_jg = y_judge_all[chosen]
    y_kw = y_kw_all[chosen]

    sent_score = -signals["sent_sims"][chosen]
    tfidf_score = -signals["tfidf_sims"][chosen]

    auc_sj, ci_sj = auc_with_ci(y_jg, sent_score, n_boot=500, seed=seed + 1)
    auc_tk, ci_tk = auc_with_ci(y_kw, tfidf_score, n_boot=500, seed=seed + 2)
    auc_tj, ci_tj = auc_with_ci(y_jg, tfidf_score, n_boot=500, seed=seed + 3)
    auc_sk, ci_sk = auc_with_ci(y_kw, sent_score, n_boot=500, seed=seed + 4)

    return dict(
        n_total=int(len(chosen)),
        n_disagree=int(y_jg.sum()),
        actual_rate=float(actual_rate),
        sent_x_judge=auc_sj, sent_x_judge_ci=ci_sj,
        tfidf_x_keyword=auc_tk, tfidf_x_keyword_ci=ci_tk,
        tfidf_x_judge=auc_tj, tfidf_x_judge_ci=ci_tj,
        sent_x_keyword=auc_sk, sent_x_keyword_ci=ci_sk,
    )


def run_benchmark(name, q_traces_path, g_traces_path, q_judge_path, g_judge_path,
                   target_rates, sent_model, n_seeds=20):
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    qwen = json.load(open(q_traces_path))["records"]
    gemma = json.load(open(g_traces_path))["records"]
    qj = load_judge(q_judge_path)
    gj = load_judge(g_judge_path)
    qm = {r["id"]: r for r in qwen}
    gm = {r["id"]: r for r in gemma}
    common = sorted(set(qm) & set(gm) & set(qj) & set(gj))
    print(f"  common ids: {len(common)}")

    signals = precompute_signals(qm, gm, qj, gj, common, sent_model)
    y_jg = (signals["q_jd"] != signals["g_jd"]).astype(int)
    print(f"  baseline disagreement rate: {y_jg.mean()*100:.1f}% ({y_jg.sum()}/{len(y_jg)})")

    results = []
    print(f"\n  {'target':>8s} {'actual':>8s} {'n':>5s} {'n_dis':>5s} "
          f"{'Sent×J μ':>10s} {'Sent×J σ':>10s} {'TF×kw μ':>9s}")
    for tr in target_rates:
        seed_results = []
        for seed in range(n_seeds):
            r = auc_at_rate(signals, tr, seed=seed)
            if r is not None and r["sent_x_judge"] is not None:
                seed_results.append(r)
        if not seed_results:
            print(f"  {tr*100:7.2f}% — too few disagree events to sub-sample")
            continue

        sj_aucs = np.array([r["sent_x_judge"] for r in seed_results])
        tk_aucs = np.array([r["tfidf_x_keyword"] for r in seed_results])
        tj_aucs = np.array([r["tfidf_x_judge"] for r in seed_results])
        sk_aucs = np.array([r["sent_x_keyword"] for r in seed_results])
        actual_rates = np.array([r["actual_rate"] for r in seed_results])
        n_dis = np.array([r["n_disagree"] for r in seed_results])
        n_total = np.array([r["n_total"] for r in seed_results])

        summary = dict(
            target_rate=float(tr),
            mean_actual_rate=float(actual_rates.mean()),
            mean_n_total=float(n_total.mean()),
            mean_n_disagree=float(n_dis.mean()),
            n_seeds=len(seed_results),
            sent_x_judge_mean=float(sj_aucs.mean()),
            sent_x_judge_std=float(sj_aucs.std()),
            sent_x_judge_min=float(sj_aucs.min()),
            sent_x_judge_max=float(sj_aucs.max()),
            tfidf_x_keyword_mean=float(tk_aucs.mean()),
            tfidf_x_keyword_std=float(tk_aucs.std()),
            tfidf_x_judge_mean=float(tj_aucs.mean()),
            tfidf_x_judge_std=float(tj_aucs.std()),
            sent_x_keyword_mean=float(sk_aucs.mean()),
            sent_x_keyword_std=float(sk_aucs.std()),
        )
        results.append(summary)

        print(f"  {tr*100:7.2f}% {summary['mean_actual_rate']*100:7.2f}% {summary['mean_n_total']:5.0f} "
              f"{summary['mean_n_disagree']:5.0f} "
              f"{summary['sent_x_judge_mean']:10.3f} {summary['sent_x_judge_std']:10.3f} "
              f"{summary['tfidf_x_keyword_mean']:9.3f}")

    return dict(
        baseline_rate=float(y_jg.mean()),
        baseline_n_total=int(len(y_jg)),
        baseline_n_disagree=int(y_jg.sum()),
        rate_curve=results,
    )


def main():
    print("=" * 78)
    print("Experiment C — Rate-controlled subsampling: is disagreement rate sufficient?")
    print("=" * 78)

    print("\nLoading sentence encoder...")
    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    out = {}

    out["XSTest"] = run_benchmark(
        "XSTest (baseline 10.2%)",
        f"{RDIR}/phase3_xstest_full_traces_qwen3.5-2b.json",
        f"{RDIR}/phase3_xstest_full_traces_gemma-4-e2b.json",
        f"{RDIR}/phase3_xstest_full_judge_qwen3.5-2b_anthropic_claude-haiku-4-5-20251001.json",
        f"{RDIR}/phase3_xstest_full_judge_gemma-4-e2b_anthropic_claude-haiku-4-5-20251001.json",
        target_rates=[0.025, 0.05, 0.075, 0.102],
        sent_model=sent_model,
        n_seeds=20,
    )

    out["OR-Bench-Hard-1K"] = run_benchmark(
        "OR-Bench-Hard-1K (baseline 35.9%)",
        f"{RDIR}/phase8_orbench_hard1k_traces_qwen3.5-2b.json",
        f"{RDIR}/phase8_orbench_hard1k_traces_gemma-4-e2b.json",
        f"{RDIR}/phase8_orbench_hard1k_judge_qwen3.5-2b_anthropic_claude-haiku-4-5-20251001.json",
        f"{RDIR}/phase8_orbench_hard1k_judge_gemma-4-e2b_anthropic_claude-haiku-4-5-20251001.json",
        target_rates=[0.05, 0.10, 0.15, 0.20, 0.25, 0.359],
        sent_model=sent_model,
        n_seeds=20,
    )

    out_path = f"{RDIR}/rate_controlled_subsample.json"
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # ─────────────────────────────────────────────────────────────────
    # Synthesis
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SYNTHESIS — Is disagreement rate sufficient?")
    print("=" * 78)

    print("\nXSTest sentence×judge AUC vs disagreement rate:")
    for r in out["XSTest"]["rate_curve"]:
        print(f"  rate {r['mean_actual_rate']*100:5.1f}%: AUC {r['sent_x_judge_mean']:.3f} ± {r['sent_x_judge_std']:.3f}")

    print("\nOR-Bench sentence×judge AUC vs disagreement rate:")
    for r in out["OR-Bench-Hard-1K"]["rate_curve"]:
        print(f"  rate {r['mean_actual_rate']*100:5.1f}%: AUC {r['sent_x_judge_mean']:.3f} ± {r['sent_x_judge_std']:.3f}")

    # Verdict
    if out["XSTest"]["rate_curve"] and out["OR-Bench-Hard-1K"]["rate_curve"]:
        xs_at_10 = next((r for r in out["XSTest"]["rate_curve"] if abs(r["mean_actual_rate"] - 0.10) < 0.02), None)
        or_at_10 = next((r for r in out["OR-Bench-Hard-1K"]["rate_curve"] if abs(r["mean_actual_rate"] - 0.10) < 0.02), None)
        if xs_at_10 and or_at_10:
            xs_auc = xs_at_10["sent_x_judge_mean"]
            or_auc = or_at_10["sent_x_judge_mean"]
            gap = xs_auc - or_auc
            print(f"\nXSTest @ 10% rate: AUC {xs_auc:.3f}")
            print(f"OR-Bench @ 10% rate: AUC {or_auc:.3f}")
            print(f"Gap at matched rate: {gap:+.3f}")
            if gap < 0.05:
                print("→ Verdict: AUC matches at matched rate → disagreement rate is SUFFICIENT (α scenario)")
            else:
                print("→ Verdict: AUC differs at matched rate → rate is necessary but NOT sufficient (β scenario)")
                print("  Implication: scope theorem must include a *separability* axis beyond rate.")


if __name__ == "__main__":
    main()
