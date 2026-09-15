#!/usr/bin/env python3
"""Regenerate Figures 2, 3, 5 for the 8-page ICLR D&B version.

Reviewer-flagged issues:
  - Figure 2: axis label says TF-IDF but headline is sentence-embedding;
    n_disagree (25) does not match judge-validated 46.
  - Figure 3: ROC legend shows AUC 0.787 (artifact cell), but caption claims
    headline (0.651) sentence × judge. Re-render against judge labels.
  - Figure 5: Caption mentions "artifact vs semantic" comparison but the plot
    only shows the TF-IDF artifact curve. Add the sentence × judge curve.
"""
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
})

CMAP_AGREE = "#1f77b4"
CMAP_DISAGREE = "#d62728"
CMAP_OURS = "#2ca02c"
CMAP_ARTIFACT = "#d62728"

REFUSAL_KW = ["sorry", "cannot", "can't", "i'm not able", "i refuse",
              "i won't", "inappropriate", "not appropriate", "i'm unable",
              "unable to", "unable to provide", "i must decline"]


def is_kw_refusal(text):
    if not text:
        return False
    s = text.strip().lower()
    return any(kw in s for kw in REFUSAL_KW)


def load_phase3():
    rdir = "results/disagree_routing"
    qwen_t = json.load(open(f"{rdir}/phase3_xstest_full_traces_qwen3.5-2b.json"))["records"]
    gemma_t = json.load(open(f"{rdir}/phase3_xstest_full_traces_gemma-4-e2b.json"))["records"]
    qwen_j = json.load(open(f"{rdir}/phase3_xstest_full_judge_qwen3.5-2b_anthropic_claude-haiku-4-5-20251001.json"))["records"]
    gemma_j = json.load(open(f"{rdir}/phase3_xstest_full_judge_gemma-4-e2b_anthropic_claude-haiku-4-5-20251001.json"))["records"]

    qt = {r["id"]: r for r in qwen_t}
    gt = {r["id"]: r for r in gemma_t}
    qj = {r["id"]: r.get("refusal_judge") for r in qwen_j if r.get("refusal_judge") is not None}
    gj = {r["id"]: r.get("refusal_judge") for r in gemma_j if r.get("refusal_judge") is not None}

    common = sorted(set(qt) & set(gt) & set(qj) & set(gj))
    return common, qt, gt, qj, gj


def auc_with_ci(labels, scores, n_boot=2000, seed=42):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(set(labels)) < 2:
        return None, None, None
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
    boots = np.array(boots)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return point, (ci_lo, ci_hi), boots


def fig2_density_corrected(sims, labels, out_path):
    sims = np.asarray(sims)
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    bins = np.linspace(min(sims.min(), 0.0), max(sims.max(), 1.0), 41)
    n_agree = (labels == 0).sum()
    n_dis = (labels == 1).sum()
    ax.hist(sims[labels == 0], bins=bins, color=CMAP_AGREE, alpha=0.55,
            label=f"Agree (n={n_agree})", density=True)
    ax.hist(sims[labels == 1], bins=bins, color=CMAP_DISAGREE, alpha=0.65,
            label=f"Disagree (n={n_dis})", density=True)
    ax.axvline(np.mean(sims[labels == 0]), color=CMAP_AGREE, linestyle="--", linewidth=1.0)
    ax.axvline(np.mean(sims[labels == 1]), color=CMAP_DISAGREE, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Cross-SLM trace sentence-embedding cosine similarity")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    ax.set_xlim(bins[0], bins[-1])
    plt.savefig(out_path + ".png")
    plt.close()
    print(f"  Fig 2 saved: {out_path}.png  agree={n_agree}, disagree={n_dis}")


def fig3_roc_corrected(sims, labels, out_path, n_boot=2000, seed=42):
    sims = np.asarray(sims)
    labels = np.asarray(labels)
    scores = -sims

    fpr_p, tpr_p, _ = roc_curve(labels, scores)
    point_auc = roc_auc_score(labels, scores)

    rng = np.random.RandomState(seed)
    n = len(scores)
    grid_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(set(labels[idx])) < 2:
            continue
        try:
            f, t, _ = roc_curve(labels[idx], scores[idx])
            interp = np.interp(grid_fpr, f, t)
            interp[0] = 0.0
            tprs.append(interp)
            aucs.append(roc_auc_score(labels[idx], scores[idx]))
        except Exception:
            pass
    tprs = np.array(tprs)
    tpr_lo = np.percentile(tprs, 2.5, axis=0)
    tpr_hi = np.percentile(tprs, 97.5, axis=0)
    auc_lo = np.percentile(aucs, 2.5)
    auc_hi = np.percentile(aucs, 97.5)

    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    ax.fill_between(grid_fpr, tpr_lo, tpr_hi, color=CMAP_OURS, alpha=0.25,
                    label="95% bootstrap CI")
    ax.plot(fpr_p, tpr_p, color=CMAP_OURS, lw=1.7,
            label=f"Sentence $\\times$ judge:\nAUC = {point_auc:.3f}\n[{auc_lo:.3f}, {auc_hi:.3f}]")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=0.9, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    plt.savefig(out_path + ".png")
    plt.close()
    print(f"  Fig 3 saved: {out_path}.png  AUC={point_auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]")


def truncation_curve(traces_a, traces_b, labels_kw, labels_judge, char_limits):
    """For each char_limit, compute (TF-IDF × keyword) and (sentence × judge) AUC."""
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    auc_artifact = []
    auc_semantic = []
    for K in char_limits:
        if K >= 99999:
            ta_k = list(traces_a)
            tb_k = list(traces_b)
        else:
            ta_k = [t[:K] for t in traces_a]
            tb_k = [t[:K] for t in traces_b]

        # TF-IDF
        try:
            vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
            vec.fit(ta_k + tb_k)
            sims_tfidf = []
            for a, b in zip(ta_k, tb_k):
                v0 = vec.transform([a])
                v1 = vec.transform([b])
                sims_tfidf.append(float(cosine_similarity(v0, v1)[0, 0]))
            sims_tfidf = np.asarray(sims_tfidf)
            scores_a = -sims_tfidf
            auc_a = roc_auc_score(labels_kw, scores_a) if len(set(labels_kw)) > 1 else 0.5
        except Exception:
            auc_a = 0.5

        # Sentence-embedding
        emb_a = encoder.encode(ta_k, show_progress_bar=False, normalize_embeddings=True)
        emb_b = encoder.encode(tb_k, show_progress_bar=False, normalize_embeddings=True)
        sims_sent = (emb_a * emb_b).sum(axis=1)
        scores_s = -sims_sent
        auc_s = roc_auc_score(labels_judge, scores_s) if len(set(labels_judge)) > 1 else 0.5

        auc_artifact.append(auc_a)
        auc_semantic.append(auc_s)
        print(f"    K={K:>6}: TF-IDF×kw AUC={auc_a:.3f}, sent×judge AUC={auc_s:.3f}")
    return auc_artifact, auc_semantic


def fig5_truncation_corrected(char_limits, auc_artifact, auc_semantic, out_path):
    labels_x = ["full" if c >= 99999 else str(c) for c in char_limits]
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    xs = np.arange(len(labels_x))
    ax.plot(xs, auc_artifact, "o-", color=CMAP_ARTIFACT, lw=1.6, ms=6,
            label="TF-IDF $\\times$ keyword (artifact)")
    ax.plot(xs, auc_semantic, "s-", color=CMAP_OURS, lw=1.6, ms=6,
            label="Sentence $\\times$ judge (semantic)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels_x)
    ax.set_xlabel("Trace truncation (first $K$ characters)")
    ax.set_ylabel("AUC")
    ax.axhline(0.5, color="gray", linestyle=":", lw=0.8, label="Chance")

    best_a = int(np.argmax(auc_artifact))
    best_s = int(np.argmax(auc_semantic))
    ax.annotate(f"{auc_artifact[best_a]:.3f}",
                xy=(best_a, auc_artifact[best_a]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold", color=CMAP_ARTIFACT)
    ax.annotate(f"{auc_semantic[best_s]:.3f}",
                xy=(best_s, auc_semantic[best_s]),
                xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold", color=CMAP_OURS)
    ax.set_ylim(0.40, 0.92)
    ax.legend(loc="upper right", fontsize=8.5)
    plt.savefig(out_path + ".png")
    plt.close()
    print(f"  Fig 5 saved: {out_path}.png")


def main():
    out_dir = "results/disagree_routing/paper/figures"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Phase 3 traces + judge labels...")
    common, qt, gt, qj, gj = load_phase3()
    print(f"  {len(common)} common IDs (with judge labels)")

    traces_a = [qt[i]["trace"] for i in common]
    traces_b = [gt[i]["trace"] for i in common]

    labels_kw = np.array([
        int(is_kw_refusal(qt[i]["trace"]) != is_kw_refusal(gt[i]["trace"])) for i in common
    ])
    labels_judge = np.array([int(qj[i] != gj[i]) for i in common])
    print(f"  Keyword disagreement: {labels_kw.sum()}/{len(common)}")
    print(f"  Judge disagreement:   {labels_judge.sum()}/{len(common)}")

    print("\nEncoding sentence embeddings (MiniLM-L12)...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    emb_a = encoder.encode(traces_a, show_progress_bar=False, normalize_embeddings=True)
    emb_b = encoder.encode(traces_b, show_progress_bar=False, normalize_embeddings=True)
    sims_sent = (emb_a * emb_b).sum(axis=1)
    print(f"  sentence cosine: mean={sims_sent.mean():.3f}, std={sims_sent.std():.3f}")

    auc_check, ci_check, _ = auc_with_ci(labels_judge, -sims_sent)
    print(f"  Headline check: sentence×judge AUC={auc_check:.3f}  CI [{ci_check[0]:.3f}, {ci_check[1]:.3f}]")

    print("\nGenerating corrected figures...")
    fig2_density_corrected(sims_sent, labels_judge, os.path.join(out_dir, "fig2_similarity_density"))
    fig3_roc_corrected(sims_sent, labels_judge, os.path.join(out_dir, "fig3_roc_ci"))

    char_limits = [50, 100, 200, 500, 1000, 99999]
    print("\nComputing truncation curves...")
    auc_artifact, auc_semantic = truncation_curve(
        traces_a, traces_b, labels_kw, labels_judge, char_limits)
    fig5_truncation_corrected(char_limits, auc_artifact, auc_semantic,
                              os.path.join(out_dir, "fig4_truncation"))

    # Mirror to prism_submission/figures
    prism_dir = "results/disagree_routing/paper/prism_submission/figures"
    os.makedirs(prism_dir, exist_ok=True)
    for f in ["fig2_similarity_density", "fig3_roc_ci", "fig4_truncation"]:
        for ext in ["pdf", "png"]:
            src = os.path.join(out_dir, f"{f}.{ext}")
            dst = os.path.join(prism_dir, f"{f}.{ext}")
            if os.path.exists(src):
                import shutil
                shutil.copy(src, dst)

    print("\nDone. Figures regenerated and mirrored to prism_submission/.")


if __name__ == "__main__":
    main()
