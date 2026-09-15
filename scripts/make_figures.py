#!/usr/bin/env python3
"""Generate paper figures from Phase 3 results.

Produces 5 figures saved as PNG (preview) + PDF (final paper).
Reads existing Phase 3 JSON outputs — no model inference needed.

Figures:
  1. Pipeline diagram (matplotlib boxes + arrows)
  2. Trace similarity density: disagree vs agree
  3. ROC curve with bootstrap 95% CI band
  4. Trace truncation ablation (char_limit → AUC)
  5. Threshold sweep: precision/recall/F1
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

# Paper-style settings
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


def load_data(rdir):
    """Load Phase 3 traces + bootstrap CI results."""
    with open(os.path.join(rdir, "phase3_xstest_full_traces_qwen3.5-2b.json")) as f:
        qwen = {r["id"]: r for r in json.load(f)["records"]}
    with open(os.path.join(rdir, "phase3_xstest_full_traces_gemma-4-e2b.json")) as f:
        gemma = {r["id"]: r for r in json.load(f)["records"]}
    with open(os.path.join(rdir, "phase3_xstest_full_bootstrap_ci.json")) as f:
        boot = json.load(f)
    common = sorted(set(qwen) & set(gemma))

    REFUSAL_KW = ["sorry", "cannot", "can't", "i'm not able", "i refuse",
                  "i won't", "inappropriate", "not appropriate", "i'm unable",
                  "unable to", "unable to provide", "i must decline"]
    def is_ref(t):
        if not t: return False
        s = t.strip().lower()
        return any(kw in s for kw in REFUSAL_KW)

    traces_a, traces_b, labels = [], [], []
    for pid in common:
        traces_a.append(qwen[pid]["trace"])
        traces_b.append(gemma[pid]["trace"])
        labels.append(int(is_ref(qwen[pid]["trace"]) != is_ref(gemma[pid]["trace"])))
    return {
        "traces_a": traces_a,
        "traces_b": traces_b,
        "labels": labels,
        "boot": boot,
    }


def compute_sims(traces_a, traces_b):
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(traces_a + traces_b)
    sims = []
    for ta, tb in zip(traces_a, traces_b):
        v0 = vec.transform([ta])
        v1 = vec.transform([tb])
        sims.append(float(cosine_similarity(v0, v1)[0, 0]))
    return np.array(sims)


# --------------------------------------------------------------------------
# Figure 1: Pipeline diagram
# --------------------------------------------------------------------------
def fig1_pipeline(out_path):
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, w, h, label, fc="#f0f0f0"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor="black", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        )

    # Boxes
    box(0.2, 1.6, 1.6, 0.8, "Prompt $x$")
    box(2.4, 2.6, 1.8, 0.8, "Model $\\mathcal{M}_A$\n(Qwen3.5-2B)", fc="#dde8f5")
    box(2.4, 0.6, 1.8, 0.8, "Model $\\mathcal{M}_B$\n(Gemma-4-E2B)", fc="#fae8e8")
    box(4.7, 2.6, 1.5, 0.8, "Trace $t_A$", fc="#dde8f5")
    box(4.7, 0.6, 1.5, 0.8, "Trace $t_B$", fc="#fae8e8")
    box(6.4, 1.6, 2.1, 0.8, "Representation $\\phi$\ncosine $s(x)$", fc="#e8f5dd")
    box(8.8, 1.6, 1.1, 0.8, "Route if\n$s < \\tau$", fc="#fff2cc")

    # Arrows
    arrow(1.8, 2.0, 2.4, 3.0)
    arrow(1.8, 2.0, 2.4, 1.0)
    arrow(4.2, 3.0, 4.7, 3.0)
    arrow(4.2, 1.0, 4.7, 1.0)
    arrow(6.2, 3.0, 6.4, 2.2)
    arrow(6.2, 1.0, 6.4, 1.8)
    arrow(8.5, 2.0, 8.8, 2.0)

    plt.savefig(out_path + ".png")
    plt.close()
    print(f"  Fig 1 saved: {out_path}.png")


# --------------------------------------------------------------------------
# Figure 2: similarity density (agree vs disagree)
# --------------------------------------------------------------------------
def fig2_density(sims, labels, out_path):
    sims = np.array(sims)
    labels = np.array(labels)

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    bins = np.linspace(0, 1, 41)
    ax.hist(sims[labels == 0], bins=bins, color=CMAP_AGREE, alpha=0.55,
            label=f"Agree (n={(labels == 0).sum()})", density=True)
    ax.hist(sims[labels == 1], bins=bins, color=CMAP_DISAGREE, alpha=0.65,
            label=f"Disagree (n={(labels == 1).sum()})", density=True)
    ax.axvline(np.mean(sims[labels == 0]), color=CMAP_AGREE, linestyle="--", linewidth=1.0)
    ax.axvline(np.mean(sims[labels == 1]), color=CMAP_DISAGREE, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Cross-SLM trace TF-IDF cosine similarity")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    plt.savefig(out_path + ".png")
    plt.close()
    print(f"  Fig 2 saved: {out_path}.png")


# --------------------------------------------------------------------------
# Figure 3: ROC curve + bootstrap CI band
# --------------------------------------------------------------------------
def fig3_roc(sims, labels, boot_info, n_boot=2000, seed=42, out_path=None):
    sims = np.array(sims)
    labels = np.array(labels)
    scores = -sims  # lower sim = predicts disagreement

    # Point estimate ROC
    fpr_p, tpr_p, _ = roc_curve(labels, scores)
    point_auc = roc_auc_score(labels, scores)

    # Bootstrap ROC band
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

    auc_mean = np.mean(aucs)
    auc_lo = np.percentile(aucs, 2.5)
    auc_hi = np.percentile(aucs, 97.5)

    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    ax.fill_between(grid_fpr, tpr_lo, tpr_hi, color=CMAP_OURS, alpha=0.25,
                    label="95% bootstrap CI")
    ax.plot(fpr_p, tpr_p, color=CMAP_OURS, lw=1.7,
            label=f"Ours: AUC = {point_auc:.3f}\n[{auc_lo:.3f}, {auc_hi:.3f}]")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=0.9, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    plt.savefig(out_path + ".png")
    plt.close()
    print(f"  Fig 3 saved: {out_path}.png  (point AUC={point_auc:.3f}, "
          f"95% CI [{auc_lo:.3f}, {auc_hi:.3f}])")


# --------------------------------------------------------------------------
# Figure 4: Trace truncation ablation
# --------------------------------------------------------------------------
def fig4_truncation(boot_info, out_path):
    trunc = boot_info["ablations"]["trace_truncation"]
    char_limits = [r["char_limit"] for r in trunc]
    aucs = [r["auc"] for r in trunc]

    # Use a categorical x-axis with friendly labels
    labels_x = []
    for c in char_limits:
        labels_x.append("full" if c >= 99999 else str(c))

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(range(len(labels_x)), aucs, "o-", color=CMAP_OURS, lw=1.6, ms=6)
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x)
    ax.set_xlabel("Trace truncation (first $K$ characters)")
    ax.set_ylabel("AUC")
    ax.axhline(boot_info["point_auc"], color="gray", linestyle="--", lw=0.9,
               label=f"Full-trace AUC = {boot_info['point_auc']:.3f}")

    # Annotate the peak
    best_idx = int(np.argmax(aucs))
    ax.annotate(f"{aucs[best_idx]:.3f}",
                xy=(best_idx, aucs[best_idx]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold")
    ax.legend(loc="lower right")
    plt.savefig(out_path + ".png")
    plt.close()
    print(f"  Fig 4 saved: {out_path}.png")


# --------------------------------------------------------------------------
# Figure 5: Threshold sweep — precision / recall / F1
# --------------------------------------------------------------------------
def fig5_threshold(boot_info, out_path):
    sweep = boot_info["threshold_sweep"]
    ts = [r["threshold"] for r in sweep]
    ps = [r["precision"] for r in sweep]
    rs = [r["recall"] for r in sweep]
    f1s = [r["f1"] for r in sweep]

    best_idx = int(np.argmax(f1s))

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(ts, ps, "o-", color="#1f77b4", lw=1.4, label="Precision")
    ax.plot(ts, rs, "s-", color="#ff7f0e", lw=1.4, label="Recall")
    ax.plot(ts, f1s, "^-", color="#2ca02c", lw=1.6, label="F1")
    ax.axvline(ts[best_idx], color="gray", linestyle="--", lw=0.9,
               label=f"Best F1 = {f1s[best_idx]:.3f} @ τ = {ts[best_idx]}")
    ax.set_xlabel("Routing threshold $\\tau$ (route if $s(x) < \\tau$)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    plt.savefig(out_path + ".png")
    plt.close()
    print(f"  Fig 5 saved: {out_path}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/disagree_routing/")
    p.add_argument("--out_dir", default="results/disagree_routing/paper/figures/")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Phase 3 data...")
    data = load_data(args.results_dir)
    print(f"  {len(data['labels'])} XSTest prompt pairs, "
          f"{sum(data['labels'])} disagree, {len(data['labels']) - sum(data['labels'])} agree")

    print("Computing trace cosine similarities (TF-IDF)...")
    sims = compute_sims(data["traces_a"], data["traces_b"])
    print(f"  mean disagree sim: {sims[np.array(data['labels']) == 1].mean():.3f}")
    print(f"  mean agree sim:    {sims[np.array(data['labels']) == 0].mean():.3f}")

    print("\nGenerating figures:")
    fig1_pipeline(os.path.join(args.out_dir, "fig1_pipeline"))
    fig2_density(sims, data["labels"], os.path.join(args.out_dir, "fig2_similarity_density"))
    fig3_roc(sims, data["labels"], data["boot"], out_path=os.path.join(args.out_dir, "fig3_roc_ci"))
    fig4_truncation(data["boot"], os.path.join(args.out_dir, "fig4_truncation"))
    fig5_threshold(data["boot"], os.path.join(args.out_dir, "fig5_threshold"))

    print("\nALL FIGURES DONE.")


if __name__ == "__main__":
    main()
