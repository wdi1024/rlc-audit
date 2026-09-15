#!/usr/bin/env python3
"""Tier A strengthening figures + sensitivity table.

Outputs:
  figures/fig10_2d_scope_plot.png          — Tier A2 (2D n_dis × sep_d scope plot)
  results/disagree_routing/threshold_sensitivity.json — Tier A1 (sensitivity table)
"""
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

FIGS_DIR = "results/disagree_routing/paper/prism_submission/figures"
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
})


def save_fig(fig, name):
    fig.savefig(f"{FIGS_DIR}/{name}.png", bbox_inches="tight", pad_inches=0.05, dpi=200)
    print(f"  saved: {name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Build the panel: 14 (d, AUC) points + n_disagreement
# ─────────────────────────────────────────────────────────────────────────────
diag = json.load(open("results/disagree_routing/separation_diagnostic.json"))

panel = []
for p in diag.get("combined_panel", []):
    src = p["source"]
    if "XSTest" in src:
        bench = "XSTest"
    elif "OR-Bench" in src:
        bench = "OR-Bench-Hard-1K"
    elif "AdvBench" in src:
        bench = "AdvBench"
    elif "SimpleSafety" in src:
        bench = "SimpleSafetyTests"
    elif "HotpotQA" in src:
        bench = "HotpotQA"
    elif "GSM8K" in src:
        bench = "GSM8K"
    else:
        bench = "Other"
    is_full = "_full" in src
    panel.append(dict(
        source=src, bench=bench, is_full=is_full,
        sep_d=p["sep_d"], auc=p["auc"], n_dis=p["n_dis"], rate=p.get("rate", 0),
    ))

# Try to add HotpotQA point too (out-of-domain validation)
try:
    hot = json.load(open("results/disagree_routing/hotpot_decomposition_report.json"))
    sj = hot["cells"]["Sentence-L12_x_judge"]
    panel.append(dict(
        source="HotpotQA_full", bench="HotpotQA", is_full=True,
        sep_d=sj["sep_d"], auc=sj["auc"], n_dis=sj["n_disagree"],
        rate=hot["judge_disagree"] / hot["n_common"],
    ))
    print("  (added HotpotQA point)")
except Exception:
    print("  (HotpotQA report not available)")

# Try to add GSM8K point too (second out-of-domain validation)
try:
    gsm = json.load(open("results/disagree_routing/gsm8k_decomposition_report.json"))
    sj_g = gsm["cells"]["Sentence-L12_x_judge"]
    panel.append(dict(
        source="GSM8K_full", bench="GSM8K", is_full=True,
        sep_d=sj_g["sep_d"], auc=sj_g["auc"], n_dis=sj_g["n_disagree"],
        rate=gsm["judge_disagree"] / gsm["n_common"],
    ))
    print("  (added GSM8K point)")
except Exception:
    print("  (GSM8K report not available)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: 2D (n_disagree × sep_d) scope plot with identifiable region
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Figure 10] 2D scope plot (n_dis × sep_d)")

# Empirical thresholds from Theorem 1
n0 = 30
d0 = 0.15

fig, ax = plt.subplots(figsize=(8.0, 5.5))

# Shaded "identifiable region": n_dis ≥ n0 AND sep_d > d0
xmin, xmax = 1.5, 1500
ymin, ymax = -1.0, 0.6

# Identifiable region (top-right): both thresholds passed
ax.axvspan(n0, xmax, ymin=(d0 - ymin) / (ymax - ymin), color="green", alpha=0.10, zorder=0)
# Hatch borders
ax.axvline(n0, color="green", ls="--", alpha=0.6, lw=1.2)
ax.axhline(d0, color="green", ls="--", alpha=0.6, lw=1.2)

# Annotate region
ax.text(700, 0.5, "Identifiable region\n$n^{dis} \\geq 30 \\wedge d > 0.15$",
        ha="center", va="center", fontsize=10, color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="darkgreen", lw=0.8, alpha=0.9))

# Plot points by benchmark
bench_colors = {
    "XSTest": "#2E86AB",
    "OR-Bench-Hard-1K": "#A23B72",
    "AdvBench": "#F18F01",
    "SimpleSafetyTests": "#C73E1D",
    "HotpotQA": "#5F4B8B",
    "GSM8K": "#1D7874",
}
bench_markers = {
    "XSTest": "o", "OR-Bench-Hard-1K": "s",
    "AdvBench": "^", "SimpleSafetyTests": "D", "HotpotQA": "P",
    "GSM8K": "X",
}
seen_legend = set()
for p in panel:
    bench = p["bench"]
    color = bench_colors.get(bench, "gray")
    marker = bench_markers.get(bench, "o")
    is_full = p["is_full"]
    size = 160 if is_full else 70
    edge_lw = 1.5 if is_full else 0.7
    alpha = 0.9 if is_full else 0.55
    label = f"{bench} (full)" if (is_full and bench not in seen_legend) else None
    if is_full:
        seen_legend.add(bench)
    ax.scatter(p["n_dis"], p["sep_d"], s=size, c=color, marker=marker,
               edgecolors="black", linewidths=edge_lw, alpha=alpha, zorder=3,
               label=label)

# Annotate AUC near each FULL point
for p in panel:
    if not p["is_full"]:
        continue
    ax.annotate(f"AUC {p['auc']:.3f}", (p["n_dis"], p["sep_d"]),
                xytext=(7, -3), textcoords="offset points", fontsize=8,
                color="black", alpha=0.85)

ax.set_xscale("log")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Number of disagreement events $n^{dis}$ (log scale)")
ax.set_ylabel("Class-conditional separation $d$ (Cohen's d)")
ax.set_title("Theorem 1 visualization: identifiability scope in $(n^{dis}, d)$ space")

# Add threshold annotations
ax.text(n0 * 1.1, ymin + 0.05, f"$n_0 = {n0}$", color="darkgreen", fontsize=9, va="bottom")
ax.text(xmin * 1.5, d0 + 0.02, f"$d_0 = {d0}$", color="darkgreen", fontsize=9, ha="left")

ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95, ncol=1)
ax.grid(True, alpha=0.25, ls="--", which="both")

plt.tight_layout()
save_fig(fig, "fig10_2d_scope_plot")
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Tier A1: Threshold sensitivity analysis
# Question: at what (n_0, d_0) thresholds does the panel cleanly partition
# identifiable from non-identifiable cases?
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Tier A1] Threshold sensitivity analysis")

# Define ground-truth identifiability: AUC is identifiable iff CI lies above 0.5.
# We approximate with: is the FULL benchmark AUC > 0.6 with sufficient n?
# Operationalize: identifiable = (auc >= 0.6 AND n_dis >= 20)
#  NB this ground truth is empirical from our 14-point panel
def is_identifiable(p):
    return p["auc"] is not None and p["auc"] >= 0.6


# Sensitivity: vary (n0, d0) and check confusion matrix on panel
n0_grid = [10, 20, 30, 40, 50]
d0_grid = [0.05, 0.10, 0.15, 0.20, 0.25]

print(f"\n  {'n0':>3s} {'d0':>5s} {'TP':>3s} {'FP':>3s} {'TN':>3s} {'FN':>3s} {'accuracy':>10s}")
sens_results = []
for n0_test in n0_grid:
    for d0_test in d0_grid:
        tp = fp = tn = fn = 0
        for p in panel:
            predicted_id = (p["n_dis"] >= n0_test) and (p["sep_d"] is not None and p["sep_d"] > d0_test)
            actual_id = is_identifiable(p)
            if predicted_id and actual_id:
                tp += 1
            elif predicted_id and not actual_id:
                fp += 1
            elif not predicted_id and not actual_id:
                tn += 1
            else:
                fn += 1
        total = tp + fp + tn + fn
        acc = (tp + tn) / total if total else 0
        sens_results.append(dict(n0=n0_test, d0=d0_test, tp=tp, fp=fp, tn=tn, fn=fn, accuracy=acc))
        marker = " ←" if (n0_test == 30 and d0_test == 0.15) else ""
        print(f"  {n0_test:>3d} {d0_test:>5.2f} {tp:>3d} {fp:>3d} {tn:>3d} {fn:>3d} {acc*100:>9.1f}%{marker}")

# Save
out_path = "results/disagree_routing/threshold_sensitivity.json"
json.dump({
    "panel_n_points": len(panel),
    "ground_truth_rule": "identifiable iff observed AUC >= 0.6",
    "sensitivity": sens_results,
}, open(out_path, "w"), indent=2, ensure_ascii=False)
print(f"\nSaved: {out_path}")

# Pick best threshold
best = max(sens_results, key=lambda r: r["accuracy"])
print(f"\nBest threshold: n0={best['n0']}, d0={best['d0']}, accuracy={best['accuracy']*100:.1f}%")
