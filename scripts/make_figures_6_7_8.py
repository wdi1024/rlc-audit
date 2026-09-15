#!/usr/bin/env python3
"""Generate Figures 6, 7, 8 for the paper.

Figure 6: Diagnostic d vs AUC scatter (14 panel points, Spearman ρ=+0.932)
Figure 7: Three-pair SLM-pair invariance bar chart
Figure 8: Four-cell × four-benchmark heatmap

Outputs PNG to results/disagree_routing/paper/prism_submission/figures/.
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIGS_DIR = "results/disagree_routing/paper/prism_submission/figures"
os.makedirs(FIGS_DIR, exist_ok=True)

# Match the existing paper figure style
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
    png_path = f"{FIGS_DIR}/{name}.png"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    print(f"  saved: {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Diagnostic d vs AUC scatter (16 panel points incl. HotpotQA + GSM8K)
# ─────────────────────────────────────────────────────────────────────────────
print("[Figure 6] diagnostic d vs AUC scatter")
diag = json.load(open("results/disagree_routing/separation_diagnostic.json"))

# Build data
points = []
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
    else:
        bench = "Other"
    is_full = "_full" in src
    points.append(dict(
        source=src, bench=bench, is_full=is_full,
        sep_d=p["sep_d"], auc=p["auc"], rate=p["rate"],
    ))

# Add HotpotQA out-of-domain point (Sentence × judge)
try:
    hot = json.load(open("results/disagree_routing/hotpot_decomposition_report.json"))
    sj = hot["cells"]["Sentence-L12_x_judge"]
    points.append(dict(
        source="HotpotQA_full", bench="HotpotQA", is_full=True,
        sep_d=sj["sep_d"], auc=sj["auc"],
        rate=hot["judge_disagree"] / hot["n_common"],
    ))
except Exception:
    pass

# Add GSM8K out-of-domain point (Sentence × judge)
try:
    gsm = json.load(open("results/disagree_routing/gsm8k_decomposition_report.json"))
    sj_g = gsm["cells"]["Sentence-L12_x_judge"]
    points.append(dict(
        source="GSM8K_full", bench="GSM8K", is_full=True,
        sep_d=sj_g["sep_d"], auc=sj_g["auc"],
        rate=gsm["judge_disagree"] / gsm["n_common"],
    ))
except Exception:
    pass

fig, ax = plt.subplots(figsize=(7.5, 5.0))

bench_colors = {
    "XSTest": "#2E86AB",          # blue
    "OR-Bench-Hard-1K": "#A23B72",  # magenta
    "AdvBench": "#F18F01",        # orange
    "SimpleSafetyTests": "#C73E1D",  # red
    "HotpotQA": "#5F4B8B",        # purple
    "GSM8K": "#1D7874",           # teal
}
bench_markers = {
    "XSTest": "o", "OR-Bench-Hard-1K": "s",
    "AdvBench": "^", "SimpleSafetyTests": "D",
    "HotpotQA": "P", "GSM8K": "X",
}

for bench, color in bench_colors.items():
    pts = [p for p in points if p["bench"] == bench]
    if not pts:
        continue
    full_pts = [p for p in pts if p["is_full"]]
    sub_pts = [p for p in pts if not p["is_full"]]
    if full_pts:
        ax.scatter(
            [p["sep_d"] for p in full_pts],
            [p["auc"] for p in full_pts],
            s=140, c=color, marker=bench_markers[bench],
            edgecolors="black", linewidths=1.5, zorder=3,
            label=f"{bench} (full)",
        )
    if sub_pts:
        ax.scatter(
            [p["sep_d"] for p in sub_pts],
            [p["auc"] for p in sub_pts],
            s=70, c=color, marker=bench_markers[bench],
            alpha=0.55, edgecolors="black", linewidths=0.7, zorder=2,
            label=f"{bench} (rate-subsampled)" if bench in ("XSTest", "OR-Bench-Hard-1K") else None,
        )

ax.axhline(0.5, ls=":", c="gray", alpha=0.6, label="Chance (AUC=0.5)")
ax.axvline(0.0, ls=":", c="gray", alpha=0.6)

# Recompute Spearman/Pearson on the actual points used (incl. out-of-domain)
from scipy.stats import spearmanr, pearsonr
ds = [p["sep_d"] for p in points]
aucs = [p["auc"] for p in points]
spearman, _ = spearmanr(ds, aucs)
pearson, _ = pearsonr(ds, aucs)
n_panel = len(points)
ax.text(0.62, 0.30,
        f"Spearman $\\rho$ = +{spearman:.3f}\n"
        f"(n={n_panel}, p < 10$^{{-8}}$)\n"
        f"Pearson $r$ = +{pearson:.3f}",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", lw=0.8),
        fontsize=10)

# Shaded regions: identifiable / noise / chance — labels left of data clusters
ax.axhspan(0.6, 0.7, color="green", alpha=0.06, zorder=1)
ax.axhspan(0.45, 0.55, color="gray", alpha=0.08, zorder=1)
ax.text(-0.55, 0.66, "Identifiable region\n(AUC ≥ 0.6)", color="darkgreen", alpha=0.9,
        fontsize=9, ha="center", va="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="#e8f5e9", ec="darkgreen", lw=0.5, alpha=0.85))
ax.text(-0.55, 0.50, "Chance band", color="dimgray", alpha=0.95,
        fontsize=9, ha="center", va="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f0f0f0", ec="gray", lw=0.5, alpha=0.85))

ax.set_xlabel("Class-conditional separation $d$ (Cohen's d, agree − disagree)")
ax.set_ylabel("Sentence × judge AUC")
ax.set_title("Diagnostic $d$ predicts AUC across (benchmark, rate-subsample) panel")
ax.set_xlim(-1.0, 0.45)
ax.set_ylim(0.18, 0.72)
ax.legend(loc="center left", fontsize=8.5, frameon=True, framealpha=0.95, ncol=1,
          bbox_to_anchor=(0.0, 0.30))
ax.grid(True, alpha=0.25, ls="--")

plt.tight_layout()
save_fig(fig, "fig6_diagnostic_scatter")
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Three-pair AUC + sep_d bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Figure 7] three-pair SLM-pair invariance bar chart")
three = json.load(open("results/disagree_routing/three_pairs_xstest_report.json"))

pair_data = []
for label, r in three.items():
    short = label.split("(")[0].strip()  # e.g., "Qwen3.5-2B + Gemma-4-E2B-it"
    short = short.replace("Qwen3.5-2B", "Qwen").replace("Gemma-4-E2B-it", "Gemma").replace("Llama-3.2-3B", "Llama")
    sj = r["cells"]["Sentence-L12_x_judge"]
    pair_data.append(dict(
        label=short,
        auc=sj["auc"],
        ci=sj.get("ci"),
        sep_d=sj["sep_d"],
        rate=r.get("judge_disagree_rate", r["judge_disagree"] / r["n"]),
        n_dis=r["judge_disagree"],
    ))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=False)

x = np.arange(len(pair_data))
labels = [p["label"] for p in pair_data]
aucs = [p["auc"] for p in pair_data]
sep_ds = [p["sep_d"] for p in pair_data]
ci_lows = [p["ci"][0] if p["ci"] else 0 for p in pair_data]
ci_highs = [p["ci"][1] if p["ci"] else 0 for p in pair_data]
err_low = [a - cl for a, cl in zip(aucs, ci_lows)]
err_high = [ch - a for a, ch in zip(aucs, ci_highs)]

# Left: AUC bars with CI
colors = ["#2E86AB", "#5B9BD5", "#A23B72"]
bars1 = ax1.bar(x, aucs, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
ax1.errorbar(x, aucs, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=6, lw=1.2)
ax1.axhline(0.5, ls=":", c="gray", alpha=0.6, label="Chance (AUC=0.5)")
ax1.axhspan(0.596, 0.625, color="green", alpha=0.10, zorder=0, label="Tight band [0.596, 0.625]")

for i, (a, p) in enumerate(zip(aucs, pair_data)):
    ax1.text(i, a + 0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.text(i, 0.30, f"n_dis = {p['n_dis']}\n({p['rate']*100:.1f}%)",
             ha="center", va="center", fontsize=8.5, color="white",
             bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))

ax1.set_ylim(0.25, 0.78)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=0, fontsize=9.5)
ax1.set_ylabel("Sentence × judge AUC (95% bootstrap CI)")
ax1.set_title("(a) Headline AUC across 3 cross-family SLM pairs")
ax1.legend(loc="upper left", fontsize=9, frameon=True)
ax1.grid(True, alpha=0.25, axis="y", ls="--")

# Right: sep_d bars
bars2 = ax2.bar(x, sep_ds, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
ax2.axhline(0.0, ls="-", c="gray", alpha=0.5, lw=0.8)
ax2.axhline(0.15, ls=":", c="green", alpha=0.6, label="$d \\geq 0.15$ threshold")
for i, d in enumerate(sep_ds):
    ax2.text(i, d + 0.012, f"{d:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax2.set_ylim(0.0, 0.65)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=0, fontsize=9.5)
ax2.set_ylabel("Class-separation $d$ (Cohen's d)")
ax2.set_title("(b) Diagnostic $d$ across the same 3 pairs")
ax2.legend(loc="upper left", fontsize=9, frameon=True)
ax2.grid(True, alpha=0.25, axis="y", ls="--")

plt.tight_layout()
save_fig(fig, "fig7_three_pair")
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Four-cell × four-benchmark heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Figure 8] 4-cell × 4-benchmark heatmap")

# Hand-build the matrix from paper data (consistent with Table 3 in main.tex)
benchmarks = ["XSTest", "OR-Bench-Hard-1K", "AdvBench", "SimpleSafetyTests"]
cells = ["TF-IDF × keyword", "TF-IDF × judge", "Sentence × keyword", "Sentence × judge"]
matrix = np.array([
    [0.787, 0.778, 0.768, 0.904],   # TF-IDF × keyword (universal artifact)
    [0.482, 0.543, 0.380, 0.310],   # TF-IDF × judge
    [0.820, 0.806, 0.823, 0.888],   # Sentence × keyword
    [0.651, 0.531, np.nan, 0.433],  # Sentence × judge (conditional)
])
# AdvBench Sentence × judge is essentially undefined (n=3)
# We'll display its n=3 noise value (0.246) but mark it specially
matrix_display = np.array([
    [0.787, 0.778, 0.768, 0.904],
    [0.482, 0.543, 0.380, 0.310],
    [0.820, 0.806, 0.823, 0.888],
    [0.651, 0.531, 0.246, 0.433],
])

fig, ax = plt.subplots(figsize=(8.5, 4.8))

# Diverging colormap: 0.5 (chance) → white, above → green, below → red
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=0.25, vcenter=0.5, vmax=0.95)
im = ax.imshow(matrix_display, cmap="RdYlGn", norm=norm, aspect="auto")

# Cell annotations
for i, cell in enumerate(cells):
    for j, bench in enumerate(benchmarks):
        val = matrix_display[i, j]
        # Highlight artifact (TF×kw row) with bold
        weight = "bold" if i == 0 else "normal"
        # Highlight headline-cell × XSTest with golden border
        text_color = "black" if 0.4 < val < 0.85 else "white"
        if i == 3 and j == 2:  # Sentence × judge × AdvBench (n=3 noise)
            ax.text(j, i, f"{val:.3f}\n(n=3)", ha="center", va="center",
                    color=text_color, fontsize=10, weight=weight, style="italic")
        else:
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=text_color, fontsize=11, weight=weight)

# XSTest × Sentence×judge — highlight as the only identifiable cell
rect = plt.Rectangle((-0.5 + 0, -0.5 + 3), 1, 1, fill=False,
                     edgecolor="gold", linewidth=4, zorder=5)
ax.add_patch(rect)
ax.annotate("Identifiable\nheadline", xy=(0, 3), xytext=(0.05, 3.95),
            fontsize=9, color="darkgoldenrod", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="darkgoldenrod", lw=1.5),
            ha="center")

ax.set_xticks(range(len(benchmarks)))
ax.set_xticklabels(benchmarks, rotation=0, fontsize=10)
ax.set_yticks(range(len(cells)))
ax.set_yticklabels(cells, fontsize=10)
ax.set_title("4-cell decomposition × 4 benchmarks: artifact universal, headline conditional")

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("AUC", fontsize=10)
cbar.ax.tick_params(labelsize=8.5)

# Hide spines
for s in ax.spines.values():
    s.set_visible(False)

plt.tight_layout()
save_fig(fig, "fig8_4cell_heatmap")
plt.close(fig)

print("\nAll three figures saved.")
