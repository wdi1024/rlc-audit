#!/usr/bin/env python3
"""Figure for the semi-synthetic calibration: Delta_dis against the injected rate."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "analysis_results" / "semisynthetic_calibration.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
cols = {"XSTest 450 (primary run)": "#0072B2", "OR-Bench hard 1k": "#D55E00"}
for ax, mode, title in zip(axes, ("A", "B"), ("(a) containment injected", "(b) off-span mismatch injected")):
    for label, res in d["settings"].items():
        xs = [float(r) for r in res[mode]]
        m = [res[mode][r]["d_dis"]["mean"] for r in res[mode]]
        lo = [res[mode][r]["d_dis"]["lo"] for r in res[mode]]; hi = [res[mode][r]["d_dis"]["hi"] for r in res[mode]]
        ax.plot(xs, m, "o-", ms=3, lw=1.2, color=cols[label], label=label.replace(" (primary run)", ""))
        ax.fill_between(xs, lo, hi, color=cols[label], alpha=0.15, lw=0)
        ma = [res[mode][r]["d_auc"]["mean"] for r in res[mode]]
        ax.plot(xs, ma, "--", lw=0.9, color=cols[label], alpha=0.6)
    ax.axhline(0.15, color="#c44e52", ls=":", lw=1); ax.axhline(0.10, color="#dd8452", ls=":", lw=1)
    ax.text(0.505, 0.152, "DIVERGENCE 0.15", fontsize=6.5, color="#c44e52", ha="right", va="bottom")
    ax.text(0.505, 0.102, "CAUTION 0.10", fontsize=6.5, color="#dd8452", ha="right", va="bottom")
    ax.set_title(title, fontsize=9); ax.set_xlabel("injected share of pairs", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(color="#e5e5e5", lw=0.5)
axes[0].set_ylabel(r"$\Delta_{\mathrm{dis}}$ (solid), $\Delta_{\mathrm{AUC}}$ (dashed)", fontsize=8)
axes[0].legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(ROOT / "paper" / "fig_semisynthetic_calibration.pdf"); print("wrote figure")
