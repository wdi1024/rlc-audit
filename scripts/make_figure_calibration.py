#!/usr/bin/env python3
"""Generate Figure 9: Calibration reliability curve (Appendix A.3).

Outputs PNG to results/disagree_routing/paper/prism_submission/figures/.
"""
import json
import os

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
# Figure 9: Calibration reliability curve
# ─────────────────────────────────────────────────────────────────────────────
print("[Figure 9] calibration reliability curve")
d = json.load(open("results/disagree_routing/calibration_ece_xstest.json"))
rc = d["reliability_curve"]

mean_probs = np.array(rc["bin_mean_prob"])
obs_rates = np.array(rc["bin_obs_rate"])
counts = np.array(rc["bin_count"])
ece10 = d["ece_10bin"]
brier = d["brier"]
brier_baseline = d["brier_baseline"]
brier_skill = d["brier_skill"]
prob_lo, prob_hi = d["prob_range"]
base_rate = d["disagreement_rate"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

# (a) Reliability curve
ax1.plot([0.05, 0.20], [0.05, 0.20], "k--", alpha=0.45, lw=1.2, label="Perfect calibration")
sc = ax1.scatter(mean_probs, obs_rates, s=counts * 1.8, c="#2E86AB",
                 alpha=0.85, edgecolors="black", linewidths=1.0, zorder=3)
# connecting line
ax1.plot(mean_probs, obs_rates, "-", c="#2E86AB", alpha=0.55, lw=1.2, zorder=2)
ax1.axhline(base_rate, ls=":", c="gray", alpha=0.6, label=f"Base rate {base_rate:.3f}")
ax1.axvline(base_rate, ls=":", c="gray", alpha=0.6)

ax1.set_xlim(prob_lo - 0.005, prob_hi + 0.005)
ax1.set_ylim(-0.02, 0.25)
ax1.set_xlabel("Mean predicted probability $\\hat{p}$ (Platt-scaled)")
ax1.set_ylabel("Observed disagreement rate")
ax1.set_title("(a) Reliability curve (10 quantile bins, n=45 each)")
ax1.legend(loc="lower right", fontsize=9, frameon=True)
ax1.grid(True, alpha=0.25, ls="--")

# Annotation: ECE
ax1.text(0.04, 0.96,
         f"ECE (10-bin) = {ece10:.3f}\nBrier = {brier:.3f}\nBrier skill = {brier_skill:+.3f}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", lw=0.8),
         fontsize=9.5)

# (b) Probability range histogram (showing narrow range)
# Reconstruct prob distribution from bin_mean × counts
ax2.bar(mean_probs, counts, width=0.0035, color="#A23B72", edgecolor="black",
        linewidth=0.6, alpha=0.85)
ax2.axvline(base_rate, ls="--", c="gray", alpha=0.7,
            label=f"Base rate {base_rate:.3f}")
ax2.axvline(prob_lo, ls=":", c="green", alpha=0.6)
ax2.axvline(prob_hi, ls=":", c="green", alpha=0.6)
ax2.text(prob_lo, ax2.get_ylim()[1] * 0.92, f"  min {prob_lo:.3f}",
         fontsize=8.5, va="top", color="green")
ax2.text(prob_hi, ax2.get_ylim()[1] * 0.92, f"max {prob_hi:.3f}  ",
         fontsize=8.5, va="top", ha="right", color="green")

ax2.set_xlim(prob_lo - 0.01, prob_hi + 0.01)
ax2.set_xlabel("Predicted probability $\\hat{p}$ (Platt-scaled)")
ax2.set_ylabel("Count of prompts")
ax2.set_title(f"(b) Predicted-probability spread (range {prob_hi - prob_lo:.3f})")
ax2.legend(loc="upper right", fontsize=9, frameon=True)
ax2.grid(True, alpha=0.25, axis="y", ls="--")

plt.tight_layout()
save_fig(fig, "fig9_calibration_reliability")
plt.close(fig)
print("\nDone.")
