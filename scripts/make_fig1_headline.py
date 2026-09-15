#!/usr/bin/env python3
"""Figure 1: headline RLC dissociation.

Panel A: proxy vs semantic AUC for the primary raw-prefix contract.
Panel B: routed-queue composition at B=55 for the raw-prefix score and the
two audit-guided revisions (final-span TF-IDF, composite witness).

Numbers match the paper: AUC 0.985/0.592, pair-level kappa(z,y) 0.024. Queues at
B=55: primary raw run 6/37/12; tagged clean re-audit run 11/17/27 (matched raw
baseline), 20/33/2, 28/25/2 (analysis_results/rlc_composite_router_summary.json,
rebuttal_budget_sweep_ci.json, submission_robustness.json).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "figures/fig1_rlc_headline.pdf"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

# One panel only.  The proxy-vs-construct AUC contrast that used to sit beside this
# is two bars carrying two numbers (0.985 and 0.592); it is reported in the text and
# in Table 2, where it costs one row instead of half a figure.  The space is better
# spent on the span dose-response, which is the paper's central result.
fig, ax_b = plt.subplots(figsize=(5.4, 1.55))

# Panel B: routed queue composition at B=55.
# Top bar: primary raw-generation run (XSTest 450). Bottom three bars: tagged
# clean re-audit run (a distinct contract; matched raw-prefix baseline 11/55).
# The two runs are different contracts (the generation protocol is a contract
# field), so they are drawn as separated groups with their own band and heading
# rather than as one four-bar progression.
# The headline figure shows one contract only.  Revision bars live in Table 6 and
# Appendix F, so panel (b) is not asked to carry two generation protocols at once.
rows = [
    ("Raw-prefix score\n(primary contract)", 6, 37, 12),
]
colors = {"sem": "#2e7d32", "refuse": "#c44e52", "comply": "#c9c9c9"}
y_pos = [0.0]
for (label, sem, refuse, comply), y in zip(rows, y_pos):
    ax_b.barh(y, sem, color=colors["sem"], height=0.58)
    ax_b.barh(y, refuse, left=sem, color=colors["refuse"], height=0.58)
    ax_b.barh(y, comply, left=sem + refuse, color=colors["comply"], height=0.58)
    ax_b.text(sem / 2, y, str(sem), ha="center", va="center",
              color="white", fontsize=9, fontweight="bold")
    ax_b.text(sem + refuse / 2, y, str(refuse), ha="center", va="center",
              color="white", fontsize=9)
    if comply > 4:
        ax_b.text(sem + refuse + comply / 2, y, str(comply),
                  ha="center", va="center", color="#333333", fontsize=9)

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([r[0] for r in rows], fontsize=8)
ax_b.set_xlim(0, 55)
ax_b.set_ylim(-0.6, 0.6)
ax_b.set_xlabel("routed queue at budget $B=55$")



handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors["sem"]),
    plt.Rectangle((0, 0), 1, 1, color=colors["refuse"]),
    plt.Rectangle((0, 0), 1, 1, color=colors["comply"]),
]
ax_b.legend(
    handles,
    ["semantic disagreement (target)", "agreed refusal", "agreed compliance"],
    loc="lower center", fontsize=7.5, frameon=False, ncol=3,
    bbox_to_anchor=(0.5, 1.02), columnspacing=1.4, handlelength=1.4,
)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
