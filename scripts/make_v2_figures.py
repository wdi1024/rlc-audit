#!/usr/bin/env python3
"""Figures for the v2 (discovery-first) restructure. All numbers come from
audited artifacts: searchandlearn_beam_audit.json, prm800k_audit.json,
prefix_span_correctness_probe.json.

Outputs: paper/fig2_deployed_spans.pdf, paper/fig3_span_dose.pdf,
paper/fig_verdict_tree.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"
OUT = ROOT / "paper"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})
BLUE, RED, ORANGE, GRAY = "#4878a8", "#c44e52", "#dd8452", "#8a8a8a"


# ---------------------------------------------------------------- figure 2 ---
def fig2():
    prm = json.load(open(AR / "prm800k_audit.json"))
    beam = json.load(open(AR / "searchandlearn_beam_audit.json"))["arms"]

    # One panel.  The selection-harm panel that used to sit beside this plotted
    # harm rates the text explicitly declares non-comparable across contracts
    # (their pools have different base rates), so drawing them on shared axes
    # invited exactly the comparison we disclaim; those numbers are in the text.
    fig, ax_a = plt.subplots(figsize=(5.0, 2.3))
    # (a) within-problem AUC of the deployed score vs span fraction
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    prm_curve = [prm["auc"][k]["within"] for k in ("first", "0.25", "0.5", "0.75", "1.0")]
    ax_a.plot(xs, prm_curve, "-o", color=BLUE, ms=4.5, lw=2, label="PRM800K (BoN, full-span)")
    keys = ("first", "0.25", "0.5", "0.75", "1.0")
    for kind, color, marker in (("beam", RED, "s"), ("dvts", ORANGE, "^"), ("bon", "#2e7d32", "D")):
        arms = [a for a in beam if a["kind"] == kind]
        curve = [float(np.mean([a["auc"][k]["within"] for a in arms])) for k in keys]
        label = {"beam": "beam search", "dvts": "DVTS", "bon": "best-of-$n$"}[kind]
        ax_a.plot(xs, curve, linestyle="--", marker=marker,
                  color=color, ms=5, lw=1.4, label=label)
    ax_a.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax_a.text(0.985, 0.492, "chance", fontsize=7.5, color=GRAY, va="top", ha="right")
    ax_a.annotate("pruning decides here", xy=(0.0, 0.47), xytext=(0.13, 0.40),
                  fontsize=7.5, color="#333333",
                  arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
    ax_a.set_xlabel("fraction of solution scored")
    ax_a.set_ylabel("within-problem AUC")
    ax_a.set_ylim(0.35, 1.10)
    ax_a.set_xticks(xs, ["first\nstep", "25%", "50%", "75%", "full"])
    ax_a.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2, handlelength=1.6,
                columnspacing=1.1, borderaxespad=0.15, labelspacing=0.25)
    ax_a.set_title("Score vs. correctness, by span scored", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_deployed_spans.pdf")
    print("wrote fig2")


# ---------------------------------------------------------------- figure 3 ---
def fig3():
    """Span dose-response, drawn as the two AUCs themselves.

    An earlier version plotted the signed gap and overlaid the orientation-robust
    gap as crosses at the spans where the sign flips, which put two different
    y-quantities on one axis.  Plotting AUC(s,z) and AUC(s,y) against the span
    removes that: the y-axis is one quantity, the gap is the vertical distance,
    and the orientation reversal is visible directly as a construct curve below
    chance.  The signed and orientation-robust gaps, with intervals, are in the
    appendix sweep table.
    """
    sweep = json.load(open(AR / "span_closure_fine_sweep.json"))
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(8.6, 2.9), gridspec_kw={"width_ratios": [1.6, 1.0]})

    series = (("HotpotQA", RED, "s"), ("GSM8K", BLUE, "o"))

    for name, color, marker in series:
        rows = [r for r in sweep[name]["rows"] if "gap" in r]
        x = [r["span_chars"] or 1800 for r in rows]
        pz = [r["auc_proxy"] for r in rows]
        py = [r["auc_semantic"] for r in rows]
        ax_a.plot(x, pz, marker + "-", color=color, ms=4, lw=1.6,
                  label=f"{name}: $AUC(s,z)$, proxy")
        ax_a.plot(x, py, marker + "--", color=color, ms=4, lw=1.4, mfc="white",
                  label=f"{name}: $AUC(s,y)$, construct")
        # Shade only where the construct is above chance.  Where it falls below,
        # the audit re-orients onto Delta_|.| and the raw vertical distance is no
        # longer the quantity the verdict reads, so shading it would show one
        # estimand while the text uses another.
        import numpy as _np
        above = _np.array(py) >= 0.5
        ax_a.fill_between(x, pz, py, where=above, color=color, alpha=0.10, lw=0,
                          interpolate=True)
        if (~above).any():
            xs_rev = [xi for xi, a in zip(x, above) if not a]
            ax_a.axvspan(min(xs_rev) * 0.88, max(xs_rev) * 1.13, color="#bbbbbb",
                         alpha=0.13, lw=0, zorder=0)
            ax_a.text((min(xs_rev) * max(xs_rev)) ** 0.5, 0.303,
                      "score reversed here: read on $\\Delta_{|\\cdot|}$", fontsize=6.0,
                      color="#555555", ha="center", va="bottom")

    # the one contract the paper reads as a demonstrated failure: annotate its gap
    hp = [r for r in sweep["HotpotQA"]["rows"] if r["span_chars"] == 50][0]
    ax_a.annotate("", xy=(50, hp["auc_proxy"]), xytext=(50, hp["auc_semantic"]),
                  arrowprops=dict(arrowstyle="<->", lw=1.0, color="#333333"))
    ax_a.annotate("$+%.3f$\n[%.3f, %.3f]" % (hp["gap"], hp["gap_ci"][0], hp["gap_ci"][1]),
                  xy=(56, 0.60), fontsize=7.0, color="#333333", ha="left", va="center")
    ax_a.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax_a.text(1750, 0.487, "chance", fontsize=7.0, color=GRAY, ha="right", va="top")
    ax_a.set_xscale("log")
    # the full-trace point is not a character count, so it is drawn past a break
    # rather than placed at a fictitious coordinate on the log axis
    ax_a.set_xticks([50, 100, 200, 400, 800, 1400],
                    ["50", "100", "200", "400", "800", "1400"])
    ax_a.axvline(1600, color=GRAY, lw=0.8, ls=(0, (2, 3)))
    ax_a.text(1800, 0.315, "full\ntrace", fontsize=6.8, color="#555555",
              ha="center", va="bottom")
    ax_a.set_xlabel("span supplied to the scorer (characters, log scale)")
    ax_a.set_ylabel("AUC of the score")
    ax_a.set_ylim(0.30, 0.83)
    ax_a.legend(frameon=False, fontsize=6.6, loc="upper right", ncol=2,
                labelspacing=0.25, columnspacing=1.0, handlelength=1.8)
    ax_a.set_title("(a) Both AUCs against the span the score and proxy read",
                   fontsize=8.5)

    # (b) where the answer actually arrives
    for i, (name, color, marker) in enumerate(series):
        r = sweep[name]
        med, q25, q75 = r["arrival_median"], r["arrival_q25"], r["arrival_q75"]
        ax_b.plot([q25, q75], [i, i], color=color, lw=3.2, solid_capstyle="round",
                  alpha=0.45)
        ax_b.plot([med], [i], marker, color=color, ms=7, zorder=3)
        ax_b.annotate(f"median {med:.0f}", xy=(med, i), xytext=(med, i + 0.20),
                      fontsize=7.5, color=color, ha="center")
    ax_b.set_yticks([0, 1], [n for n, _, _ in series])
    ax_b.set_ylim(-0.55, 1.62)
    ax_b.set_xscale("log")
    ax_b.set_xticks([100, 300, 1000], ["100", "300", "1000"])
    ax_b.set_xlabel("gold answer's first appearance (chars, log scale)")
    ax_b.set_title("(b) Where the answer arrives", fontsize=8.5)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_span_dose.pdf")
    print("wrote fig3")


# ------------------------------------------------------------- verdict tree ---
def fig_tree():
    """The verdict function as three stages, with five outcomes.

    The previous version drew seven exits and, after the rule gained a
    construct-orientation branch, no longer showed where that branch led: the
    headline verdict was absent from the headline figure. It also drew three dashed
    boxes while the text said two. Redrawn: one question box per stage-question,
    five distinct verdicts, and the dashed count stated by the figure itself so text
    and drawing cannot drift apart again.
    """
    fig, ax = plt.subplots(figsize=(8.9, 4.15))
    ax.set_xlim(-3, 152)
    ax.set_ylim(-19, 84)
    ax.axis("off")

    W, H, GAP = 28.0, 10.0, 6.4
    VW = 28.0
    GRAY, SLATE, RED2, ORANGE2, GREEN = "#7f7f7f", "#5b6b7a", "#c44e52", "#dd8452", "#2e7d32"

    def qbox(x, y, text, dashed):
        ax.add_patch(plt.Rectangle((x, y - H / 2), W, H, fill=True,
                                   fc="#efe7f4" if dashed else "#f2f2f2",
                                   ec="#9467bd" if dashed else "#555555",
                                   lw=1.0, ls=(0, (3, 2)) if dashed else "-"))
        ax.text(x + W / 2, y, text, ha="center", va="center", fontsize=9.3)

    def vbox(xc, y, text, colour, w=VW):
        ax.add_patch(plt.Rectangle((xc - w / 2, y - 3.8), w, 7.6, fill=True,
                                   fc=colour, ec="none"))
        ax.text(xc, y, text, ha="center", va="center", fontsize=9.1,
                color="white", fontweight="bold")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555"))

    stages = [
        ("1. can the contract be decided?", 66.0, 54.0,
         [("label a function of\nreleased evidence?", True, "no"),
          ("a required field missing\nfrom released files?", True, "yes"),
          ("fails min-count, complement\nor resolution?", False, "yes")],
         "UNDECIDABLE", GRAY),
    ]
    for title, qy, vy, qs, verdict, colour in stages:
        ax.text(-3.0, qy + H / 2 + 2.6, title, fontsize=9.3, color="#333333",
                style="italic", ha="left")
        xs = [i * (W + GAP) for i in range(len(qs))]
        for x, (q, dashed, lab) in zip(xs, qs):
            qbox(x, qy, q, dashed)
            arrow(x + W / 2, qy - H / 2, x + W / 2, vy + 3.8)
            ax.text(x + W / 2 + 1.2, (qy - H / 2 + vy + 3.8) / 2, lab, fontsize=8.7,
                    color="#555555", ha="left", va="center")
            vbox(x + W / 2, vy, verdict, colour)
        for x0, x1 in zip(xs, xs[1:]):
            arrow(x0 + W, qy, x1, qy)
            ax.text((x0 + W + x1) / 2, qy + 1.6, "no", fontsize=8.7, color="#555555",
                    ha="center", va="bottom")
        xe = xs[-1] + W
        arrow(xe, qy, xe + 7.0, qy)
        ax.text(xe + 8.0, qy, "no,\nto stage 2", fontsize=8.5, color="#555555",
                ha="left", va="center")

    # stage 2: is the score evidence about the construct?  (matches Algorithm 1
    # step 2: "at chance" must be SHOWN by equivalence -- 90% CI inside
    # [0.45, 0.55] -> NO DEMONSTRATED CONSTRUCT RANKING (off-span proxy still ranked);
    # an interval wholly below 1/2 is a reversed score -> re-orient with the
    # |.|-gap and continue; covering 1/2 without fitting the band -> UNDECIDABLE.)
    qy2, vy2 = 34.0, 22.0
    ax.text(-3.0, qy2 + H / 2 + 4.6, "2. is the score evidence about the construct?",
            fontsize=9.3, color="#333333", style="italic", ha="left")
    xs2 = [i * (W + GAP) for i in range(4)]
    qbox(xs2[0], qy2, "shown at chance?\n(90% CI in [0.45, 0.55])", False)
    arrow(xs2[0] + W, qy2, xs2[1], qy2)
    ax.text((xs2[0] + W + xs2[1]) / 2, qy2 + 1.6, "yes", fontsize=8.7, color="#555555",
            ha="center", va="bottom")
    # the branch the verdict function actually takes: which exit an at-chance arm gets
    qbox(xs2[1], qy2, "off-span proxy\nstill ranked?", False)
    arrow(xs2[1] + W / 2, qy2 - H / 2, xs2[1] + W / 2, vy2 + 3.8)
    ax.text(xs2[1] + W / 2 + 1.2, (qy2 - H / 2 + vy2 + 3.8) / 2, "yes / no", fontsize=7.0,
            color="#555555", ha="left", va="center")
    hw2 = VW / 2 - 0.6
    ax.add_patch(plt.Rectangle((xs2[1] + W / 2 - VW / 2, vy2 - 3.8), hw2, 7.6, fill=True,
                               fc=SLATE, ec="none"))
    ax.text(xs2[1] + W / 2 - VW / 2 + hw2 / 2, vy2, "NO\nDEMONSTRATED\nCONSTRUCT\nRANKING", ha="center", va="center",
            fontsize=4.6, color="white", fontweight="bold")
    ax.add_patch(plt.Rectangle((xs2[1] + W / 2 + 0.6, vy2 - 3.8), hw2, 7.6, fill=True,
                               fc="#7b5aa6", ec="none"))
    ax.text(xs2[1] + W / 2 + 0.6 + hw2 / 2, vy2, "CONTAIN-\nMENT", ha="center", va="center",
            fontsize=6.8, color="white", fontweight="bold")
    ax.text(xs2[1] + W / 2, vy2 - 5.6, "reported gap $\\geq 0.15$ / else UNDECIDABLE",
            ha="center", va="center", fontsize=6.2, color="#555555")
    # not at chance: route the 'no' of box 1 over the top into box 3, so the line
    # never crosses the reversed-score box below
    top = qy2 + H / 2 + 1.8
    ax.plot([xs2[0] + W / 2, xs2[0] + W / 2], [qy2 + H / 2, top], color="#555555", lw=1.0)
    ax.plot([xs2[0] + W / 2, xs2[2] + W / 2], [top, top], color="#555555", lw=1.0)
    arrow(xs2[2] + W / 2, top, xs2[2] + W / 2, qy2 + H / 2)
    ax.text((xs2[0] + xs2[2]) / 2 + W / 2, top - 0.4, "no", fontsize=8.2, color="#555555",
            ha="center", va="top")
    qbox(xs2[2], qy2, "interval wholly\nbelow 1/2?", False)
    arrow(xs2[2] + W / 2, qy2 - H / 2, xs2[2] + W / 2, vy2 + 3.8)
    ax.text(xs2[2] + W / 2 + 1.2, (qy2 - H / 2 + vy2 + 3.8) / 2, "yes", fontsize=8.7,
            color="#555555", ha="left", va="center")
    ax.add_patch(plt.Rectangle((xs2[2] + W / 2 - VW / 2, vy2 - 3.8), VW, 7.6, fill=True,
                               fc="white", ec=GREEN, lw=1.1))
    ax.text(xs2[2] + W / 2, vy2, "score reversed:\nreport $\\Delta_{|\\cdot|}$, to stage 3",
            ha="center", va="center", fontsize=8.7, color=GREEN)
    arrow(xs2[2] + W, qy2, xs2[3], qy2)
    ax.text((xs2[2] + W + xs2[3]) / 2, qy2 + 1.6, "no", fontsize=8.7, color="#555555",
            ha="center", va="bottom")
    qbox(xs2[3], qy2, "interval excludes 1/2?\n(oriented)", False)
    arrow(xs2[3] + W / 2, qy2 - H / 2, xs2[3] + W / 2, vy2 + 3.8)
    ax.text(xs2[3] + W / 2 + 1.2, (qy2 - H / 2 + vy2 + 3.8) / 2, "no", fontsize=8.7,
            color="#555555", ha="left", va="center")
    vbox(xs2[3] + W / 2, vy2, "UNDECIDABLE", GRAY)
    xe2 = xs2[3] + W
    arrow(xe2, qy2, xe2 + 7.0, qy2)
    ax.text(xe2 + 8.0, qy2, "yes,\nto stage 3", fontsize=8.5, color="#555555",
            ha="left", va="center")

    # stage 3: what the surviving gap says.  The below-null exit splits: a large
    # certificate gap that the disjoint control erased was containment (the
    # contract's span must move); a small one was never materially inflated.
    PURPLE2 = "#7b5aa6"
    QY3, VY3 = 2.0, -10.0
    LY3 = (QY3 - H / 2 + VY3 + 3.8) / 2
    ax.text(-3.0, QY3 + H / 2 + 2.6, "3. what does the surviving gap say?", fontsize=9.3,
            color="#333333", style="italic", ha="left")
    xs = [i * (W + GAP) for i in range(3)]
    # box 2: below the contract's null -> containment or aligned by certificate size
    qbox(xs[0], QY3, "gap below this\ncontract's null?", False)
    arrow(xs[0] + W / 2, QY3 - H / 2, xs[0] + W / 2, VY3 + 3.8)
    ax.text(xs[0] + W / 2 + 1.2, LY3, "yes", fontsize=8.7, color="#555555",
            ha="left", va="center")
    hw = VW / 2 - 0.6
    ax.add_patch(plt.Rectangle((xs[0] + W / 2 - VW / 2, VY3 - 3.8), hw, 7.6,
                               fill=True, fc=PURPLE2, ec="none"))
    ax.text(xs[0] + W / 2 - VW / 2 + hw / 2, VY3, "CONTAIN-\nMENT",
            ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
    ax.add_patch(plt.Rectangle((xs[0] + W / 2 + 0.6, VY3 - 3.8), hw, 7.6,
                               fill=True, fc=GREEN, ec="none"))
    ax.text(xs[0] + W / 2 + 0.6 + hw / 2, VY3, "NO FLAG",
            ha="center", va="center", fontsize=6.3, color="white", fontweight="bold")
    ax.text(xs[0] + W / 2, VY3 - 5.6, "reported gap $\\geq 0.15$ / else",
            ha="center", va="center", fontsize=6.6, color="#555555")
    # boxes 3-4: magnitude bands
    for x, q, verdict, colour in [(xs[1], "$\\Delta_{\\mathrm{dis}} \\geq 0.15$?", "DIVERGENCE", RED2),
                                  (xs[2], "$\\Delta_{\\mathrm{dis}} \\geq 0.10$?", "CAUTION", ORANGE2)]:
        qbox(x, QY3, q, False)
        arrow(x + W / 2, QY3 - H / 2, x + W / 2, VY3 + 3.8)
        ax.text(x + W / 2 + 1.2, LY3, "yes", fontsize=8.7, color="#555555",
                ha="left", va="center")
        vbox(x + W / 2, VY3, verdict, colour)
    for x0, x1 in zip(xs, xs[1:]):
        arrow(x0 + W, QY3, x1, QY3)
        ax.text((x0 + W + x1) / 2, QY3 + 1.6, "no", fontsize=8.7, color="#555555",
                ha="center", va="bottom")
    xe = xs[-1] + W
    arrow(xe, QY3, xe + 14.0, QY3)
    arrow(xe + 14.0, QY3, xe + 14.0, VY3 + 3.8)
    ax.text(xe + 7.0, QY3 + 1.6, "no", fontsize=8.7, color="#555555", ha="center", va="bottom")
    vbox(xe + 14.0, VY3, "NO FLAG", GREEN, w=VW * 0.95)

    fig.tight_layout()
    fig.savefig(OUT / "fig_verdict_tree.pdf")
    print("wrote verdict tree")


if __name__ == "__main__":
    fig2()
    fig3()
    fig_tree()
