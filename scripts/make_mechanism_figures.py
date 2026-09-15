#!/usr/bin/env python3
"""Generate mechanism figures that read as figures rather than tables."""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
CASES_PATH = ROOT / "analysis_results" / "inspect_2b_disagree_50char.json"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#7A5195"
GRAY = "#6E6E6E"
LIGHT = "#F7F7F7"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_cases():
    with CASES_PATH.open() as f:
        return json.load(f)


def add_card(ax, xy, wh, title, body, color):
    x, y = xy
    w, h = wh
    title_h = 0.08
    card = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=color,
        facecolor="#FFFFFF",
        zorder=2,
    )
    ax.add_patch(card)
    ax.add_patch(
        patches.Rectangle(
            (x, y + h - title_h),
            w,
            title_h,
            linewidth=0,
            facecolor=color,
            alpha=0.15,
            zorder=3,
        )
    )
    ax.text(
        x + 0.02,
        y + h - title_h / 2,
        title,
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=color,
        zorder=4,
    )
    wrapped = "\n".join(textwrap.fill(line, width=34) for line in body.split("\n"))
    ax.text(
        x + 0.025,
        y + h - title_h - 0.035,
        wrapped,
        ha="left",
        va="top",
        fontsize=7.8,
        color="#202020",
        linespacing=1.08,
        zorder=4,
    )


def add_arrow(ax, x0, x1, y):
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4),
        zorder=1,
    )


def make_schematic(data):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    w = 0.39
    h = 0.22
    left_x = 0.07
    right_x = 0.54
    top_y = 0.67
    bot_y = 0.40
    add_card(
        ax,
        (left_x, top_y),
        (w, h),
        "Score span",
        "Qwen: thinking preamble\nGemma: 'I cannot'\ncosine = 0.000 (13/13)",
        BLUE,
    )
    add_card(
        ax,
        (right_x, top_y),
        (w, h),
        "Surface proxy",
        "Gemma keyword fires\nQwen preamble is missed\nproxy says disagreement",
        ORANGE,
    )
    add_card(
        ax,
        (left_x, bot_y),
        (w, h),
        "Semantic construct",
        "Judge reads answer decision\n11/13 are same refusal\nsemantic split is rare",
        GREEN,
    )
    add_card(
        ax,
        (right_x, bot_y),
        (w, h),
        "Routed queue",
        "At XSTest budget B=55\n6 semantic disagreements\n37 agreed refusals",
        RED,
    )

    add_arrow(ax, left_x + w + 0.02, right_x - 0.02, top_y + h / 2)
    add_arrow(ax, left_x + w + 0.02, right_x - 0.02, bot_y + h / 2)
    ax.annotate(
        "",
        xy=(left_x + w / 2, bot_y + h + 0.015),
        xytext=(left_x + w / 2, top_y - 0.015),
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4),
        zorder=1,
    )
    ax.annotate(
        "",
        xy=(right_x + w / 2, bot_y + h + 0.015),
        xytext=(right_x + w / 2, top_y - 0.015),
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4),
        zorder=1,
    )

    ax.text(
        0.5,
        0.955,
        "Opening templates create a proxy signal before the semantic decision appears",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
    )

    # Bottom diagnostic bars.
    ax.text(0.07, 0.27, "Inspected prefix-keyword positives", fontsize=8.5, color="#202020")
    dot_x = np.linspace(0.09, 0.43, 13)
    for i, dx in enumerate(dot_x):
        c = GREEN if i not in (0, 8) else RED
        ax.scatter(dx, 0.205, s=82, color=c, edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(0.45, 0.205, "11 same semantic decision / 2 semantic disagreements", va="center", fontsize=8)

    ax.text(0.07, 0.145, "Routed queue at B=55", fontsize=8.5, color="#202020")
    bar_x, bar_y, bar_w, bar_h = 0.29, 0.065, 0.56, 0.048
    sem_dis = 6 / 55
    agreed_refusal = 37 / 55
    other = 1 - sem_dis - agreed_refusal
    segments = [
        (sem_dis, RED, "6 sem. dis."),
        (agreed_refusal, GREEN, "37 agreed refusals"),
        (other, "#CFCFCF", "12 other agreements"),
    ]
    cur = bar_x
    for frac, color, label in segments:
        ax.add_patch(patches.Rectangle((cur, bar_y), bar_w * frac, bar_h, color=color, lw=0))
        ax.text(cur + bar_w * frac / 2, bar_y - 0.022, label, ha="center", va="top", fontsize=7)
        cur += bar_w * frac
    ax.add_patch(patches.Rectangle((bar_x, bar_y), bar_w, bar_h, fill=False, edgecolor="#333333", lw=0.7))

    out = OUT_DIR / "fig3_mechanism_schematic.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


def opening_category(prefix):
    low = prefix.lower()
    if "thinking process" in low or "thinking" in low:
        return "thinking preamble"
    if low.startswith("here"):
        return "reasoning preamble"
    return "other preamble"


def make_case_map(data):
    cases = data["cases"]
    n = len(cases)
    labels = [f"C{i + 1}" for i in range(n)]
    same = np.array([c["judge_qwen"] == c["judge_gemma"] for c in cases])
    qwen_cats = [opening_category(c["qwen_prefix"]) for c in cases]
    cat_colors = {
        "thinking preamble": BLUE,
        "reasoning preamble": "#86A9D4",
        "other preamble": PURPLE,
    }

    fig = plt.figure(figsize=(7.4, 4.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.75, 0.75, 1.05, 0.9], hspace=0.34)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_title(
        "All 13 prefix-keyword positives share the same opening-template mechanism",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    for i, cat in enumerate(qwen_cats):
        ax0.scatter(i, 0.62, s=135, marker="s", color=cat_colors[cat], edgecolor="white", lw=0.8)
        ax0.scatter(i, 0.22, s=135, marker="s", color=ORANGE, edgecolor="white", lw=0.8)
    ax0.text(-0.65, 0.62, "Qwen prefix", ha="right", va="center", fontsize=8)
    ax0.text(-0.65, 0.22, "Gemma prefix", ha="right", va="center", fontsize=8)
    ax0.text(n - 0.2, 0.62, "thinking / reasoning preambles", va="center", fontsize=8, color=BLUE)
    ax0.text(n - 0.2, 0.22, "'I cannot' refusal openings", va="center", fontsize=8, color=ORANGE)
    ax0.set_xlim(-1.2, n + 2.3)
    ax0.set_ylim(0, 0.85)
    ax0.set_xticks(range(n), labels, fontsize=7)
    ax0.set_yticks([])
    ax0.spines[["left", "bottom"]].set_visible(False)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.scatter(range(n), np.zeros(n), s=95, color="#222222", edgecolor="white", lw=0.7, zorder=3)
    ax1.axhline(0, color="#BBBBBB", lw=0.8)
    ax1.set_ylim(-0.12, 0.18)
    ax1.set_yticks([0], ["0.000"], fontsize=8)
    ax1.set_ylabel("prefix\ncosine", rotation=0, ha="right", va="center", fontsize=8)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.spines[["bottom"]].set_visible(False)
    ax1.text(n - 0.2, 0.02, "lexical orthogonality in every inspected case", fontsize=8, va="center")

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    colors = [GREEN if s else RED for s in same]
    ax2.scatter(range(n), np.zeros(n), s=190, color=colors, edgecolor="white", lw=1.1, zorder=3)
    for i, c in enumerate(cases):
        label = "same" if same[i] else "split"
        ax2.text(i, -0.32, label, ha="center", va="top", rotation=35, fontsize=7, color=colors[i])
    ax2.set_ylim(-0.62, 0.4)
    ax2.set_yticks([])
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.spines[["left", "bottom"]].set_visible(False)
    ax2.text(-0.65, 0.0, "Judge outcome", ha="right", va="center", fontsize=8)
    ax2.text(n - 0.2, 0.05, "11/13 same semantic decision", fontsize=8, color=GREEN, va="center")

    ax3 = fig.add_subplot(gs[3])
    counts = [int(same.sum()), int((~same).sum())]
    ax3.barh([0], [counts[0]], color=GREEN, height=0.45, label="same judge label")
    ax3.barh([0], [counts[1]], left=[counts[0]], color=RED, height=0.45, label="semantic split")
    ax3.set_xlim(0, n)
    ax3.set_yticks([])
    ax3.set_xlabel("Inspected prefix-keyword positives", fontsize=8)
    ax3.text(counts[0] / 2, 0, "11 same", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax3.text(counts[0] + counts[1] / 2, 0, "2 split", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax3.text(0, -0.62, "Green = same judge label; red = semantic split", ha="left", va="center", fontsize=8)

    out = OUT_DIR / "fig4_mechanism_case_map.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


def main():
    data = load_cases()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [make_schematic(data), make_case_map(data)]:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
