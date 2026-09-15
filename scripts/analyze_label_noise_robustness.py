#!/usr/bin/env python3
"""Label-noise robustness checks for semantic-refusal labels.

The paper does not rely on fresh human annotation for every routed pair. This
script quantifies how sensitive the main RLC-Audit conclusion is to plausible
semantic-label error by asking two questions:

1. Random-noise check: if a percentage of pair-level semantic labels are flipped
   at random, does the semantic AUC become competitive with the surface AUC?
2. Adversarial swap check: preserving the semantic-positive count, how many
   score-aligned pair-label swaps are needed before the AUC gap would no longer
   trigger the default FAIL rule?

Outputs:
  analysis_results/label_noise_robustness.json
  paper/appendix_label_noise_robustness.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_submission_robustness import (
    SETTINGS,
    build_arrays,
    metric_bundle,
    safe_auc,
    tfidf_scores,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_results"
PAPER_DIR = ROOT / "paper"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_FLIP_RATES = [0.05, 0.10, 0.20, 0.30]
RANDOM_TRIALS = 1000
FAIL_AUC_GAP = 0.15


def prefix50(text: str) -> str:
    return (text or "")[:50]


def adversarial_swap_curve(y: np.ndarray, score: np.ndarray, target_auc: float) -> dict:
    """Swap high-score negatives with low-score positives until target AUC.

    This preserves the number of semantic positives. It is intentionally a
    strong stress test: the swaps are chosen in the direction that most helps
    the routing score look semantically aligned.
    """

    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]
    low_positive = positives[np.argsort(score[positives])]
    high_negative = negatives[np.argsort(-score[negatives])]
    max_swaps = int(min(len(low_positive), len(high_negative)))

    checkpoints = []
    min_swaps = None
    min_pair_label_flips = None
    auc_at_min = None

    selected_t = sorted(
        set(
            [
                0,
                1,
                2,
                3,
                5,
                10,
                int(round(0.05 * len(y) / 2)),
                int(round(0.10 * len(y) / 2)),
                int(round(0.20 * len(y) / 2)),
                max_swaps,
            ]
        )
    )

    for t in range(max_swaps + 1):
        y_swapped = y.copy()
        if t:
            y_swapped[low_positive[:t]] = 0
            y_swapped[high_negative[:t]] = 1
        auc_t = safe_auc(y_swapped, score)
        if t in selected_t:
            checkpoints.append(
                {
                    "swaps": t,
                    "pair_label_flips": 2 * t,
                    "flip_rate": (2 * t) / len(y),
                    "semantic_auc": auc_t,
                }
            )
        if min_swaps is None and auc_t is not None and auc_t >= target_auc:
            min_swaps = t
            min_pair_label_flips = 2 * t
            auc_at_min = auc_t

    if checkpoints[-1]["swaps"] != max_swaps:
        y_swapped = y.copy()
        if max_swaps:
            y_swapped[low_positive[:max_swaps]] = 0
            y_swapped[high_negative[:max_swaps]] = 1
        checkpoints.append(
            {
                "swaps": max_swaps,
                "pair_label_flips": 2 * max_swaps,
                "flip_rate": (2 * max_swaps) / len(y),
                "semantic_auc": safe_auc(y_swapped, score),
            }
        )

    return {
        "target_semantic_auc": target_auc,
        "min_swaps_to_target": min_swaps,
        "min_pair_label_flips_to_target": min_pair_label_flips,
        "min_pair_label_flip_rate_to_target": (
            (min_pair_label_flips / len(y)) if min_pair_label_flips is not None else None
        ),
        "semantic_auc_at_target": auc_at_min,
        "max_swaps": max_swaps,
        "checkpoints": checkpoints,
    }


def random_flip_summary(y: np.ndarray, score: np.ndarray, rng: np.random.Generator) -> list[dict]:
    rows = []
    for rate in RANDOM_FLIP_RATES:
        aucs = []
        for _ in range(RANDOM_TRIALS):
            flips = rng.random(len(y)) < rate
            y_noisy = y.copy()
            y_noisy[flips] = 1 - y_noisy[flips]
            auc_t = safe_auc(y_noisy, score)
            if auc_t is not None:
                aucs.append(auc_t)
        arr = np.array(aucs, dtype=float)
        rows.append(
            {
                "flip_rate": rate,
                "trials": int(len(arr)),
                "semantic_auc_mean": float(arr.mean()) if len(arr) else None,
                "semantic_auc_p05": float(np.quantile(arr, 0.05)) if len(arr) else None,
                "semantic_auc_p95": float(np.quantile(arr, 0.95)) if len(arr) else None,
            }
        )
    return rows


def analyze_setting(setting, rng: np.random.Generator) -> dict:
    arrays = build_arrays(setting, prefix50)
    score = tfidf_scores(arrays["a_texts"], arrays["b_texts"])
    kw = arrays["kw_dis"].astype(int)
    sem = arrays["judge_dis"].astype(int)

    surface = metric_bundle(kw, score)
    semantic = metric_bundle(sem, score)
    surface_auc = surface["roc_auc"]
    semantic_auc = semantic["roc_auc"]
    target_auc = None
    swap = None
    if surface_auc is not None and semantic_auc is not None:
        target_auc = surface_auc - FAIL_AUC_GAP
        swap = adversarial_swap_curve(sem, score, target_auc)

    return {
        "setting": setting.name,
        "short": setting.short,
        "n": int(len(sem)),
        "surface_n_pos": int(kw.sum()),
        "semantic_n_pos": int(sem.sum()),
        "surface_auc": surface_auc,
        "semantic_auc": semantic_auc,
        "auc_gap": (
            float(surface_auc - semantic_auc)
            if surface_auc is not None and semantic_auc is not None
            else None
        ),
        "default_fail_gap": FAIL_AUC_GAP,
        "random_pair_label_flips": random_flip_summary(sem, score, rng),
        "adversarial_prevalence_preserving_swaps": swap,
    }


def fmt(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}"


def write_markdown(report: dict) -> None:
    lines = [
        "# Appendix: Label-Noise Robustness",
        "",
        "The main audit does not assume fresh human annotation for every routed pair. "
        "This check asks how much pair-level semantic-label error would be needed "
        "to erase the default RLC-Audit AUC-gap diagnosis.",
        "",
        "## A. Adversarial prevalence-preserving swaps",
        "",
        "A swap changes one low-score semantic positive into a negative and one "
        "high-score semantic negative into a positive. This preserves the semantic "
        "positive count and is chosen to maximally help the score. The target is "
        "`surface AUC - semantic AUC < 0.15`, the default FAIL gap threshold.",
        "",
        "| Setting | n | surf AUC | sem AUC | target sem AUC | min pair-label flips | flip rate | sem AUC after flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        swap = row["adversarial_prevalence_preserving_swaps"] or {}
        lines.append(
            f"| {row['setting']} | {row['n']} | {fmt(row['surface_auc'])} | "
            f"{fmt(row['semantic_auc'])} | {fmt(swap.get('target_semantic_auc'))} | "
            f"{swap.get('min_pair_label_flips_to_target') or '-'} | "
            f"{fmt(swap.get('min_pair_label_flip_rate_to_target'))} | "
            f"{fmt(swap.get('semantic_auc_at_target'))} |"
        )

    lines.extend(
        [
            "",
            "## B. Random pair-label flips",
            "",
            f"Each row summarizes {RANDOM_TRIALS} random-noise trials.",
            "",
            "| Setting | flip rate | semantic AUC mean | p05 | p95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["rows"]:
        for noise in row["random_pair_label_flips"]:
            lines.append(
                f"| {row['setting']} | {noise['flip_rate']:.2f} | "
                f"{fmt(noise['semantic_auc_mean'])} | {fmt(noise['semantic_auc_p05'])} | "
                f"{fmt(noise['semantic_auc_p95'])} |"
            )

    (PAPER_DIR / "appendix_label_noise_robustness.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    rng = np.random.default_rng(20260518)
    rows = [analyze_setting(setting, rng) for setting in SETTINGS]
    report = {
        "description": (
            "Label-noise robustness for semantic refusal-disagreement labels. "
            "Adversarial swaps preserve semantic-positive prevalence and are "
            "chosen in the score-aligned direction."
        ),
        "random_trials": RANDOM_TRIALS,
        "rows": rows,
    }
    out_path = OUT_DIR / "label_noise_robustness.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    write_markdown(report)
    print(f"Saved: {out_path}")
    print(f"Saved: {PAPER_DIR / 'appendix_label_noise_robustness.md'}")


if __name__ == "__main__":
    main()
