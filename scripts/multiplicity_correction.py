#!/usr/bin/env python3
"""Do the verdicts survive a multiplicity correction? (2026-08-20)

A reviewer noted that the paper applies threshold verdicts to roughly thirty
contracts while reporting every interval and permutation test on its own.  With
that many tests, some positive gaps are expected by chance, and a paper that
recommends a reporting standard should not exempt itself from the correction it
would ask of others.

This script collects every gap test the paper reports as evidence, applies
Benjamini--Hochberg across the whole family, and reports which verdicts survive.
Three families are pooled because a reviewer would pool them: the six audited
refusal settings, the ten-pair final-span panel, and the correctness span sweep.

Permutation p-values are floored at 1/(B+1) rather than reported as zero, since a
permutation test cannot resolve below its resampling budget; using 0 would make the
correction look better than the evidence supports.

  python3 multiplicity_correction.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"
ALPHA = 0.05


def benjamini_hochberg(pvals: list[float], alpha: float = ALPHA) -> tuple[np.ndarray, float]:
    """Return the reject mask and the adaptive threshold."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    crit = alpha * (np.arange(1, m + 1) / m)
    passing = np.where(ranked <= crit)[0]
    if len(passing) == 0:
        return np.zeros(m, dtype=bool), 0.0
    kmax = passing[-1]
    cutoff = ranked[kmax]
    return p <= cutoff, float(cutoff)


def bh_adjusted(pvals: list[float]) -> np.ndarray:
    """Benjamini--Hochberg adjusted p-values (step-up, monotone-enforced)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out


def ci_to_p(lo: float, hi: float) -> float:
    """A conservative two-sided p-value from a bootstrap interval.

    We do not have the resample draws for the panel rows, only their 95% CIs, so we
    convert with a normal approximation: sd = (hi-lo)/(2*1.96), z = mean/sd.  This is
    an approximation and is labelled as one in the output.
    """
    mean = 0.5 * (lo + hi)
    sd = (hi - lo) / (2 * 1.959964)
    if sd <= 0:
        return 0.0 if mean != 0 else 1.0
    from math import erfc, sqrt
    return float(erfc(abs(mean / sd) / sqrt(2)))


def main() -> None:
    tests: list[dict] = []

    # ---- family 1: the six audited refusal settings ------------------------
    gs = json.loads((AR / "gap_significance.json").read_text())
    for e in gs:
        n_perm = e.get("perm_n", 10000)
        p = max(e["p_perm"], 1.0 / (n_perm + 1))
        tests.append({"family": "refusal setting", "name": e["name"],
                      "effect": e["D_obs"], "p": p,
                      "source": "permutation", "floored": e["p_perm"] == 0})

    # ---- family 2: the ten-pair final-span panel ---------------------------
    panel_rows = [
        ("Qwen9B/Gemma9B", -0.015, -0.092, 0.062), ("Qwen9B/Mistral7B", 0.018, -0.060, 0.096),
        ("Qwen9B/Phi3.5", 0.128, 0.043, 0.212), ("Qwen8B/Phi3.5", 0.118, 0.032, 0.204),
        ("Gemma9B/Mistral7B", -0.006, -0.083, 0.070), ("Gemma9B/Qwen8B", 0.042, -0.036, 0.121),
        ("Gemma9B/Phi3.5", 0.015, -0.062, 0.093), ("Mistral7B/Qwen8B", 0.068, -0.010, 0.147),
        ("Mistral7B/Phi3.5", 0.054, -0.024, 0.232), ("Qwen8B/Qwen9B", -0.094, -0.170, -0.018),
    ]
    for name, eff, lo, hi in panel_rows:
        tests.append({"family": "pair panel", "name": name, "effect": eff,
                      "p": ci_to_p(lo, hi), "source": "CI (normal approx)", "floored": False})

    # ---- family 3: the correctness span sweep ------------------------------
    sweep = json.loads((AR / "span_closure_fine_sweep.json").read_text())
    for dom in ("GSM8K", "HotpotQA"):
        for r in sweep[dom]["rows"]:
            if "gap" not in r:
                continue
            lo, hi = r["gap_ci"]
            tests.append({"family": f"span sweep ({dom})", "name": f"{dom} {r['span']}",
                          "effect": r["gap"], "p": ci_to_p(lo, hi),
                          "source": "CI (normal approx)", "floored": False})

    pvals = [t["p"] for t in tests]
    reject, cutoff = benjamini_hochberg(pvals)
    adj = bh_adjusted(pvals)

    print(f"{len(tests)} gap tests pooled into one Benjamini--Hochberg family "
          f"(alpha={ALPHA})")
    print(f"BH cutoff on raw p: {cutoff:.5f}\n")
    print(f"{'family':22s} {'contract':26s} {'effect':>8s} {'raw p':>10s} {'BH p':>9s}  survives")
    for t, r, a in zip(tests, reject, adj):
        flag = "yes" if r else "NO"
        note = " (floored)" if t["floored"] else ""
        print(f"{t['family'][:22]:22s} {t['name'][:26]:26s} {t['effect']:+8.3f} "
              f"{t['p']:10.5f} {a:9.5f}  {flag}{note}")

    n_pos = sum(1 for t in tests if t["effect"] > 0)
    surv_pos = sum(1 for t, r in zip(tests, reject) if r and t["effect"] > 0)
    print(f"\npositive-gap tests: {n_pos}; surviving BH: {surv_pos}")
    changed = [t["name"] for t, r in zip(tests, reject)
               if not r and t["p"] < ALPHA]
    print(f"tests significant before correction but not after ({len(changed)}): {changed}")

    out = {"alpha": ALPHA, "n_tests": len(tests), "bh_cutoff": cutoff,
           "tests": [{**t, "p_bh": float(a), "survives": bool(r)}
                     for t, a, r in zip(tests, adj, reject)],
           "lost_to_correction": changed}
    path = AR / "multiplicity_correction.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
