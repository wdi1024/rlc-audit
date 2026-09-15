#!/usr/bin/env python3
"""Is the instrument calibrated, and is the design powered? (2026-08-21)

Two objections that belong together, because both ask whether a number this paper
prints can be trusted before anyone argues about what it means.

POWER. Our primary contract carries 46 construct positives and its disjoint
interval is roughly $\\pm 0.16$ wide. A reviewer notes that a design cannot detect
its own headline effect if the interval is wider than the cutoff, and that is worth
checking rather than asserting. We compute, for each audited contract, the
half-width of the paired-bootstrap interval and the smallest gap that interval
could separate from zero -- the minimum detectable effect -- and the construct
positives that would be needed to bring the MISMATCH cutoff of 0.15 inside range.

CALIBRATION. The disjoint control is an estimator and nothing so far shows it
recovers what it claims. We simulate contracts with a known containment fraction
alpha: a latent quality drives the construct, an opening artifact drives part of
the score, and the proxy is built to draw a controlled share of its agreement from
the shared span. Sweeping alpha from 0 (all evidence, no containment) to 1 (all
containment) we ask whether Delta_dis tracks the earned share and Delta_ext
overstates it. If the estimator is right, Delta_dis should fall roughly linearly
in alpha and reach zero at alpha = 1 while Delta_ext stays positive.

  python3 power_and_calibration.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"
N_BOOT = 1500
MISMATCH_CUT = 0.15
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- power ----
def power_table() -> list[dict]:
    rows = json.loads((AR / "complement_span_proxy_control.json").read_text())["rows"]
    out = []
    for r in rows:
        c = r["comp"]
        if c["degenerate"] or c["ci_abs"][0] is None:
            continue
        lo, hi = c["ci_abs"]
        half = (hi - lo) / 2
        # a paired interval of this width separates from zero only past its half-width
        mde = half
        # bootstrap width scales roughly as 1/sqrt(y+); solve for the y+ that puts
        # the MISMATCH cutoff outside the interval
        yp = r["y_positives"]
        # A contract excluded on power cannot be called "powered" by its own interval:
        # with three construct positives the bootstrap width is not interpretable at all.
        if r["underpowered"]:
            out.append({"contract": r["contract"], "n": r["n"], "y_positives": yp,
                        "delta_dis": c["delta_abs"], "half_width": half,
                        "mde": None, "y_needed_for_0.15": None, "powered": None})
            continue
        need = int(np.ceil(yp * (half / MISMATCH_CUT) ** 2)) if half > 0 else yp
        out.append({"contract": r["contract"], "n": r["n"], "y_positives": yp,
                    "delta_dis": c["delta_abs"], "half_width": half,
                    "mde": mde, "y_needed_for_0.15": need,
                    "powered": half < MISMATCH_CUT})
    return out


# ---------------------------------------------------------- calibration ----
def simulate(alpha: float, n: int = 4000, seed: int = 0) -> dict:
    """One synthetic contract whose containment fraction alpha is known.

    Three latents, because two are not enough to pose the question. The construct
    latent q drives y. The artifact a lives only in the scored span -- an opening
    template. The vocabulary latent v is expressed in both the span and the
    remainder: it is genuine, earned predictability from the span about text beyond
    it, and it is *not* the construct. That third latent is what a disjoint control
    is supposed to find and a containment-only story predicts is absent.

    The proxy rule fires on a mixture: alpha of its signal is the artifact
    (containment) and 1 - alpha is the vocabulary (earned). Read on the span it sees
    both; read off the span it can only see the vocabulary. So Delta_dis should fall
    from clearly positive at alpha = 0 to about zero at alpha = 1, while Delta_ext,
    which still touches the span, should stay high throughout.
    """
    rng = np.random.default_rng(seed + int(alpha * 1000))
    q = rng.normal(size=n)                       # construct latent
    a = rng.normal(size=n)                       # span-only artifact
    v = rng.normal(size=n)                       # shared vocabulary latent

    mix = lambda lat, w: w * lat + np.sqrt(max(1e-9, 1 - w ** 2)) * rng.normal(size=n)
    q_span, v_span = mix(q, 0.30), mix(v, 0.80)  # what the span expresses
    v_off = mix(v, 0.80)                         # what the remainder expresses

    s = 0.55 * a + 0.25 * q_span + 0.55 * v_span  # the score reads the span
    y = (q > 0).astype(int)

    thr = lambda lat: (lat > np.median(lat)).astype(int)
    z = thr(alpha * a + (1 - alpha) * v_span)                    # same span
    ze = thr(alpha * a + (1 - alpha) * (0.5 * v_span + 0.5 * v_off))  # whole output
    # the same rule read off-span: its artifact component finds nothing there, so
    # only the (1 - alpha) vocabulary component survives, plus noise for the rest
    zd = thr((1 - alpha) * v_off + alpha * rng.normal(size=n))

    ay = roc_auc_score(y, s)
    gap = lambda lab: abs(roc_auc_score(lab, s) - 0.5) - abs(ay - 0.5)
    return {"alpha": alpha, "auc_y": ay, "auc_z": float(roc_auc_score(z, s)),
            "auc_zd": float(roc_auc_score(zd, s)),
            "d_auc": gap(z), "d_ext": gap(ze), "d_dis": gap(zd)}


def main() -> None:
    print("=== power: can each contract detect the cutoff it is judged against? ===")
    pt = power_table()
    print(f"{'contract':26s} {'n':>6s} {'y+':>5s} {'dDis':>7s} {'half-CI':>8s} "
          f"{'MDE':>7s} {'y+ for 0.15':>12s}  powered")
    for r in pt:
        if r["powered"] is None:
            print(f"{r['contract'][:26]:26s} {r['n']:6d} {r['y_positives']:5d} "
                  f"{r['delta_dis']:+7.3f} {r['half_width']:8.3f} {'n/a':>7s} "
                  f"{'n/a':>12s}  excluded on power")
            continue
        print(f"{r['contract'][:26]:26s} {r['n']:6d} {r['y_positives']:5d} "
              f"{r['delta_dis']:+7.3f} {r['half_width']:8.3f} {r['mde']:7.3f} "
              f"{r['y_needed_for_0.15']:12d}  {'yes' if r['powered'] else 'NO'}")
    npow = sum(1 for r in pt if r["powered"] is True)
    print(f"\n  {npow}/{len(pt)} contracts have an interval narrow enough to separate a")
    print(f"  {MISMATCH_CUT:.2f} gap from zero. The others cannot support a MISMATCH on the")
    print("  disjoint control however large the point estimate is, and we mark them.")

    print("\n=== calibration: does Delta_dis recover the earned share? ===")
    print(f"{'alpha (containment)':>20s} {'AUC(s,y)':>9s} {'dAUC':>8s} {'dExt':>8s} {'dDis':>8s}")
    sims = []
    for alpha in np.linspace(0.0, 1.0, 11):
        r = simulate(float(alpha))
        sims.append(r)
        print(f"{alpha:20.2f} {r['auc_y']:9.3f} {r['d_auc']:+8.3f} {r['d_ext']:+8.3f} "
              f"{r['d_dis']:+8.3f}")

    a = np.array([s["alpha"] for s in sims])
    dd = np.array([s["d_dis"] for s in sims])
    de = np.array([s["d_ext"] for s in sims])
    from scipy.stats import pearsonr
    rd, _ = pearsonr(a, dd)
    re_, _ = pearsonr(a, de)
    print(f"\n  corr(alpha, Delta_dis) = {rd:+.3f};  corr(alpha, Delta_ext) = {re_:+.3f}")
    print(f"  Delta_dis at alpha=0: {dd[0]:+.3f};  at alpha=1: {dd[-1]:+.3f}")
    print(f"  Delta_ext at alpha=1: {de[-1]:+.3f}  (a control that removed containment would be 0)")
    ok = rd < -0.8 and abs(dd[-1]) < 0.05
    print("\nreading:")
    if ok:
        print("  the disjoint control falls with the containment fraction and vanishes when the")
        print("  agreement is entirely containment, while the extended control still reports a")
        print("  gap there. The estimator measures what it claims on data where we know alpha.")
    else:
        print("  the estimator does not track alpha cleanly in this generator; report the")
        print("  simulation as inconclusive rather than as validation.")

    out = {"power": pt, "n_powered": npow, "calibration": sims,
           "corr_alpha_dis": float(rd), "corr_alpha_ext": float(re_)}
    p = AR / "power_and_calibration.json"
    p.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", p)


if __name__ == "__main__":
    main()
