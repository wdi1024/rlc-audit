#!/usr/bin/env python3
"""Error rates of the final rule on synthetic cases with known structure, and a
baseline that reports AUC(s,y) and kappa(z,y) under the same preconditions.

Cases (construct y = [q > 0]; the score reads the span; the proxy rule is read on
the span, z, and strictly beyond it, z^c):
  valid            proxy is a noisy reading of the construct on both spans; no artifact
  containment      proxy fires on a span-only opening artifact the score also reads
  mismatch         proxy measures a latent m that is not the construct; score tracks q and m
  spillover        proxy measures a vocabulary latent expressed on and beyond the span
  score_failure    score tracks the artifact and the vocabulary but not the construct
  high_kappa       z^c agrees with y on 80% of items (kappa 0.6) and the score predicts z^c exactly
  mixed            half containment, half mismatch
For each case and sample size the final rule of Section 3 is applied with its
preconditions, equivalence test, per-contract null and magnitude bands, and the
distribution of exits is reported. The baseline flags a contract when the
construct arm is at chance (TOST) or when it ranks and kappa(z,y) < 0.40; it has no
off-span arm, so it can neither split SCORE FAILURE from CONTAINMENT nor reverse a
reading. Run from the repository root.
"""
import json, importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis_results" / "diagnostic_error_rates.json"
BAND, DIV, CAU, RANKED = (0.45, 0.55), 0.15, 0.10, 0.10
N_BOOT, N_PERM, REPS = 200, 150, 60


def kappa(a, b):
    po = float((a == b).mean()); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def make(case, n, rng):
    q, a, v, m = (rng.normal(size=n) for _ in range(4))
    mix = lambda lat, w: w * lat + np.sqrt(max(1e-9, 1 - w ** 2)) * rng.normal(size=n)
    thr = lambda lat: (lat > np.median(lat)).astype(int)
    y = (q > 0).astype(int)
    q_span, v_span, v_off, m_span, m_off = mix(q, .30), mix(v, .80), mix(v, .80), mix(m, .80), mix(m, .80)
    if case == "valid":
        s = 0.7 * q_span + 0.3 * rng.normal(size=n); z = thr(mix(q, .7)); zc = thr(mix(q, .7))
    elif case == "containment":
        s = 0.55 * a + 0.25 * q_span; z = thr(a); zc = thr(rng.normal(size=n))
    elif case == "mismatch":
        s = 0.35 * q_span + 0.55 * m_span; z = thr(m_span); zc = thr(m_off)
    elif case == "spillover":
        s = 0.25 * q_span + 0.55 * v_span; z = thr(v_span); zc = thr(v_off)
    elif case == "score_failure":
        s = 0.55 * a + 0.55 * v_span; z = thr(a + v_span); zc = thr(v_off)
    elif case == "high_kappa":
        zc = np.where(rng.random(n) < 0.20, 1 - y, y); z = zc.copy(); s = zc + 0.05 * rng.normal(size=n)
    elif case == "mixed":
        s = 0.55 * a + 0.35 * q_span + 0.4 * m_span; z = thr(0.5 * a + 0.5 * m_span); zc = thr(m_off)
    return np.asarray(s, float), z, zc, y


def boot_ci(s, y, rng, q):
    v = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(y[sel])) < 2: continue
        v.append(roc_auc_score(y[sel], s[sel]))
    return np.percentile(v, q[0]), np.percentile(v, q[1])


def null_p95(s, zc, y, rng):
    base = abs(roc_auc_score(y, s) - 0.5); idx = {c: np.flatnonzero(y == c) for c in (0, 1)}
    zp = zc.copy(); out = []
    for _ in range(N_PERM):
        for c, ix in idx.items(): zp[ix] = zc[rng.permutation(ix)]
        if len(set(zp)) < 2: continue
        out.append(abs(roc_auc_score(zp, s) - 0.5) - base)
    return np.percentile(out, 95)


def rule(s, z, zc, y, rng, kappa_exit=False):
    if y.sum() < 10 or (len(y) - y.sum()) < 10: return "UNDECIDABLE"
    vals, cnt = np.unique(np.round(s, 6), return_counts=True)
    if cnt.max() / len(s) > 0.5: return "UNDECIDABLE"
    lo90, hi90 = boot_ci(s, y, rng, (5, 95)); lo95, hi95 = boot_ci(s, y, rng, (2.5, 97.5))
    a_y, a_z, a_zc = roc_auc_score(y, s), roc_auc_score(z, s), roc_auc_score(zc, s)
    if BAND[0] <= lo90 and hi90 <= BAND[1]:
        return "SCORE FAILURE" if abs(a_zc - 0.5) >= RANKED else "CONTAINMENT"
    if lo95 <= 0.5 <= hi95: return "UNDECIDABLE"
    if kappa_exit and kappa(zc, y) >= 0.40: return "ALIGNED"
    d_dis = abs(a_zc - 0.5) - abs(a_y - 0.5); d_auc = abs(a_z - 0.5) - abs(a_y - 0.5)
    if d_dis <= null_p95(s, zc, y, rng): return "CONTAINMENT" if d_auc >= DIV else "ALIGNED"
    return "DIVERGENCE" if d_dis >= DIV else ("CAUTION" if d_dis >= CAU else "ALIGNED")


def baseline(s, z, y, rng):
    """Same preconditions and equivalence test; no off-span arm."""
    if y.sum() < 10 or (len(y) - y.sum()) < 10: return "UNDECIDABLE"
    lo90, hi90 = boot_ci(s, y, rng, (5, 95)); lo95, hi95 = boot_ci(s, y, rng, (2.5, 97.5))
    if BAND[0] <= lo90 and hi90 <= BAND[1]: return "FLAG: not evidence"
    if lo95 <= 0.5 <= hi95: return "UNDECIDABLE"
    return "FLAG: label" if kappa(z, y) < 0.40 else "NO FLAG"


def main():
    cases = ["valid", "containment", "mismatch", "spillover", "score_failure", "high_kappa", "mixed"]
    expected = {"valid": "ALIGNED", "containment": "CONTAINMENT", "mismatch": "DIVERGENCE", "spillover": "DIVERGENCE (not separable from mismatch)",
                "score_failure": "SCORE FAILURE", "high_kappa": "DIVERGENCE", "mixed": "DIVERGENCE or CONTAINMENT"}
    res = {}
    for n in (450, 2000):
        for case in cases:
            rng = np.random.default_rng(20260906 + n)
            tally, tally_k, tally_b = {}, {}, {}
            for r in range(REPS):
                s, z, zc, y = make(case, n, rng)
                v = rule(s, z, zc, y, rng); tally[v] = tally.get(v, 0) + 1
                if case == "high_kappa":
                    vk = rule(s, z, zc, y, rng, kappa_exit=True); tally_k[vk] = tally_k.get(vk, 0) + 1
                b = baseline(s, z, y, rng); tally_b[b] = tally_b.get(b, 0) + 1
            res[f"{case}@{n}"] = dict(expected=expected[case], rule={k: v / REPS for k, v in tally.items()},
                                     baseline={k: v / REPS for k, v in tally_b.items()},
                                     rule_with_kappa_exit={k: v / REPS for k, v in tally_k.items()} if tally_k else None)
            print(f"n={n:4d} {case:14s} rule={ {k: round(v/REPS,2) for k,v in tally.items()} }  baseline={ {k: round(v/REPS,2) for k,v in tally_b.items()} }"
                  + (f"  with-kappa-exit={ {k: round(v/REPS,2) for k,v in tally_k.items()} }" if tally_k else ""))
    OUT.write_text(json.dumps({"reps": REPS, "n_boot": N_BOOT, "n_perm": N_PERM, "results": res}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
