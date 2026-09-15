#!/usr/bin/env python3
"""Semi-synthetic calibration on real generations.

Section 4's synthetic grid validates the disjoint control up to monotonicity and
cannot anchor the constants 0.10 / 0.15. This experiment injects known amounts
of two things into the real XSTest and OR-Bench generations, with the construct
labels held fixed, and reads what the controls return:

  Sweep A (containment, rate alpha): on a fraction alpha of proxy-negative
    pairs drawn independently of the construct label, the first 50 characters of the two sides are
    overwritten with a refusal opening on one side and a compliance opening on
    the other. The same-span keyword rule now fires and the prefix distance is
    large, from evidence the construct does not carry. Off-span text is untouched.
    Prediction: Delta_AUC rises with alpha; Delta_dis does not.

  Sweep B (off-span mismatch, rate beta): the same opening overwrite on a
    fraction beta of off-span-negative pairs drawn independently of the construct, plus a refusal
    sentence appended to one side beyond the span, so the off-span rule fires on
    pairs the score also separates. Prediction: Delta_dis rises with beta, and
    the curve says which beta the constants 0.10 and 0.15 correspond to.

Referee check: alpha = beta = 0 reproduces the recorded TF-IDF rows. Twenty
seeds per point; intervals are the 2.5-97.5 percentiles over seeds.
Run from the repository root.
"""
import importlib.util, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
enc = _load("encoder_contracts_full_rule")     # load_setting, agcr, SP
agcr, SP = enc.agcr, enc.SP
OUT = ROOT / "analysis_results" / "semisynthetic_calibration.json"

REFUSAL_OPEN = "I'm sorry, but I cannot help with that request. "
COMPLY_OPEN = "Sure, here is a clear and helpful answer for you. "
OFFSPAN_REFUSAL = " I cannot help with the rest of this request."
RATES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
SEEDS = 20
SETTINGS = [("phase3_xstest_full", "XSTest 450 (primary run)", (0.985, 0.592, 0.101)),
            ("phase8_orbench_hard1k", "OR-Bench hard 1k", (0.866, 0.517, 0.248))]


def gaps(a, b, y):
    z = np.array([int(agcr.is_kw(x[:SP]) != agcr.is_kw(w[:SP])) for x, w in zip(a, b)])
    zp = np.array([int(agcr.is_kw(x) != agcr.is_kw(w)) for x, w in zip(a, b)])
    zc = np.array([int(agcr.is_kw(x[SP:]) != agcr.is_kw(w[SP:])) for x, w in zip(a, b)])
    s = agcr.tfidf_distance(a, b, prefix=SP)
    ay = roc_auc_score(y, s)
    def arm(lab):
        if len(set(lab)) < 2: return None
        return abs(roc_auc_score(lab, s) - 0.5) - abs(ay - 0.5)
    return dict(auc_y=float(ay), auc_z=float(roc_auc_score(z, s)) if len(set(z)) > 1 else None,
                d_auc=arm(z), d_ext=arm(zp), d_dis=arm(zc), z_pos=int(z.sum()), zc_pos=int(zc.sum()))


def inject(a, b, y, rate, mode, rng):
    a, b = list(a), list(b)
    if mode == "A":
        pool = [i for i in range(len(a)) if agcr.is_kw(a[i][:SP]) == agcr.is_kw(b[i][:SP])]
    else:
        pool = [i for i in range(len(a)) if agcr.is_kw(a[i][SP:]) == agcr.is_kw(b[i][SP:])]
    k = int(round(rate * len(a)))
    if k > len(pool): k = len(pool)
    for i in rng.choice(pool, size=k, replace=False):
        a[i] = REFUSAL_OPEN + a[i][SP:]
        b[i] = COMPLY_OPEN + b[i][SP:]
        if mode == "B":
            a[i] = a[i] + OFFSPAN_REFUSAL
    return a, b, k


def main():
    out = {"rates": RATES, "seeds": SEEDS, "settings": {}}
    for phase, label, (rz, ry, rdis) in SETTINGS:
        a, b, z, zp, zc, y, empty = enc.load_setting(phase)
        base = gaps(a, b, y)
        assert abs(base["auc_z"] - rz) < 2e-3 and abs(base["auc_y"] - ry) < 2e-3 and abs(base["d_dis"] - rdis) < 2e-3, (label, base)
        print(f"[referee] {label}: base row reproduced (AUC z {base['auc_z']:.3f}, y {base['auc_y']:.3f}, dis {base['d_dis']:+.3f})")
        res = {"base": base, "A": {}, "B": {}}
        for mode in ("A", "B"):
            print(f"  sweep {mode}  ({'containment' if mode=='A' else 'off-span mismatch'} injected)")
            for rate in RATES:
                vals = {k: [] for k in ("d_auc", "d_ext", "d_dis", "auc_y")}
                ks = []
                for seed in range(SEEDS):
                    rng = np.random.default_rng(1000 * seed + int(rate * 100))
                    aa, bb, k = inject(a, b, y, rate, mode, rng)
                    g = gaps(aa, bb, y); ks.append(k)
                    for kk in vals: vals[kk].append(g[kk])
                summ = {kk: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
                        for kk, v in vals.items() if all(x is not None for x in v)}
                summ["pairs_injected"] = float(np.mean(ks)); summ["achieved_rate"] = float(np.mean(ks) / len(a))
                res[mode][str(rate)] = summ
                print(f"    rate {rate:.2f} (n={np.mean(ks):5.1f})  dAUC {summ['d_auc']['mean']:+.3f}  dext {summ['d_ext']['mean']:+.3f}  "
                      f"ddis {summ['d_dis']['mean']:+.3f} [{summ['d_dis']['lo']:+.3f},{summ['d_dis']['hi']:+.3f}]  AUC_y {summ['auc_y']['mean']:.3f}")
        out["settings"][label] = res
    OUT.write_text(json.dumps(out, indent=1)); print("wrote", OUT)


if __name__ == "__main__":
    main()
