#!/usr/bin/env python3
"""The opening-strip dissection, transferred to OR-Bench hard 1k. (2026-09-04)

Review round 4, A2: the paper dissects the primary XSTest contract (CAUTION,
underpowered) while OR-Bench hard 1k is a certified SCORE FAILURE scored by the
same prefix-50 TF-IDF distance, so the same intervention applies there with
y+ = 473. This script mirrors opening_strip_disjoint_and_volume_control.py
exactly -- same score, same labels, same twenty-seed volume-matched deletion --
parametrised by phase, and runs it in two arms:

  xstest-recipe   the primary run's side-specific families verbatim
                  (Qwen: thinking templates; Gemma: refusal openers)
  matched         each side's family chosen by what actually fires on this
                  phase (on OR-Bench both models open with a thinking template;
                  the refusal family fires on 3 of 1,319 Gemma sides)

The phase-3 run is executed first as a referee check and must reproduce the
recorded 0.985 -> 0.645 (same-span keyword), matched 0.966 (sd 0.007), and the
construct arm 0.592 -> 0.428 before any OR-Bench number is written.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results" / "disagree_routing"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
N_BOOT = 2000
N_SEEDS = 20
SPAN = 50
OUT = ROOT / "analysis_results" / "opening_strip_orbench.json"

spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec); sys.modules["agcr"] = agcr; spec.loader.exec_module(agcr)
_iso = importlib.util.spec_from_file_location("strip_mod", HERE / "intervention_strip_openings.py")
strip_mod = importlib.util.module_from_spec(_iso); sys.modules["strip_mod"] = strip_mod; _iso.loader.exec_module(strip_mod)

ARMS = {
    "xstest-recipe": (strip_mod.TARGETED_QWEN_STRIP_PATTERNS, strip_mod.REFUSAL_STRIP_PATTERNS),
    "matched": (strip_mod.THINKING_STRIP_PATTERNS, strip_mod.THINKING_STRIP_PATTERNS),
}


def strip(text, patterns):
    return strip_mod.strip_opening(text, patterns)[0]


def excise_random(text, n_removed, rng):
    if n_removed <= 0 or len(text) <= n_removed:
        return text
    start = int(rng.integers(1, len(text) - n_removed + 1))
    return text[:start] + text[start + n_removed:]


def boot_ci(label, s, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(label[sel])) < 2:
            continue
        out.append(roc_auc_score(label[sel], s[sel]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def load(phase):
    def judge(model):
        rows = json.loads((DATA / f"{phase}_judge_{model}_{JUDGE}.json").read_text())["records"]
        return {r["id"]: int(bool(r["refusal_judge"])) for r in rows if r.get("refusal_judge") is not None}
    ta = {r["id"]: r for r in json.loads((DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
    tb = {r["id"]: r for r in json.loads((DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
    ja, jb = judge("qwen3.5-2b"), judge("gemma-4-e2b")
    ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
    a = [ta[i].get("trace") or "" for i in ids]
    b = [tb[i].get("trace") or "" for i in ids]
    y = np.array([int(ja[i] != jb[i]) for i in ids])
    return ids, a, b, y


def run(phase, arm, pats_a, pats_b, seed_base=1000):
    ids, a_txt, b_txt, y = load(phase)
    labels = {
        "z (same span)": np.array([int(agcr.is_kw(x[:SPAN]) != agcr.is_kw(w[:SPAN])) for x, w in zip(a_txt, b_txt)]),
        "z_comp (off-span only)": np.array([int(agcr.is_kw(x[SPAN:]) != agcr.is_kw(w[SPAN:])) for x, w in zip(a_txt, b_txt)]),
        "y (construct)": y,
    }
    a_str = [strip(x, pats_a) for x in a_txt]
    b_str = [strip(x, pats_b) for x in b_txt]
    rem_a = [len(o) - len(s) for o, s in zip(a_txt, a_str)]
    rem_b = [len(o) - len(s) for o, s in zip(b_txt, b_str)]
    sides_altered = sum(1 for r in rem_a + rem_b if r > 0)
    s_intact = agcr.tfidf_distance(a_txt, b_txt, prefix=SPAN)
    s_strip = agcr.tfidf_distance(a_str, b_str, prefix=SPAN)
    matched = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed_base + seed)
        matched.append(agcr.tfidf_distance([excise_random(x, n, rng) for x, n in zip(a_txt, rem_a)],
                                           [excise_random(x, n, rng) for x, n in zip(b_txt, rem_b)],
                                           prefix=SPAN))
    rows = {}
    for name, lab in labels.items():
        if len(set(lab)) < 2:
            rows[name] = None
            continue
        a_i = float(roc_auc_score(lab, s_intact)); a_s = float(roc_auc_score(lab, s_strip))
        ms = [float(roc_auc_score(lab, s)) for s in matched]
        lo, hi = boot_ci(lab, s_strip, abs(hash(phase + arm + name)) % (2**31))
        rows[name] = {"intact": a_i, "targeted": a_s, "targeted_ci": [lo, hi],
                      "targeted_drop": a_i - a_s, "matched_mean": float(np.mean(ms)),
                      "matched_sd": float(np.std(ms)), "matched_drop": a_i - float(np.mean(ms)),
                      "positives": int(lab.sum())}
    return {"n": len(ids), "sides_altered": sides_altered, "sides": 2 * len(ids),
            "chars_removed": int(sum(rem_a) + sum(rem_b)),
            "chars_total": int(sum(len(x) for x in a_txt + b_txt)), "targets": rows}


def main():
    out = {"span": SPAN, "n_seeds": N_SEEDS, "phases": {}}
    for phase in ("phase3_xstest_full", "phase8_orbench_hard1k"):
        out["phases"][phase] = {}
        for arm, (pa, pb) in ARMS.items():
            r = run(phase, arm, pa, pb)
            out["phases"][phase][arm] = r
            print(f"\n== {phase} / {arm}: n={r['n']}, sides altered {r['sides_altered']}/{r['sides']}, "
                  f"chars removed {r['chars_removed']}/{r['chars_total']} ({r['chars_removed']/r['chars_total']:.3f})")
            for name, t in r["targets"].items():
                if t is None:
                    print(f"  {name:24s} degenerate"); continue
                print(f"  {name:24s} pos {t['positives']:4d}  intact {t['intact']:.3f}  targeted {t['targeted']:.3f} "
                      f"[{t['targeted_ci'][0]:.3f},{t['targeted_ci'][1]:.3f}]  drop {t['targeted_drop']:+.3f} | "
                      f"matched {t['matched_mean']:.3f} (sd {t['matched_sd']:.3f}) drop {t['matched_drop']:+.3f}")
    ref = out["phases"]["phase3_xstest_full"]["xstest-recipe"]["targets"]
    z, y = ref["z (same span)"], ref["y (construct)"]
    assert abs(z["intact"] - 0.985) < 0.002 and abs(z["targeted"] - 0.645) < 0.002, "phase3 keyword arm not reproduced"
    assert abs(z["matched_mean"] - 0.966) < 0.004, "phase3 matched arm not reproduced"
    assert abs(y["intact"] - 0.592) < 0.002 and abs(y["targeted"] - 0.428) < 0.002, "phase3 construct arm not reproduced"
    print("\nreferee check: phase3 reproduced")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
