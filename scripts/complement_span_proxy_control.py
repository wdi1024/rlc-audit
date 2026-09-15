#!/usr/bin/env python3
"""Is Delta_perp actually a span-independent control? (2026-08-21)

A reviewer points out that it is not, and the objection is correct.  We defined
z_perp as the contract's own proxy rule re-evaluated over the *complete output*,
and called the result span-independent.  But the complete output contains the
scored span.  If "I cannot" appears in the first 50 characters then z = 1 and
z_perp = 1 by the same evidence, so z_perp dilutes the geometry the score shares
with the proxy rather than removing it.  The name over-claims and the notation
(the perpendicular sign) over-claims harder.

The control the objection asks for evaluates the same rule on the *complement*
of the scored span -- the text the score provably never reads:

    z        rule on x[:span]                    the contract as written
    z_perp   rule on x                           the published control (partial)
    z_comp   rule on x[span:]                    disjoint from the score

Only the third is orthogonal in the sense the symbol suggests.  This script
computes all three for every contract the paper audits and reports what changes.

Two things are measured rather than assumed:

  1.  How much of the surviving certificate survives the *stronger* control.
      Delta_comp = AUC(s,z_comp) - AUC(s,y), with a paired bootstrap interval.

  2.  Whether the complement is even non-degenerate.  Where an output is shorter
      than the span, the complement is empty and the rule sees nothing, which
      manufactures negatives.  The share of empty complements is reported per
      contract and any contract where it is large cannot carry the control.

Two preconditions gate every row, because the first version of this script
reported six survivors and two of them were artefacts of exactly the failures the
paper warns about elsewhere:

  power        a contract with three semantic positives cannot support an AUC
               against the construct.  Rows below MIN_POSITIVES are reported but
               excluded from every count.
  orientation  AdvBench reaches AUC(s,y) = 0.290 on those three positives, so the
               raw gap is large because the construct ranks *below* chance, not
               because the proxy ranks above it.  All deltas are therefore
               reported in the orientation-robust form the paper already uses,
               Delta_abs = |AUC(s,z) - 0.5| - |AUC(s,y) - 0.5|, and survival is
               judged on that.

The script also stratifies by contract family, because the same reviewer notes
that the paper's headline median pools eleven span-sweep rows drawn from two
tasks with seven refusal settings, and the pooled median is dominated by the
near-zero sweep rows where the ratio is least stable.

  python3 complement_span_proxy_control.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results" / "disagree_routing"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
N_BOOT = 2000
REFUSAL_SPAN = 50
MIN_POSITIVES = 10      # fewest semantic positives an AUC may rest on
MAX_EMPTY_COMPLEMENT = 0.15  # above this the complement rule mostly sees nothing

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)

REFUSAL_SETTINGS = [
    ("phase3_xstest_full", "XSTest 450 (primary run)"),
    ("phase4_advbench", "AdvBench 520"),
    ("phase4_simplesafety", "SimpleSafety 100"),
    ("phase5_xstest100_512tok", "XSTest 100 / 512tok"),
    ("phase5_advbench100_512tok", "AdvBench 100 / 512tok"),
    ("phase8_orbench_hard1k", "OR-Bench hard 1k"),
    ("phase15_jailbreakbench", "JailbreakBench"),
]
CORRECTNESS = [("phase14_gsm8k_2k", "GSM8K"), ("phase14_hotpot_2k", "HotpotQA")]
CORRECTNESS_SPANS = [50, 80, 120, 160, 200, 300, 600]


def boot_ci(lab, y, s, seed, orient=False):
    """Paired bootstrap interval on the gap. With orient=True the orientation-robust
    form is bootstrapped instead, so a construct ranking below chance cannot inflate it."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(lab[sel])) < 2 or len(set(y[sel])) < 2:
            continue
        az = roc_auc_score(lab[sel], s[sel])
        ay = roc_auc_score(y[sel], s[sel])
        d.append((abs(az - 0.5) - abs(ay - 0.5)) if orient else (az - ay))
    if not d:
        return None, None
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def make_row(label, family, span, s, z, z_perp, z_comp, y, empty_share, seed):
    if len(set(z)) < 2 or len(set(y)) < 2:
        return None
    a_z = float(roc_auc_score(z, s))
    a_y = float(roc_auc_score(y, s))
    d_auc = a_z - a_y

    def arm(lab, tag):
        if len(set(lab)) < 2:
            return {"auc": None, "delta": None, "ci": [None, None],
                    "delta_abs": None, "ci_abs": [None, None], "degenerate": True}
        a = float(roc_auc_score(lab, s))
        lo, hi = boot_ci(lab, y, s, seed + hash(tag) % 1000)
        alo, ahi = boot_ci(lab, y, s, seed + hash(tag) % 1000, orient=True)
        return {"auc": a, "delta": a - a_y, "ci": [lo, hi],
                "delta_abs": abs(a - 0.5) - abs(a_y - 0.5), "ci_abs": [alo, ahi],
                "degenerate": False}

    perp, comp = arm(z_perp, "perp"), arm(z_comp, "comp")

    # A low Delta_dis has two readings: the certificate really was not containment, or the
    # proxy rule simply does not fire off-span and z_comp is close to constant. Prevalence and
    # kappa(z_comp, y) separate them, so both are reported for every row.
    def kappa(a, b):
        n_ = len(a)
        po = float((a == b).mean())
        pe = float((a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean())))
        return (po - pe) / (1 - pe) if pe < 1 else None

    comp["prevalence"] = float(z_comp.mean())
    comp["kappa_vs_y"] = kappa(z_comp, y) if len(set(z_comp)) > 1 else None
    comp["auc_only"] = comp["auc"]
    perp["prevalence"] = float(z_perp.mean())
    perp["kappa_vs_y"] = kappa(z_perp, y) if len(set(z_perp)) > 1 else None
    underpowered = int(y.sum()) < MIN_POSITIVES or int(len(y) - y.sum()) < MIN_POSITIVES
    empty_gate = empty_share > MAX_EMPTY_COMPLEMENT
    return {
        "contract": label, "family": family, "span": span, "n": int(len(s)),
        "underpowered": underpowered, "complement_degenerate": empty_gate,
        "delta_abs": abs(a_z - 0.5) - abs(a_y - 0.5),
        "z_positives": int(z.sum()), "z_perp_positives": int(z_perp.sum()),
        "z_comp_positives": int(z_comp.sum()), "y_positives": int(y.sum()),
        "empty_complement_share": empty_share,
        "auc_z": a_z, "auc_y": a_y, "delta_auc": d_auc,
        "perp": perp, "comp": comp,
        "share_removed_by_perp": (1.0 - perp["delta"] / d_auc)
            if (d_auc > 0 and perp["delta"] is not None) else None,
        "share_removed_by_comp": (1.0 - comp["delta"] / d_auc)
            if (d_auc > 0 and comp["delta"] is not None) else None,
    }


def refusal_rows() -> list[dict]:
    out = []
    for phase, label in REFUSAL_SETTINGS:
        try:
            def judge(model):
                rows = json.loads(
                    (DATA / f"{phase}_judge_{model}_{JUDGE}.json").read_text())["records"]
                return {r["id"]: int(bool(r["refusal_judge"])) for r in rows}

            ta = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
            tb = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
            ja, jb = judge("qwen3.5-2b"), judge("gemma-4-e2b")
        except (FileNotFoundError, KeyError) as exc:
            print(f"  {label}: skipped ({type(exc).__name__})", flush=True)
            continue
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        a_txt = [ta[i].get("trace") or "" for i in ids]
        b_txt = [tb[i].get("trace") or "" for i in ids]
        y = np.array([int(ja[i] != jb[i]) for i in ids])
        sp = REFUSAL_SPAN
        z = np.array([int(agcr.is_kw(a[:sp]) != agcr.is_kw(b[:sp]))
                      for a, b in zip(a_txt, b_txt)])
        z_perp = np.array([int(agcr.is_kw(a) != agcr.is_kw(b))
                           for a, b in zip(a_txt, b_txt)])
        z_comp = np.array([int(agcr.is_kw(a[sp:]) != agcr.is_kw(b[sp:]))
                           for a, b in zip(a_txt, b_txt)])
        empty = float(np.mean([len(t) <= sp for t in a_txt + b_txt]))
        s = agcr.tfidf_distance(a_txt, b_txt, prefix=sp)
        r = make_row(label, "refusal", sp, s, z, z_perp, z_comp, y, empty,
                     abs(hash(phase)) % (2**31))
        if r:
            out.append(r)
            c = r["comp"]
            cs = (f"{c['delta']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]"
                  if not c["degenerate"] else "degenerate")
            print(f"  {label:28s} dAUC {r['delta_auc']:+.3f}  "
                  f"dPerp {r['perp']['delta']:+.3f}  dComp {cs}", flush=True)
    return out


def correctness_rows() -> list[dict]:
    def cos(a, b):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(a + b)
        xa, xb = v.transform(a), v.transform(b)
        num = np.asarray(xa.multiply(xb).sum(1)).ravel()
        na = np.sqrt(np.asarray(xa.multiply(xa).sum(1)).ravel())
        nb = np.sqrt(np.asarray(xb.multiply(xb).sum(1)).ravel())
        return 1 - num / np.maximum(na * nb, 1e-9)

    out = []
    for phase, dom in CORRECTNESS:
        ta = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
        tb = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
        ja = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_correctness_qwen3.5-2b_{JUDGE}.json").read_text())["records"]}
        jb = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_correctness_gemma-4-e2b_{JUDGE}.json").read_text())["records"]}
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        a_txt = [ta[i].get("trace") or "" for i in ids]
        b_txt = [tb[i].get("trace") or "" for i in ids]
        gold = {i: (ja[i].get("gold_answer") or "").strip().lower() for i in ids}
        y = np.array([int(bool(ja[i].get("correct_judge")) != bool(jb[i].get("correct_judge")))
                      for i in ids])
        z_perp = np.array([int((gold[i] in a.lower()) != (gold[i] in b.lower()))
                           for i, a, b in zip(ids, a_txt, b_txt)])
        for span in CORRECTNESS_SPANS:
            pa, pb = [t[:span] for t in a_txt], [t[:span] for t in b_txt]
            ca, cb = [t[span:] for t in a_txt], [t[span:] for t in b_txt]
            z = np.array([int((gold[i] in a.lower()) != (gold[i] in b.lower()))
                          for i, a, b in zip(ids, pa, pb)])
            z_comp = np.array([int((gold[i] in a.lower()) != (gold[i] in b.lower()))
                               for i, a, b in zip(ids, ca, cb)])
            empty = float(np.mean([len(t) <= span for t in a_txt + b_txt]))
            s = cos(pa, pb)
            r = make_row(f"{dom}, {span}-char span", "correctness", span, s,
                         z, z_perp, z_comp, y, empty,
                         abs(hash(phase + str(span))) % (2**31))
            if r:
                out.append(r)
                c = r["comp"]
                cs = (f"{c['delta']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]"
                      if not c["degenerate"] else "degenerate")
                print(f"  {r['contract']:28s} dAUC {r['delta_auc']:+.3f}  "
                      f"dPerp {r['perp']['delta']:+.3f}  dComp {cs}", flush=True)
    return out


def survives(row, key) -> bool:
    """A gap survives only if the contract can carry the test at all: enough semantic
    positives, a non-degenerate complement, and an orientation-robust interval above zero."""
    a = row[key]
    if row["underpowered"] or a["degenerate"]:
        return False
    if key == "comp" and row["complement_degenerate"]:
        return False
    return a["ci_abs"][0] is not None and a["ci_abs"][0] > 0


def main() -> None:
    print("=== refusal settings ===", flush=True)
    rows = refusal_rows()
    print("\n=== correctness contracts ===", flush=True)
    rows += correctness_rows()

    print(f"\n{'contract':26s} {'y+':>4s} {'emptyC':>7s} {'dAUC':>7s} {'dPerp':>7s} {'dComp':>7s}"
          f"  {'dComp_abs (95% CI)':>24s}  gate")
    for r in rows:
        c = r["comp"]
        da = f"{c['delta_abs']:+.3f} [{c['ci_abs'][0]:+.3f},{c['ci_abs'][1]:+.3f}]" \
            if not c["degenerate"] else "degenerate"
        gate = []
        if r["underpowered"]:
            gate.append(f"underpowered y+={r['y_positives']}")
        if r["complement_degenerate"]:
            gate.append(f"empty complement {r['empty_complement_share']:.0%}")
        print(f"{r['contract'][:26]:26s} {r['y_positives']:4d} "
              f"{r['empty_complement_share']:7.3f} {r['delta_auc']:+7.3f} "
              f"{r['perp']['delta']:+7.3f} "
              + (f"{c['delta']:+7.3f}" if c["delta"] is not None else "     --")
              + f"  {da:>24s}  {'; '.join(gate)}")

    usable = [r for r in rows if not r["underpowered"]]
    pos = [r for r in usable if r["delta_abs"] > 0]
    s_perp = [r for r in pos if survives(r, "perp")]
    s_comp = [r for r in pos if survives(r, "comp")]
    dropped_power = [r for r in rows if r["underpowered"]]
    dropped_empty = [r for r in usable if r["complement_degenerate"]]

    print(f"\n=== {len(rows)} contracts ===")
    print(f"excluded, fewer than {MIN_POSITIVES} semantic positives: {len(dropped_power)}"
          + (f" ({', '.join(r['contract'] for r in dropped_power)})" if dropped_power else ""))
    print(f"excluded from the disjoint arm, complement empty on >"
          f"{MAX_EMPTY_COMPLEMENT:.0%} of sides: {len(dropped_empty)}"
          + (f" ({', '.join(r['contract'] for r in dropped_empty)})" if dropped_empty else ""))
    print(f"usable contracts with a positive orientation-robust gap: {len(pos)}")
    print(f"  survive the published partial control (z_perp): {len(s_perp)}")
    print(f"  survive the disjoint control (z_comp):          {len(s_comp)}")
    for r in s_comp:
        c = r["comp"]
        print(f"      {r['contract']:26s} dComp_abs {c['delta_abs']:+.3f} "
              f"[{c['ci_abs'][0]:+.3f},{c['ci_abs'][1]:+.3f}]  n={r['n']}, y+={r['y_positives']}")

    print("\n=== stratified by family ===")
    for fam in ("refusal", "correctness"):
        fr = [r for r in usable if r["family"] == fam]
        fp = [r for r in fr if r["delta_abs"] > 0]
        sp_ = [r for r in fp if survives(r, "perp")]
        sc_ = [r for r in fp if survives(r, "comp")]
        print(f"  {fam:12s} usable={len(fr):2d}  positive={len(fp):2d}  "
              f"survive perp={len(sp_)}  survive comp={len(sc_)}")
        share = [1.0 - r["comp"]["delta_abs"] / r["delta_abs"] for r in fp
                 if r["comp"]["delta_abs"] is not None and not r["complement_degenerate"]]
        if share:
            print(f"               median share of the gap removed by the disjoint "
                  f"control: {np.median(share):.2f} (n={len(share)})")
        g = [r["delta_abs"] for r in fp]
        if g:
            print(f"               orientation-robust gaps span {min(g):+.3f} to {max(g):+.3f}")

    ok = [r for r in usable if r["comp"]["delta_abs"] is not None
          and not r["complement_degenerate"]]
    if len(ok) > 2:
        rho, p = spearmanr([r["delta_abs"] for r in ok],
                           [r["comp"]["delta_abs"] for r in ok])
        print(f"\nSpearman(dAUC_abs, dComp_abs) = {rho:+.3f} (p={p:.3f}) over {len(ok)} contracts")

    prim = next((r for r in rows if r["contract"].startswith("XSTest 450")), None)
    if prim:
        c = prim["comp"]
        print("\nthe primary contract, which Section 5 dissects:")
        print(f"  dAUC {prim['delta_auc']:+.3f} -> dPerp {prim['perp']['delta']:+.3f} "
              f"-> dComp_abs {c['delta_abs']:+.3f} [{c['ci_abs'][0]:+.3f},{c['ci_abs'][1]:+.3f}]")
        if c["ci_abs"][0] is not None and c["ci_abs"][0] <= 0:
            print("  its certificate does NOT survive a proxy read strictly off the scored span.")
            print("  The partial control credited it with two-thirds surviving; the disjoint")
            print("  control cannot distinguish what remains from zero. This must be reported,")
            print("  and the dissected example is no longer the paper's strongest certificate.")

    out = {"min_positives": MIN_POSITIVES, "max_empty_complement": MAX_EMPTY_COMPLEMENT,
           "n_contracts": len(rows), "n_usable": len(usable), "n_positive": len(pos),
           "n_survive_perp": len(s_perp), "n_survive_comp": len(s_comp),
           "surviving_comp": [r["contract"] for r in s_comp],
           "excluded_underpowered": [r["contract"] for r in dropped_power],
           "excluded_empty_complement": [r["contract"] for r in dropped_empty],
           "rows": rows}
    path = ROOT / "analysis_results" / "complement_span_proxy_control.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
