#!/usr/bin/env python3
"""What is the surviving certificate made of? (2026-08-20)

On the primary refusal contract two-thirds of the proxy certificate survives moving
the proxy off the scored span ($\\Delta_perp = +0.263$).  That rules out the
objection that the agreement was manufactured by co-truncation, but it leaves two
readings standing, and the paper has no right to the first without separating them:

  (M) the mechanism reading.  The opening span carries evidence about whether either
      model will use refusal vocabulary anywhere in its output.  Score and proxy
      succeed together because the opening predicts the whole output's surface form.

  (L) the label reading.  The keyword rule is simply a poor measure of the refusal
      construct, everywhere and independently of span, so any score that reads any
      text will track it better than it tracks the judge.  This is already known.

The two make opposite predictions under the intervention the paper already runs.
Suppressing opening templates with labels held fixed should, under (M), cost the
score its grip on the *full-output* keyword label as well, because the opening was
what carried it; under (L) it should not, because the keyword label's divergence
from the construct has nothing to do with the opening.

So we re-run the opening strip and score it against z_perp rather than z_prefix:

    AUC(s_stripped, z_perp)  vs  AUC(s_intact, z_perp)

and report the drop next to the already-published drop against z_prefix
(0.985 -> 0.645) and the construct's response.

  python3 opening_strip_span_independent.py
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

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)

# Reuse the published intervention's own strip rules rather than re-inventing one, so
# that the drop against z_prefix here reproduces the 0.985 -> 0.645 already reported.
_iso = importlib.util.spec_from_file_location(
    "strip_mod", HERE / "intervention_strip_openings.py")
strip_mod = importlib.util.module_from_spec(_iso)
sys.modules["strip_mod"] = strip_mod
_iso.loader.exec_module(strip_mod)

STRIP_A = strip_mod.TARGETED_QWEN_STRIP_PATTERNS
STRIP_B = strip_mod.REFUSAL_STRIP_PATTERNS


def strip_opening(text: str, patterns) -> str:
    out, _ = strip_mod.strip_opening(text, patterns)
    return out


def boot_auc(label: np.ndarray, s: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(label[sel])) < 2:
            continue
        out.append(roc_auc_score(label[sel], s[sel]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    def plain_judge(phase, model):
        rows = json.loads((DATA / f"{phase}_judge_{model}_{JUDGE}.json").read_text())["records"]
        return {r["id"]: int(bool(r["refusal_judge"])) for r in rows}

    phase = "phase3_xstest_full"
    ta = {r["id"]: r for r in json.loads(
        (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
    tb = {r["id"]: r for r in json.loads(
        (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
    ja, jb = plain_judge(phase, "qwen3.5-2b"), plain_judge(phase, "gemma-4-e2b")
    ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))

    a_txt = [ta[i].get("trace") or "" for i in ids]
    b_txt = [tb[i].get("trace") or "" for i in ids]
    y = np.array([int(ja[i] != jb[i]) for i in ids])

    # labels are held fixed across the intervention, exactly as in the published strip
    z_pre = np.array([int(agcr.is_kw(a[:50]) != agcr.is_kw(b[:50]))
                      for a, b in zip(a_txt, b_txt)])
    z_perp = np.array([int(agcr.is_kw(a) != agcr.is_kw(b)) for a, b in zip(a_txt, b_txt)])

    a_str = [strip_opening(x, STRIP_A) for x in a_txt]
    b_str = [strip_opening(x, STRIP_B) for x in b_txt]
    kept = float(np.mean([len(x) for x in a_str + b_str])
                 / max(1e-9, np.mean([len(x) for x in a_txt + b_txt])))
    empty = float(np.mean([not x.strip() for x in a_str + b_str]))

    s_intact = agcr.tfidf_distance(a_txt, b_txt, prefix=50)
    s_strip = agcr.tfidf_distance(a_str, b_str, prefix=50)

    print(f"primary run, n={len(ids)}   y+={int(y.sum())}  "
          f"z_prefix+={int(z_pre.sum())}  z_perp+={int(z_perp.sum())}")
    print(f"opening strip keeps {kept:.3f} of characters; {empty:.3f} of sides become empty\n")

    rows = {}
    print(f"{'target':28s} {'intact':>8s} {'stripped':>9s} {'drop':>8s}   95% CI on stripped")
    for name, lab in (("z_prefix (same span)", z_pre),
                      ("z_perp (full output)", z_perp),
                      ("y (construct)", y)):
        a_i = float(roc_auc_score(lab, s_intact))
        a_s = float(roc_auc_score(lab, s_strip))
        lo, hi = boot_auc(lab, s_strip, abs(hash(name)) % (2**31))
        rows[name] = {"intact": a_i, "stripped": a_s, "drop": a_i - a_s,
                      "stripped_ci": [lo, hi]}
        print(f"{name:28s} {a_i:8.3f} {a_s:9.3f} {a_i-a_s:+8.3f}   [{lo:+.3f},{hi:+.3f}]")

    d_pre = rows["z_prefix (same span)"]["drop"]
    d_perp = rows["z_perp (full output)"]["drop"]
    d_y = rows["y (construct)"]["drop"]
    print("\nreading:")
    if d_perp > 0.5 * d_pre and d_perp > 2 * abs(d_y):
        print("  the opening carried the full-output keyword signal too: stripping it costs the")
        print("  span-independent proxy nearly as much as the same-span one, and far more than it")
        print("  costs the construct. This is the mechanism reading (M).")
    elif d_perp < 0.25 * d_pre:
        print("  the opening carried only the same-span signal: the span-independent proxy survives")
        print("  the strip, so what Delta_perp measures is a label-quality gap that has nothing to")
        print("  do with the opening. This is the label reading (L), and the paper should say so.")
    else:
        print("  intermediate: the opening carries part of the full-output keyword signal.")
        print("  Report the decomposition rather than either clean reading.")

    out = {"n": len(ids), "chars_kept": kept, "empty_share": empty,
           "y_positives": int(y.sum()), "z_prefix_positives": int(z_pre.sum()),
           "z_perp_positives": int(z_perp.sum()), "targets": rows}
    path = ROOT / "analysis_results" / "opening_strip_span_independent.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
