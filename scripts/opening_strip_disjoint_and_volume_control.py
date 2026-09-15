#!/usr/bin/env python3
"""The opening-strip intervention, re-run against the disjoint proxy and against a
volume-matched control. (2026-08-21)

Two objections, both correct, both cheap to answer.

FIRST.  The strip experiment was scored against z (same span) and z_perp (complete
output), and the second was called span-independent.  It is not: the complete
output contains the scored span.  The disjoint proxy z_comp, the same keyword rule
applied only to text beyond the scored span, is what the mechanism claim actually
needs.  If suppressing the opening costs the score its grip on refusal vocabulary
that appears *later in the output*, the opening genuinely predicted the rest of the
response, and the mechanism reading holds.  If it does not, what the strip removed
was only the opening's agreement with a rule reading the opening.

SECOND.  The repair experiment carries a twenty-seed random-relocation control; the
strip experiment carries none at the text level.  Removing the opening removes
characters, and removing characters perturbs a TF-IDF distance whatever the
characters are.  Without a volume-matched control the drop cannot be attributed to
the opening rather than to the deletion.  So we delete the same number of
characters from the same responses at a random offset instead of at the start, over
twenty seeds, and compare.

The comparison the paper needs is the pair of drops, not either alone:

    targeted  strip the matched opening template
    matched   delete an equal-length run of characters starting at a random offset

  python3 opening_strip_disjoint_and_volume_control.py
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

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)

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


def excise_random(text: str, n_removed: int, rng) -> str:
    """Delete n_removed characters starting at a random offset, never at position 0,
    so the deletion is volume-matched but not opening-matched."""
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


def main() -> None:
    def plain_judge(model):
        rows = json.loads(
            (DATA / f"phase3_xstest_full_judge_{model}_{JUDGE}.json").read_text())["records"]
        return {r["id"]: int(bool(r["refusal_judge"])) for r in rows}

    ta = {r["id"]: r for r in json.loads(
        (DATA / "phase3_xstest_full_traces_qwen3.5-2b.json").read_text())["records"]}
    tb = {r["id"]: r for r in json.loads(
        (DATA / "phase3_xstest_full_traces_gemma-4-e2b.json").read_text())["records"]}
    ja, jb = plain_judge("qwen3.5-2b"), plain_judge("gemma-4-e2b")
    ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))

    a_txt = [ta[i].get("trace") or "" for i in ids]
    b_txt = [tb[i].get("trace") or "" for i in ids]
    y = np.array([int(ja[i] != jb[i]) for i in ids])

    # labels are fixed across every intervention arm
    labels = {
        "z (same span)": np.array([int(agcr.is_kw(a[:SPAN]) != agcr.is_kw(b[:SPAN]))
                                   for a, b in zip(a_txt, b_txt)]),
        "z_perp (whole output)": np.array([int(agcr.is_kw(a) != agcr.is_kw(b))
                                           for a, b in zip(a_txt, b_txt)]),
        "z_comp (off-span only)": np.array([int(agcr.is_kw(a[SPAN:]) != agcr.is_kw(b[SPAN:]))
                                            for a, b in zip(a_txt, b_txt)]),
        "y (construct)": y,
    }

    a_str = [strip_opening(x, STRIP_A) for x in a_txt]
    b_str = [strip_opening(x, STRIP_B) for x in b_txt]
    rem_a = [len(o) - len(s) for o, s in zip(a_txt, a_str)]
    rem_b = [len(o) - len(s) for o, s in zip(b_txt, b_str)]
    tot = sum(len(x) for x in a_txt + b_txt)
    removed = sum(rem_a) + sum(rem_b)
    print(f"n={len(ids)}   targeted strip removes {removed}/{tot} characters "
          f"({removed/tot:.3f}); mean per stripped side "
          f"{np.mean([r for r in rem_a + rem_b if r > 0]):.1f}")
    print(f"sides actually altered: {sum(1 for r in rem_a + rem_b if r > 0)}"
          f"/{len(rem_a) + len(rem_b)}\n")
    for k, v in labels.items():
        print(f"  {k:24s} positives {int(v.sum()):4d}")

    s_intact = agcr.tfidf_distance(a_txt, b_txt, prefix=SPAN)
    s_strip = agcr.tfidf_distance(a_str, b_str, prefix=SPAN)

    # volume-matched control: same per-response deletion length, random offset
    matched = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        a_m = [excise_random(x, n, rng) for x, n in zip(a_txt, rem_a)]
        b_m = [excise_random(x, n, rng) for x, n in zip(b_txt, rem_b)]
        matched.append(agcr.tfidf_distance(a_m, b_m, prefix=SPAN))

    print(f"\n{'target':24s} {'intact':>8s} {'targeted':>9s} {'drop':>8s}"
          f" | {'matched':>9s} {'drop':>8s}  {'excess':>8s}")
    rows = {}
    for name, lab in labels.items():
        a_i = float(roc_auc_score(lab, s_intact))
        a_s = float(roc_auc_score(lab, s_strip))
        ms = [float(roc_auc_score(lab, s)) for s in matched]
        m_mean = float(np.mean(ms))
        lo_s, hi_s = boot_ci(lab, s_strip, abs(hash(name)) % (2**31))
        rows[name] = {
            "intact": a_i, "targeted": a_s, "targeted_drop": a_i - a_s,
            "targeted_ci": [lo_s, hi_s],
            "matched_mean": m_mean, "matched_drop": a_i - m_mean,
            "matched_sd": float(np.std(ms)),
            "matched_range": [float(min(ms)), float(max(ms))],
            "excess_drop": (a_i - a_s) - (a_i - m_mean),
        }
        r = rows[name]
        print(f"{name:24s} {a_i:8.3f} {a_s:9.3f} {r['targeted_drop']:+8.3f}"
              f" | {m_mean:9.3f} {r['matched_drop']:+8.3f} {r['excess_drop']:+8.3f}")

    print(f"\nmatched arm spread over {N_SEEDS} seeds (sd / min-max):")
    for name, r in rows.items():
        print(f"  {name:24s} sd {r['matched_sd']:.3f}  "
              f"[{r['matched_range'][0]:.3f}, {r['matched_range'][1]:.3f}]")

    # The mechanism test is a *paired contrast*: does suppressing the opening cost the
    # off-span proxy more than it costs the construct? A large drop in both would only
    # show the score was entirely opening-driven, which is not the same claim.
    def paired_drop_gap(lab_p, lab_y, seed):
        rng = np.random.default_rng(seed)
        d = []
        for _ in range(N_BOOT):
            sel = rng.integers(0, len(s_intact), len(s_intact))
            if len(set(lab_p[sel])) < 2 or len(set(lab_y[sel])) < 2:
                continue
            dp = roc_auc_score(lab_p[sel], s_intact[sel]) - roc_auc_score(lab_p[sel], s_strip[sel])
            dy = roc_auc_score(lab_y[sel], s_intact[sel]) - roc_auc_score(lab_y[sel], s_strip[sel])
            d.append(dp - dy)
        return (float(np.mean(d)), float(np.percentile(d, 2.5)),
                float(np.percentile(d, 97.5)))

    print("\nmechanism contrast (proxy drop minus construct drop, paired bootstrap):")
    contrasts = {}
    for key in ("z (same span)", "z_perp (whole output)", "z_comp (off-span only)"):
        m, lo, hi = paired_drop_gap(labels[key], y, abs(hash("c" + key)) % (2**31))
        contrasts[key] = {"mean": m, "ci": [lo, hi]}
        verdict = "separable from zero" if lo > 0 else "NOT separable from zero"
        print(f"  {key:24s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]  {verdict}")

    print("\nreading:")
    ex_z = rows["z (same span)"]["excess_drop"]
    ex_c = rows["z_comp (off-span only)"]["excess_drop"]
    ex_y = rows["y (construct)"]["excess_drop"]
    print(f"  excess drop attributable to the opening rather than to deleted volume:")
    print(f"    same-span proxy  {ex_z:+.3f}")
    print(f"    disjoint proxy   {ex_c:+.3f}")
    print(f"    construct        {ex_y:+.3f}")
    c_lo = contrasts["z_comp (off-span only)"]["ci"][0]
    if ex_c > 0.5 * ex_z and c_lo > 0:
        print("  the opening carried the score's grip on refusal vocabulary appearing beyond")
        print("  the scored span, over and above deleted volume AND over and above what the")
        print("  strip costs the construct. The mechanism reading survives the stronger control.")
    elif ex_c > 0.5 * ex_z:
        print("  the strip costs the off-span proxy far more than deleted volume does, but the")
        print("  paired contrast against the construct does not separate from zero. What is")
        print("  established is that the score was opening-driven; that the opening predicted")
        print("  the proxy specifically, rather than everything, is NOT established.")
    elif ex_c < 0.25 * ex_z:
        print("  once volume is matched and the proxy is read strictly off-span, the opening")
        print("  explains little: what the strip removed was the opening's agreement with a")
        print("  rule reading the opening. The mechanism claim must be narrowed to that.")
    else:
        print("  partial: the opening carries some of the off-span signal. Report the")
        print("  decomposition rather than the mechanism reading.")

    out = {"n": len(ids), "span": SPAN, "n_seeds": N_SEEDS,
           "mechanism_contrast": contrasts,
           "chars_removed": removed, "chars_total": tot,
           "sides_altered": int(sum(1 for r in rem_a + rem_b if r > 0)),
           "targets": rows}
    path = ROOT / "analysis_results" / "opening_strip_disjoint_and_volume_control.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
