#!/usr/bin/env python3
"""Frozen-rule audit of a held-out contract (2026-09-08).

Algorithm 1 and every threshold were fixed before this contract was read; the
protocol, the selection rule and the constants are in
notes/holdout_preregistration_20260908.md, written before this script was run.

The verdict function is the one from downsample_full_verdict.py, unchanged except
that it takes the hold-out arrays and that the complement and resolution
preconditions -- applied outside that function there -- are applied here too.

  python3 scripts/holdout_frozen_rule.py
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "scripts"
DATA = ROOT / "results" / "disagree_routing"
spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec); sys.modules["agcr"] = agcr; spec.loader.exec_module(agcr)

# ---- pre-registered selection: first panel pair with traces + Haiku labels ----
PHASE = "phase_colab_qwen_gemma_big_xstest450_tagged768_thinkingoff_greedy"
SIDE_A, SIDE_B = "qwen3.5-9b", "gemma-2-9b-it"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
SPAN = 50

# ---- frozen constants (see the pre-registration table) -----------------------
MIN_POS = 10
COMPLEMENT_MIN = 0.85
MODAL_MAX = 0.5
STILL_RANKED = 0.10
CAUTION_BAND, DIVERGENCE_BAND = 0.10, 0.15
N_BOOT, N_PERM = 2000, 2000

def traces(model):
    recs = json.loads((DATA / f"{PHASE}_traces_{model}.json").read_text())["records"]
    return {r["id"]: (r.get("trace") or "") for r in recs}

def judge(model):
    recs = json.loads((DATA / f"{PHASE}_judge_final_{model}_{JUDGE}.json").read_text())["records"]
    return {r["id"]: int(bool(r["refusal_judge"])) for r in recs}

ta, tb = traces(SIDE_A), traces(SIDE_B)
ja, jb = judge(SIDE_A), judge(SIDE_B)
ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
a_txt = [ta[i] for i in ids]; b_txt = [tb[i] for i in ids]
n = len(ids)

y   = np.array([int(ja[i] != jb[i]) for i in ids])
z   = np.array([int(agcr.is_kw(a[:SPAN]) != agcr.is_kw(b[:SPAN])) for a, b in zip(a_txt, b_txt)])
z_c = np.array([int(agcr.is_kw(a[SPAN:]) != agcr.is_kw(b[SPAN:])) for a, b in zip(a_txt, b_txt)])
s   = agcr.tfidf_distance(a_txt, b_txt, prefix=SPAN)

def auc(yy, ss):
    return roc_auc_score(yy, ss) if 0 < yy.sum() < len(yy) else np.nan

# ---- preconditions -----------------------------------------------------------
n_pos = int(y.sum())
complement_nonempty = float(np.mean([len(t) > SPAN for t in a_txt + b_txt]))
vals, counts = np.unique(s, return_counts=True)
modal_share = float(counts.max() / n)

pre = []
if n_pos < MIN_POS: pre.append("minimum-positive-count")
if complement_nonempty < COMPLEMENT_MIN: pre.append("complement")
if modal_share > MODAL_MAX: pre.append("resolution")

auc_z  = auc(z, s)   if 0 < z.sum() < n   else float("nan")
auc_zc = auc(z_c, s) if 0 < z_c.sum() < n else float("nan")

out = {
    "contract": {"pair": f"{SIDE_A}/{SIDE_B}", "phase": PHASE, "judge": JUDGE,
                 "span_chars": SPAN, "n": n, "construct_positives": n_pos,
                 "z_positives": int(z.sum()), "z_c_positives": int(z_c.sum())},
    "preconditions": {"complement_nonempty_share": complement_nonempty,
                      "modal_score_share": modal_share, "failed": pre},
    "auc": {"AUC_s_z": auc_z, "AUC_s_zc": auc_zc},
}

if pre:
    out["verdict"] = f"UNDECIDABLE ({pre[0]})"
    print(json.dumps(out, indent=1))
    (ROOT / "analysis_results" / "holdout_frozen_rule.json").write_text(json.dumps(out, indent=1))
    print("wrote"); sys.exit(0)

# ---- verdict function, frozen ------------------------------------------------
r = np.random.default_rng(20260908)
a_y = auc(y, s)
boots = []
for _ in range(N_BOOT):
    idx = r.integers(0, n, n); v = auc(y[idx], s[idx])
    if not np.isnan(v): boots.append(v)
boots = np.array(boots)
lo95, hi95 = np.percentile(boots, [2.5, 97.5]); lo90, hi90 = np.percentile(boots, [5, 95])
d_dis = abs(auc_zc - 0.5) - abs(a_y - 0.5)
d_auc = abs(auc_z - 0.5) - abs(a_y - 0.5)

# paired bootstrap interval on Delta_dis
dboot = []
for _ in range(N_BOOT):
    idx = r.integers(0, n, n)
    va, vb = auc(z_c[idx], s[idx]), auc(y[idx], s[idx])
    if not (np.isnan(va) or np.isnan(vb)): dboot.append(abs(va - 0.5) - abs(vb - 0.5))
d_lo, d_hi = np.percentile(dboot, [2.5, 97.5])

null = []
for _ in range(N_PERM):
    zp = z_c.copy()
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls); zp[idx] = zp[r.permutation(idx)]
    v = auc(zp, s); null.append(abs(v - 0.5) - abs(a_y - 0.5))
p95 = float(np.percentile(null, 95))
p_val = float(np.mean(np.array(null) >= d_dis))

if 0.45 <= lo90 and hi90 <= 0.55:
    verdict = ("SCORE FAILURE" if abs(auc_zc - 0.5) >= STILL_RANKED
               else ("CONTAINMENT" if d_auc >= DIVERGENCE_BAND else "UNDECIDABLE (no signal)"))
elif lo95 <= 0.5 <= hi95:
    verdict = "UNDECIDABLE (equivalence)"
elif d_dis <= p95:
    verdict = "CONTAINMENT" if d_auc >= DIVERGENCE_BAND else "NO RESIDUAL FLAG"
else:
    verdict = ("DIVERGENCE" if d_dis >= DIVERGENCE_BAND
               else ("CAUTION" if d_dis >= CAUTION_BAND else "NO RESIDUAL FLAG"))

out["continuous"] = {
    "AUC_s_y": a_y, "AUC_s_y_CI95": [float(lo95), float(hi95)],
    "AUC_s_y_CI90": [float(lo90), float(hi90)],
    "Delta_dis": float(d_dis), "Delta_dis_CI95": [float(d_lo), float(d_hi)],
    "Delta_AUC": float(d_auc), "null_p95": p95, "permutation_p": p_val,
}
out["verdict"] = verdict
print(json.dumps(out, indent=1))
(ROOT / "analysis_results" / "holdout_frozen_rule.json").write_text(json.dumps(out, indent=1))
print("wrote")
