#!/usr/bin/env python3
"""Full-verdict stability under judge-positive down-sampling (round 19, item 3).

For each random draw that keeps 13 of the 46 judge positives (the human rate),
re-run the whole verdict function on the down-sampled construct: power
precondition, orientation (95% bootstrap CI below 0.5), equivalence (90% CI in
[0.45,0.55]), permutation null of Delta_dis within strata of y, magnitude bands.
Reduced resampling (N_BOOT, N_PERM) keeps the run to minutes; the recorded
verdict on the full labels is reproduced first as a referee check.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent.parent; HERE=ROOT/'scripts'
spec=importlib.util.spec_from_file_location("agcr", HERE/"analyze_rlc_composite_router.py"); agcr=importlib.util.module_from_spec(spec); sys.modules["agcr"]=agcr; spec.loader.exec_module(agcr)
DATA=ROOT/"results"/"disagree_routing"; JUDGE="anthropic_claude-haiku-4-5-20251001"; phase="phase3_xstest_full"; SPAN=50
N_DRAWS=200; N_BOOT=400; N_PERM=300; HUMAN_RATE=0.028
def judge(model):
    rows=json.loads((DATA/f"{phase}_judge_{model}_{JUDGE}.json").read_text())["records"]; return {r["id"]:int(bool(r["refusal_judge"])) for r in rows}
ta={r["id"]:r for r in json.loads((DATA/f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
tb={r["id"]:r for r in json.loads((DATA/f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
ja,jb=judge("qwen3.5-2b"),judge("gemma-4-e2b"); ids=sorted(set(ta)&set(tb)&set(ja)&set(jb))
a_txt=[ta[i].get("trace") or "" for i in ids]; b_txt=[tb[i].get("trace") or "" for i in ids]
y=np.array([int(ja[i]!=jb[i]) for i in ids]); z=np.array([int(agcr.is_kw(a[:SPAN])!=agcr.is_kw(b[:SPAN])) for a,b in zip(a_txt,b_txt)])
z_c=np.array([int(agcr.is_kw(a[SPAN:])!=agcr.is_kw(b[SPAN:])) for a,b in zip(a_txt,b_txt)])
s=agcr.tfidf_distance(a_txt,b_txt,prefix=SPAN); n=len(ids); auc_z=roc_auc_score(z,s); auc_zc=roc_auc_score(z_c,s)
rng=np.random.default_rng(20260907)
def auc(yy,ss):
    return roc_auc_score(yy,ss) if 0<yy.sum()<len(yy) else np.nan
def verdict(yy,seed):
    r=np.random.default_rng(seed)
    if yy.sum()<10: return "UNDECIDABLE (power)"
    a_y=auc(yy,s); boots=[]
    for _ in range(N_BOOT):
        idx=r.integers(0,n,n); v=auc(yy[idx],s[idx])
        if not np.isnan(v): boots.append(v)
    boots=np.array(boots); lo95,hi95=np.percentile(boots,[2.5,97.5]); lo90,hi90=np.percentile(boots,[5,95])
    d_dis=abs(auc_zc-0.5)-abs(a_y-0.5); d_auc=abs(auc_z-0.5)-abs(a_y-0.5)
    if 0.45<=lo90 and hi90<=0.55:
        return "SCORE FAILURE" if abs(auc_zc-0.5)>=0.10 else ("CONTAINMENT" if d_auc>=0.15 else "UNDECIDABLE (no signal)")
    if lo95<=0.5<=hi95: return "UNDECIDABLE (equivalence)"
    # permutation null within strata of y
    null=[]
    for _ in range(N_PERM):
        zp=z_c.copy()
        for cls in (0,1):
            idx=np.flatnonzero(yy==cls); zp[idx]=zp[r.permutation(idx)]
        v=auc(zp,s); null.append(abs(v-0.5)-abs(a_y-0.5))
    p95=np.percentile(null,95)
    if d_dis<=p95: return "CONTAINMENT" if d_auc>=0.15 else "ALIGNED"
    return "DIVERGENCE" if d_dis>=0.15 else ("CAUTION" if d_dis>=0.10 else "ALIGNED")
ref=verdict(y,0); print("referee (full labels):",ref); assert ref=="CAUTION", ref
pos=np.flatnonzero(y==1); keep_n=max(1,int(round(HUMAN_RATE*n))); assert (n,len(pos),keep_n)==(450,46,13)
from collections import Counter
tally=Counter(); dd=[]
for k in range(N_DRAWS):
    keep=rng.choice(pos,size=keep_n,replace=False); yr=np.zeros_like(y); yr[keep]=1
    v=verdict(yr,1000+k); tally[v]+=1; dd.append(abs(auc_zc-0.5)-abs(auc(yr,s)-0.5))
    if k%20==0: print(k,dict(tally),flush=True)
out={"n_draws":N_DRAWS,"n_boot":N_BOOT,"n_perm":N_PERM,"kept_positives":keep_n,"referee_full_labels":ref,"verdict_shares":{k:v/N_DRAWS for k,v in tally.items()},"frac_ddis_ge_0.10":float(np.mean(np.array(dd)>=0.10))}
print(json.dumps(out,indent=1)); (ROOT/"analysis_results"/"downsample_full_verdict.json").write_text(json.dumps(out,indent=1)); print("wrote")
