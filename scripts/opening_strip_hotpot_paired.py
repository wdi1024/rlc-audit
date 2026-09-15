#!/usr/bin/env python3
"""Paired change intervals for the HotpotQA opening intervention (round 20, item 3).

opening_strip_hotpot.py reported each arm's AUC before and after the intervention with a
bootstrap interval on the stripped value only. Three things the review asks for:
  (a) the 90% interval on the stripped construct arm, the level the equivalence test uses,
  (b) paired change intervals (stripped - intact) per arm, resampling prompts,
  (c) the same for the matched random-deletion control (mean over its 20 seeds).
Recorded point values are reproduced first as a referee check.
"""
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent.parent; DATA=ROOT/"results"/"disagree_routing"; JUDGE="anthropic_claude-haiku-4-5-20251001"
PH="phase14_hotpot_2k"; SPAN=50; N_SEEDS=20; N_BOOT=2000
QWEN=[r"^thinking process:\s*\n+\s*1\.\s*\*\*analyze the request:\*\*\s*", r"^the user wants to (know|identify|find|determine|compare|answer|confirm|verify|understand|figure out)\s+"]
GEMMA=[r"^based on the (provided )?passages( provided)?,?\s*(there is \*{0,2}no information\*{0,2}\s*)?", r"^the provided passages\s+(do not|does not)?\s*", r"^there is (\*\*)?no information(\*\*)? in the provided passages\s*"]
def strip(t,pats):
    for p in pats:
        m=re.match(p,t,flags=re.I)
        if m: return t[m.end():]
    return t
def excise_random(text,n,rng):
    if n<=0 or len(text)<=n: return text
    st=int(rng.integers(1,len(text)-n+1)); return text[:st]+text[st+n:]
def cos(a,b):
    v=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2).fit(a+b); xa,xb=v.transform(a),v.transform(b)
    num=np.asarray(xa.multiply(xb).sum(1)).ravel(); na=np.sqrt(np.asarray(xa.multiply(xa).sum(1)).ravel()); nb=np.sqrt(np.asarray(xb.multiply(xb).sum(1)).ravel())
    return 1-num/np.maximum(na*nb,1e-9)
ta={r["id"]:r for r in json.loads((DATA/f"{PH}_traces_qwen3.5-2b.json").read_text())["records"]}
tb={r["id"]:r for r in json.loads((DATA/f"{PH}_traces_gemma-4-e2b.json").read_text())["records"]}
ja={r["id"]:r for r in json.loads((DATA/f"{PH}_correctness_qwen3.5-2b_{JUDGE}.json").read_text())["records"]}
jb={r["id"]:r for r in json.loads((DATA/f"{PH}_correctness_gemma-4-e2b_{JUDGE}.json").read_text())["records"]}
ids=sorted(set(ta)&set(tb)&set(ja)&set(jb)); a=[ta[i].get("trace") or "" for i in ids]; b=[tb[i].get("trace") or "" for i in ids]
gold={i:(ja[i].get("gold_answer") or "").strip().lower() for i in ids}
y=np.array([int(bool(ja[i].get("correct_judge"))!=bool(jb[i].get("correct_judge"))) for i in ids])
z=np.array([int((gold[i] in x[:SPAN].lower())!=(gold[i] in w[:SPAN].lower())) for i,x,w in zip(ids,a,b)])
zc=np.array([int((gold[i] in x[SPAN:].lower())!=(gold[i] in w[SPAN:].lower())) for i,x,w in zip(ids,a,b)])
s0=cos([t[:SPAN] for t in a],[t[:SPAN] for t in b])
a_s=[strip(x,QWEN) for x in a]; b_s=[strip(w,GEMMA) for w in b]
rem_a=[len(o)-len(t) for o,t in zip(a,a_s)]; rem_b=[len(o)-len(t) for o,t in zip(b,b_s)]
s1=cos([t[:SPAN] for t in a_s],[t[:SPAN] for t in b_s])
matched=[]
for seed in range(N_SEEDS):
    rng=np.random.default_rng(1000+seed)
    matched.append(cos([excise_random(x,n,rng)[:SPAN] for x,n in zip(a,rem_a)],[excise_random(w,n,rng)[:SPAN] for w,n in zip(b,rem_b)]))
rec=json.load(open(ROOT/"analysis_results"/"opening_strip_hotpot.json"))["targets"]
n=len(ids); rng=np.random.default_rng(7)
out={"n":n,"n_boot":N_BOOT,"arms":{}}
for name,lab in (("z (same span)",z),("z_comp (off-span only)",zc),("y (construct)",y)):
    i0=float(roc_auc_score(lab,s0)); i1=float(roc_auc_score(lab,s1)); mm=[float(roc_auc_score(lab,m)) for m in matched]
    assert abs(i0-rec[name]["intact"])<2e-3 and abs(i1-rec[name]["targeted"])<2e-3, (name,i0,i1)
    dS=[];dM=[];S=[]
    for _ in range(N_BOOT):
        sel=rng.integers(0,n,n)
        if len(set(lab[sel]))<2: continue
        aS=roc_auc_score(lab[sel],s1[sel]); a0=roc_auc_score(lab[sel],s0[sel]); aM=np.mean([roc_auc_score(lab[sel],m[sel]) for m in matched[:5]])
        S.append(aS); dS.append(aS-a0); dM.append(aM-a0)
    q=lambda v,lo,hi: [float(np.percentile(v,lo)),float(np.percentile(v,hi))]
    out["arms"][name]={"intact":i0,"stripped":i1,"matched_mean":float(np.mean(mm)),
        "stripped_ci95":q(S,2.5,97.5),"stripped_ci90":q(S,5,95),
        "change_stripped":i1-i0,"change_ci95":q(dS,2.5,97.5),
        "change_matched":float(np.mean(mm))-i0,"change_matched_ci95":q(dM,2.5,97.5)}
    o=out["arms"][name]
    print(f"{name:24s} intact {i0:.3f} -> stripped {i1:.3f}  [95% {o['stripped_ci95'][0]:.3f},{o['stripped_ci95'][1]:.3f}] [90% {o['stripped_ci90'][0]:.3f},{o['stripped_ci90'][1]:.3f}]")
    print(f"{'':24s}   change {i1-i0:+.3f} [{o['change_ci95'][0]:+.3f},{o['change_ci95'][1]:+.3f}]   matched change {o['change_matched']:+.3f} [{o['change_matched_ci95'][0]:+.3f},{o['change_matched_ci95'][1]:+.3f}]",flush=True)
# gap between off-span proxy and construct, before and after
g0=abs(float(roc_auc_score(zc,s0))-0.5)-abs(float(roc_auc_score(y,s0))-0.5)
g1=abs(float(roc_auc_score(zc,s1))-0.5)-abs(float(roc_auc_score(y,s1))-0.5)
out["delta_dis_intact"]=g0; out["delta_dis_stripped"]=g1
print(f"\nDelta_dis: intact {g0:+.3f} -> stripped {g1:+.3f}")
print(f"off-span drop {out['arms']['z_comp (off-span only)']['intact']-out['arms']['z_comp (off-span only)']['stripped']:.4f}; "
      f"as share of AUC {(out['arms']['z_comp (off-span only)']['intact']-out['arms']['z_comp (off-span only)']['stripped'])/out['arms']['z_comp (off-span only)']['intact']:.3f}; "
      f"of above-chance {(out['arms']['z_comp (off-span only)']['intact']-out['arms']['z_comp (off-span only)']['stripped'])/(out['arms']['z_comp (off-span only)']['intact']-0.5):.3f}")
(ROOT/"analysis_results"/"opening_strip_hotpot_paired.json").write_text(json.dumps(out,indent=1)); print("wrote")
