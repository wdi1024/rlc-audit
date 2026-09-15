#!/usr/bin/env python3
"""Instruction-level cluster bootstrap for the Hybrid LLM / MixInstruct audit (round 19, item 4).

The recorded intervals resample comparisons i.i.d.; comparisons within one
instruction share candidates and are not independent. This script reproduces the
pooled point estimates as a referee check and recomputes the intervals by
resampling instructions (5,000 test rows) with replacement, recomputing kappa,
overall disagreement, AUC(gap,y), the length AUCs and the top-10% queue
disagreement on every resample.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score, roc_auc_score
ROOT=Path(__file__).resolve().parent.parent; N_BOOT=500; MIN_PAIR_N=200
ds=load_dataset("llm-blender/mix-instruct", split="test")
per_pair=defaultdict(list)
for ri,row in enumerate(ds):
    cand={c["model"]:c for c in row["candidates"]}
    bs={m:c["scores"]["bartscore"] for m,c in cand.items() if c["scores"].get("bartscore") is not None}
    try: cmp=json.loads(row["cmp_results"]) if row["cmp_results"] else {}
    except (json.JSONDecodeError,TypeError): continue
    if not isinstance(cmp,dict): continue
    for key,verdict in cmp.items():
        if "," not in key: continue
        a,b=key.split(",",1)
        if a not in bs or b not in bs: continue
        if verdict=="A is better": y=1
        elif verdict=="B is better": y=0
        else: continue
        per_pair[(a,b)].append((bs[a]-bs[b],y,len(cand[a]["text"] or "")-len(cand[b]["text"] or ""),ri))
gap,y,ln,inst=[],[],[],[]
for (a,b),items in per_pair.items():
    if len(items)<MIN_PAIR_N: continue
    yy=[v for _,v,_,_ in items]; zz=[int(g>0) for g,_,_,_ in items]
    if len(set(yy))<2 or len(set(zz))<2: continue
    for g,v,d,ri in items: gap.append(g); y.append(v); ln.append(d); inst.append(ri)
gap=np.array(gap); y=np.array(y); ln=np.array(ln,float); inst=np.array(inst); z=(gap>0).astype(int); n=len(y)
def stats(idx):
    g=gap[idx]; yy=y[idx]; zz=z[idx]; l=ln[idx]
    B=max(1,int(round(0.10*len(idx)))); order=np.argsort(-np.abs(g),kind="stable")[:B]
    return dict(kappa=cohen_kappa_score(zz,yy), disagreement=float((zz!=yy).mean()), auc=roc_auc_score(yy,g),
                auc_len_z=roc_auc_score(zz,l), auc_len_y=roc_auc_score(yy,l), queue10=float((zz[order]!=yy[order]).mean()))
pt=stats(np.arange(n)); print("n",n,{k:round(v,4) for k,v in pt.items()},flush=True)
assert n==253519 and abs(pt['kappa']-0.4263)<5e-4 and abs(pt['disagreement']-0.2868)<5e-4 and abs(pt['queue10']-0.1379)<5e-4 and abs(pt['auc_len_z']-0.6090)<5e-4
by=defaultdict(list)
for i,r in enumerate(inst): by[r].append(i)
clusters=[np.array(v) for v in by.values()]; C=len(clusters); print("instructions (clusters):",C,flush=True)
rng=np.random.default_rng(20260907); acc=defaultdict(list)
for b in range(N_BOOT):
    pick=rng.integers(0,C,C); idx=np.concatenate([clusters[c] for c in pick]); st=stats(idx)
    for k,v in st.items(): acc[k].append(v)
    acc['len_asym'].append(st['auc_len_z']-st['auc_len_y'])
    if b%100==0: print(b,flush=True)
out={"n":int(n),"clusters":int(C),"n_boot":N_BOOT,"point":pt,"cluster_ci":{k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in acc.items()}}
print(json.dumps(out,indent=1)); (ROOT/"analysis_results"/"hybridllm_cluster_bootstrap.json").write_text(json.dumps(out,indent=1)); print("wrote")
