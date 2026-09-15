#!/usr/bin/env python3
"""Targeted test of the question-paraphrase explanation (round 20, item 2).

v52 attributed the off-span association that survives the opener strip to the opening being a
paraphrase of the question. v53 withdrew that attribution as unidentified. This isolates it.

Intervention: inside each side's trace, replace every content word that also occurs in the
question with a same-length filler ("xxxx"), so character offsets, and therefore the 50-character
scoring boundary, are unchanged; only the question-overlapping lexical content is destroyed.
Gold-answer tokens are never masked, so the span-measurable proxy z keeps its meaning; the
off-span proxy z^c and the construct y read text beyond the span and are untouched by construction.

Control: mask the same number of content words per side, drawn at random from words that do NOT
occur in the question, same-length fillers, twenty seeds. If question paraphrase carries the
residual off-span association, the targeted mask should lower AUC(s,z^c) well beyond the control.

Recorded intact AUCs are reproduced first as a referee check.
"""
import json, re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent.parent; DATA=ROOT/"results"/"disagree_routing"
JUDGE="anthropic_claude-haiku-4-5-20251001"; PH="phase14_hotpot_2k"; SPAN=50; N_SEEDS=20; N_BOOT=2000
STOP=set("the a an of to in and or is are was were be been on for with as at by from that this it its "
         "he she they what which who when where how why did does do we you i not no than then there here "
         "about into over under after before between during his her their our your".split())
def cos(a,b):
    v=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2).fit(a+b); xa,xb=v.transform(a),v.transform(b)
    num=np.asarray(xa.multiply(xb).sum(1)).ravel(); na=np.sqrt(np.asarray(xa.multiply(xa).sum(1)).ravel()); nb=np.sqrt(np.asarray(xb.multiply(xb).sum(1)).ravel())
    return 1-num/np.maximum(na*nb,1e-9)
def content(w):
    c=re.sub(r"\W","",w).lower(); return c if len(c)>2 and c not in STOP and c.isalpha() else None
def mask(text, targets, gold, limit=400):
    """Replace occurrences of `targets` (lowercase words) in the first `limit` chars with same-length fillers."""
    out=[]; pos=0
    for m in re.finditer(r"\S+", text[:limit]):
        w=m.group(0); c=content(w)
        if c and c in targets and c not in gold:
            out.append((m.start(), m.end(), "x"*len(w)))
    if not out: return text, 0
    s=list(text)
    for a,b,rep in out: s[a:b]=list(rep)
    return "".join(s), len(out)
ta={r["id"]:r for r in json.loads((DATA/f"{PH}_traces_qwen3.5-2b.json").read_text())["records"]}
tb={r["id"]:r for r in json.loads((DATA/f"{PH}_traces_gemma-4-e2b.json").read_text())["records"]}
ja={r["id"]:r for r in json.loads((DATA/f"{PH}_correctness_qwen3.5-2b_{JUDGE}.json").read_text())["records"]}
jb={r["id"]:r for r in json.loads((DATA/f"{PH}_correctness_gemma-4-e2b_{JUDGE}.json").read_text())["records"]}
ids=sorted(set(ta)&set(tb)&set(ja)&set(jb))
a=[ta[i].get("trace") or "" for i in ids]; b=[tb[i].get("trace") or "" for i in ids]
q=[(ta[i].get("prompt") or ta[i].get("question") or "") for i in ids]
gold={i:(ja[i].get("gold_answer") or "").strip().lower() for i in ids}
y=np.array([int(bool(ja[i].get("correct_judge"))!=bool(jb[i].get("correct_judge"))) for i in ids])
z=np.array([int((gold[i] in x[:SPAN].lower())!=(gold[i] in w[:SPAN].lower())) for i,x,w in zip(ids,a,b)])
zc=np.array([int((gold[i] in x[SPAN:].lower())!=(gold[i] in w[SPAN:].lower())) for i,x,w in zip(ids,a,b)])
s0=cos([t[:SPAN] for t in a],[t[:SPAN] for t in b])
rec=json.load(open(ROOT/"analysis_results"/"opening_strip_hotpot.json"))["targets"]
for nm,lab in (("z (same span)",z),("z_comp (off-span only)",zc),("y (construct)",y)):
    assert abs(roc_auc_score(lab,s0)-rec[nm]["intact"])<2e-3, nm
print(f"n={len(ids)}  intact z {roc_auc_score(z,s0):.3f}  zc {roc_auc_score(zc,s0):.3f}  y {roc_auc_score(y,s0):.3f}",flush=True)
# targeted mask
aT=[];bT=[];cnt_a=[];cnt_b=[]
for i,qq,x,w in zip(ids,q,a,b):
    tg={content(t) for t in qq.split()}-{None}
    g=set(gold[i].split())
    xa,na_=mask(x,tg,g); xb,nb_=mask(w,tg,g)
    aT.append(xa); bT.append(xb); cnt_a.append(na_); cnt_b.append(nb_)
print(f"targeted masked words per side: qwen mean {np.mean(cnt_a):.2f}, gemma mean {np.mean(cnt_b):.2f}; "
      f"sides with >=1 mask: {sum(1 for c in cnt_a if c)}/{len(ids)}, {sum(1 for c in cnt_b if c)}/{len(ids)}",flush=True)
s1=cos([t[:SPAN] for t in aT],[t[:SPAN] for t in bT])
# matched control: same count, random NON-question content words
ctrl=[]
for seed in range(N_SEEDS):
    rng=np.random.default_rng(500+seed); aC=[];bC=[]
    for i,qq,x,w,ca,cb in zip(ids,q,a,b,cnt_a,cnt_b):
        tg={content(t) for t in qq.split()}-{None}; g=set(gold[i].split())
        for txt,k,acc in ((x,ca,aC),(w,cb,bC)):
            pool=list({content(t) for t in txt[:400].split()}-{None}-tg-g)
            pick=set(rng.choice(pool,size=min(k,len(pool)),replace=False)) if k and pool else set()
            acc.append(mask(txt,pick,g)[0])
    ctrl.append(cos([t[:SPAN] for t in aC],[t[:SPAN] for t in bC]))
rng=np.random.default_rng(11); n=len(ids); out={"n":n,"n_boot":N_BOOT,"masked_per_side":[float(np.mean(cnt_a)),float(np.mean(cnt_b))],"arms":{}}
for nm,lab in (("z (same span)",z),("z_comp (off-span only)",zc),("y (construct)",y)):
    i0=float(roc_auc_score(lab,s0)); i1=float(roc_auc_score(lab,s1)); mc=[float(roc_auc_score(lab,c)) for c in ctrl]
    dT=[];dC=[]
    for _ in range(N_BOOT):
        sel=rng.integers(0,n,n)
        if len(set(lab[sel]))<2: continue
        a0=roc_auc_score(lab[sel],s0[sel])
        dT.append(roc_auc_score(lab[sel],s1[sel])-a0)
        dC.append(np.mean([roc_auc_score(lab[sel],c[sel]) for c in ctrl[:5]])-a0)
    Q=lambda v: [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
    out["arms"][nm]={"intact":i0,"targeted":i1,"targeted_change":i1-i0,"targeted_change_ci":Q(dT),
                     "control_mean":float(np.mean(mc)),"control_change":float(np.mean(mc))-i0,"control_change_ci":Q(dC)}
    o=out["arms"][nm]
    print(f"{nm:24s} intact {i0:.3f} -> question-masked {i1:.3f} (change {o['targeted_change']:+.3f} {[round(v,3) for v in o['targeted_change_ci']]}) | "
          f"matched control {o['control_mean']:.3f} (change {o['control_change']:+.3f} {[round(v,3) for v in o['control_change_ci']]})",flush=True)
(ROOT/"analysis_results"/"question_paraphrase_intervention.json").write_text(json.dumps(out,indent=1)); print("wrote")
