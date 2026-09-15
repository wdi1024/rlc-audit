#!/usr/bin/env python3
"""The opening-strip dissection on the HotpotQA 50-character contract (round 19, item 5).

The one SCORE FAILURE is the 50-character HotpotQA span, but the fixed-label
opening intervention had only been run on refusal contracts. This mirrors
opening_strip_orbench.py on phase14_hotpot_2k: suppress each side's fixed
opening template (Qwen: the 'Thinking Process / Analyze the Request' header and
the 'The user wants to <verb>' opener; Gemma: the 'Based on the (provided)
passages' / 'The provided passages' / 'There is no information in the provided
passages' openers), hold z, z^c and y fixed, re-score the prefix-50 distance with
the correctness contract's own TF-IDF (char_wb 3-5, min_df=2), and compare against
deleting the same number of characters at a random offset over twenty seeds. The
intact AUCs must reproduce the recorded row before anything is written.
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
rec=json.load(open(ROOT/"analysis_results"/"complement_span_proxy_control.json"))["rows"]; row=next(r for r in rec if r["contract"]=="HotpotQA, 50-char span")
i_y,i_z,i_zc=roc_auc_score(y,s0),roc_auc_score(z,s0),roc_auc_score(zc,s0)
print(f"n={len(ids)} intact AUC y={i_y:.4f} z={i_z:.4f} zc={i_zc:.4f} | recorded y={row['auc_y']:.4f} z={row['auc_z']:.4f} zc={row['comp']['auc']:.4f}",flush=True)
assert abs(i_y-row['auc_y'])<2e-3 and abs(i_z-row['auc_z'])<2e-3 and abs(i_zc-row['comp']['auc'])<2e-3, "recorded row not reproduced"
a_s=[strip(x,QWEN) for x in a]; b_s=[strip(w,GEMMA) for w in b]
rem_a=[len(o)-len(t) for o,t in zip(a,a_s)]; rem_b=[len(o)-len(t) for o,t in zip(b,b_s)]
print("sides altered:",sum(1 for r in rem_a if r>0),"qwen,",sum(1 for r in rem_b if r>0),"gemma; chars removed",sum(rem_a)+sum(rem_b),"of",sum(len(t) for t in a+b),flush=True)
s1=cos([t[:SPAN] for t in a_s],[t[:SPAN] for t in b_s])
matched=[]
for seed in range(N_SEEDS):
    rng=np.random.default_rng(1000+seed)
    matched.append(cos([excise_random(x,n,rng)[:SPAN] for x,n in zip(a,rem_a)],[excise_random(w,n,rng)[:SPAN] for w,n in zip(b,rem_b)]))
def boot(lab,s,seed):
    rng=np.random.default_rng(seed); out=[]
    for _ in range(N_BOOT):
        sel=rng.integers(0,len(s),len(s))
        if len(set(lab[sel]))<2: continue
        out.append(roc_auc_score(lab[sel],s[sel]))
    return [float(np.percentile(out,2.5)),float(np.percentile(out,97.5))]
res={}
for name,lab,seed in (("z (same span)",z,1),("z_comp (off-span only)",zc,2),("y (construct)",y,3)):
    ai=float(roc_auc_score(lab,s0)); at=float(roc_auc_score(lab,s1)); ms=[float(roc_auc_score(lab,m)) for m in matched]
    res[name]={"positives":int(lab.sum()),"intact":ai,"targeted":at,"targeted_ci":boot(lab,s1,seed),"targeted_drop":ai-at,"matched_mean":float(np.mean(ms)),"matched_sd":float(np.std(ms)),"matched_drop":ai-float(np.mean(ms))}
    print(f"  {name:24s} pos {lab.sum():4d} intact {ai:.3f} targeted {at:.3f} {res[name]['targeted_ci']} drop {ai-at:+.3f} | matched {np.mean(ms):.3f} (sd {np.std(ms):.3f}) drop {ai-np.mean(ms):+.3f}",flush=True)
out={"phase":PH,"n":len(ids),"span":SPAN,"n_seeds":N_SEEDS,"patterns":{"qwen":QWEN,"gemma":GEMMA},"sides_altered":{"qwen":sum(1 for r in rem_a if r>0),"gemma":sum(1 for r in rem_b if r>0)},"chars_removed":int(sum(rem_a)+sum(rem_b)),"chars_total":int(sum(len(t) for t in a+b)),"targets":res}
(ROOT/"analysis_results"/"opening_strip_hotpot.json").write_text(json.dumps(out,indent=1)); print("wrote")
