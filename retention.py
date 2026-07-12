#!/usr/bin/env python3
# living-fused — retention: the Life retains tens of thousands of facts with ZERO forgetting.
# Copyright (C) 2026 Kancheti Devieswar — AGPL-3.0 (see LICENSE)
"""
Store N facts in the Life and show it retains ALL of them (100%) as N grows to tens of
thousands. Each fact lives in its own row, so a new fact physically cannot disturb an old
one (zero forgetting by construction) — the opposite of a fine-tune, which forgets as you
cram more in. Memory grows ~linearly (~0.5 KB/fact).

    python3 retention.py            # uses REAL Wikipedia if an `enwik8` file is present,
                                    # else a synthetic fact set of the same size.

Measured on real Wikipedia (enwik8), Apple M4 Pro:
    N=1,000 -> 100% (0 forgotten, 0.5 MB)   N=30,000 -> 100% (0 forgotten, 14.7 MB)
"""
import os, re, sys
try:
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    def enc(s): return tok.encode(s, add_special_tokens=False)
except Exception:
    enc=lambda s: [ord(c) for c in s]            # fallback: byte-ish tokens (still valid keys)
def getsize(store):
    t=sys.getsizeof(store)
    for k,v in store.items(): t+=sys.getsizeof(k)+sys.getsizeof(v)+sum(sys.getsizeof(x) for x in v)
    return t
def P(m): sys.stdout.write(m+"\n"); sys.stdout.flush()

facts={}; ENW=os.environ.get("ENWIK8","enwik8")
if os.path.exists(ENW):
    raw=open(ENW,"rb").read(40_000_000).decode("utf-8","ignore")
    txt=re.sub(r"<[^>]+>"," ",raw); txt=re.sub(r"\[\[|\]\]|\{\{|\}\}|==+|\||\[|\]"," ",txt)
    txt=re.sub(r"&[a-z]+;"," ",txt); txt=re.sub(r"http\S+"," ",txt); txt=re.sub(r"\s+"," ",txt)
    sents=[s.strip() for s in re.findall(r"[A-Z][A-Za-z0-9 ,'\-\(\)]{45,130}\.",txt)
           if 9<=len(s.split())<=22 and sum(c.isalpha() for c in s)/len(s)>0.75]
    src="REAL Wikipedia (enwik8)"
    for s in sents:
        ids=enc(s.rstrip("."))
        if len(ids)>=7: facts[tuple(ids[:-1])]=ids[-1]
        if len(facts)>=30000: break
else:
    src="synthetic (no enwik8 found)"
    for i in range(30000): facts[tuple(enc(f"fact number {i} in the record about topic"))]=i%50000
facts=list(facts.items())
P(f"{len(facts):,} DISTINCT facts from {src}")
P(f"\n  {'N facts':>10} {'retained':>18} {'forgotten':>10} {'Life memory':>13}")

class Life:
    def __init__(s): s.store={}
    def learn(s,k,n): d=s.store.setdefault(k,{}); tot=sum(d.values()); d[n]=d.get(n,0)+tot+1

for N in [100,1000,5000,10000,20000,30000]:
    if N>len(facts): break
    life=Life()
    for k,v in facts[:N]: life.learn(k,v)
    ret=sum(1 for k,v in facts[:N] if max(life.store[k],key=life.store[k].get)==v)
    P(f"  {N:>10,} {ret:>8,}/{N:<8,}({100*ret/N:.0f}%) {N-ret:>9,} {getsize(life.store)/1e6:>10.1f} MB")
P("\n  retention stays 100% as N grows — disjoint rows mean a new fact CANNOT disturb an old")
P("  one (zero forgetting by construction). Memory grows ~linearly (~0.5 KB/fact).")
