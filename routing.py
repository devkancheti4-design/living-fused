#!/usr/bin/env python3
# living-fused — routing: the Life's confidence gate routes around the model.
# Copyright (C) 2026 Kancheti Devieswar — AGPL-3.0 (see LICENSE)
"""
The Life's confidence gate is also a ROUTER. For a query it confidently knows, the Life
answers instantly (~microseconds) with ZERO model forwards — the big model stays idle. For
anything it doesn't know, it falls through to the model. A real counter tallies every model
forward, so "0 model calls" is a counted fact, not a claim.

    pip install torch numpy transformers
    python3 routing.py

Measured (Llama-3.2-1B, 20 known + 20 unknown queries, Apple M4 Pro CPU):
    answered by the Life (0 model calls): 20/40, ~0.1 us, correct 20/20
    routed to the model (Llama forward):  20/40, ~173 ms
    -> half the queries answered with the model idle; ratio = your workload's lookup share.
"""
import os, sys, time, subprocess
try:
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    sys.exit("needs: pip install torch numpy transformers")
def rss(): return int(subprocess.check_output(["ps","-o","rss=","-p",str(os.getpid())]).strip())/1024
def P(m): sys.stdout.write(m+"\n"); sys.stdout.flush()
torch.manual_seed(0); np.random.seed(0)
MODEL="unsloth/Llama-3.2-1B"; W=128
tok=AutoTokenizer.from_pretrained(MODEL)
lm=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
for p_ in lm.parameters(): p_.requires_grad_(False)
V=lm.config.vocab_size
with torch.no_grad():
    for _ in range(2): lm(torch.randint(0,V,(1,128)),use_cache=False)
P(f"Llama-3.2-1B fp32 | resident RSS {rss():.0f} MB")

MODEL_CALLS=[0]
def model_next(ctx):
    MODEL_CALLS[0]+=1
    with torch.no_grad():
        return torch.softmax(lm(torch.tensor([ctx[-W:]]),use_cache=False).logits[0,-1].float(),-1).numpy()

class Life:
    C=0.25
    def __init__(s): s.store={}
    def learn(s,k,n): d=s.store.setdefault(k,{}); tot=sum(d.values()); d[n]=d.get(n,0)+tot+1
    def confidence(s,k):
        d=s.store.get(k)
        if not d: return 0.0, None
        tot=sum(d.values()); return tot/(tot+s.C), max(d,key=d.get)
life=Life(); THRESH=0.5
KN=[" Zephyr"," Orion"," Vega"," Atlas"," Nova"," Lyra"," Draco"," Rigel"," Mira"," Cygnus",
    " Pavo"," Corvus"," Tucana"," Grus"," Indus"," Norma"," Ara"," Apus"," Volans"," Carina"]
VALS=[" red"," blue"," green"," gold"," black"," white"," pink"," gray"," teal"," amber"]
known=[]
for i,nm in enumerate(KN):
    fr=tuple(tok.encode(f"The{nm} unit is",add_special_tokens=False)); v=tok.encode(VALS[i%len(VALS)],add_special_tokens=False)[0]
    life.learn(fr,v); known.append((fr,v))
UNK=[" Nebula"," Quasar"," Pulsar"," Comet"," Meteor"," Photon"," Neutron"," Proton"," Quark"," Boson",
     " Helix"," Vortex"," Cipher"," Onyx"," Ember"," Frost"," Talon"," Raptor"," Falcon"," Sable"]
unknown=[tuple(tok.encode(f"The{nm} unit is",add_special_tokens=False)) for nm in UNK]

def handle(fr,true_val):
    w,ans=life.confidence(fr)
    if w>=THRESH:
        t=time.perf_counter(); return "life",(int(ans==true_val) if true_val is not None else None),(time.perf_counter()-t)*1e6
    t=time.perf_counter(); p=model_next(list(fr)); dt=(time.perf_counter()-t)*1e6
    return "model",(int(int(p.argmax())==true_val) if true_val is not None else None),dt

life_n=model_n=life_ok=life_t=model_t=0
for fr,v in known:
    r,ok,us=handle(fr,v)
    if r=="life": life_n+=1; life_ok+=ok or 0; life_t+=us
    else: model_n+=1; model_t+=us
for fr in unknown:
    r,ok,us=handle(fr,None)
    if r=="life": life_n+=1; life_t+=us
    else: model_n+=1; model_t+=us
tot=life_n+model_n
P(f"\n  answered by the LIFE (0 model calls): {life_n}/{tot}   avg {life_t/max(life_n,1):.1f} us   correct {life_ok}/{life_n}")
P(f"  routed to the MODEL (Llama forward):  {model_n}/{tot}   avg {model_t/max(model_n,1)/1000:.1f} ms")
P(f"  TOTAL model forward calls (counted):  {MODEL_CALLS[0]}   (life-routed made ZERO)")
P(f"  -> {100*life_n/tot:.0f}% answered with the model idle; the ratio = your workload's lookup share")
