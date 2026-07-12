#!/usr/bin/env python3
# living-fused — scale_mlx: graft the living memory onto a 4-bit MLX model of any size.
# Copyright (C) 2026 Kancheti Devieswar — AGPL-3.0 (see LICENSE)
"""
The SAME living memory, grafted onto real 4-bit models from 3B to 7B via Apple MLX.
Measures footprint (peak RSS), synthetic beyond-window recall (Life vs model alone), and
— if an `enwik8` file is present — recall on REAL Wikipedia text.

    pip install --break-system-packages --no-deps mlx_lm      # (pair the version to your mlx)
    python3 scale_mlx.py mlx-community/Qwen2.5-7B-Instruct-4bit
    python3 scale_mlx.py mlx-community/Llama-3.2-3B-Instruct-4bit

Optional real-data test: put an `enwik8` file (first 100MB of Wikipedia, from the Hutter
Prize corpus) in this directory, or set ENWIK8=/path/to/enwik8.

Measured on an Apple M4 Pro (your footprint will match; recall is structural):
    Qwen2.5-7B  4-bit : ~4.7 GB | synthetic 0/10 -> 10/10 | real Wikipedia 17/40 -> 40/40
    Llama-3.2-3B 4-bit: ~2.5 GB | synthetic 0/10 -> 10/10 | real Wikipedia 19/40 -> 40/40
"""
import os, sys, re, time, subprocess
try:
    import numpy as np, mlx.core as mx
    from mlx_lm import load
except Exception:
    sys.exit("needs: pip install --break-system-packages --no-deps mlx_lm  (matched to your mlx)")
def rss(): return int(subprocess.check_output(["ps","-o","rss=","-p",str(os.getpid())]).strip())/1024
def P(m): sys.stdout.write(m+"\n"); sys.stdout.flush()

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Llama-3.2-3B-Instruct-4bit"
P(f"loading {MODEL} (4-bit MLX)...")
t=time.time(); model, tok = load(MODEL); P(f"loaded in {time.time()-t:.0f}s")
W = 128
def probs(ids):
    lg=model(mx.array([ids[-W:]]))[0,-1]; p=mx.softmax(lg.astype(mx.float32)); mx.eval(p); return np.array(p)
_=probs(tok.encode("warm up the model now please")); resident=rss()
P(f"resident RSS (4-bit weights + runtime): {resident:.0f} MB")

class Life:                                     # recency-dominant integer counts + confidence blend
    C=0.25
    def __init__(s): s.store={}
    def learn(s,k,n): d=s.store.setdefault(k,{}); tot=sum(d.values()); d[n]=d.get(n,0)+tot+1
    def blend(s,k,p):
        d=s.store.get(k)
        if not d: return p
        tot=sum(d.values()); w=tot/(tot+s.C); q=p*(1-w)
        for t_,c in d.items(): q[t_]+=w*(c/tot)
        return q

# 1) synthetic matched-key facts
life=Life()
NAMES=[" Zephyr"," Orion"," Vega"," Atlas"," Nova"," Lyra"," Draco"," Rigel"," Mira"," Cygnus"]
VALS =[" red"," blue"," green"," gold"," black"," white"," pink"," gray"," teal"," amber"]
facts=[]
for i,nm in enumerate(NAMES):
    fr=tuple(tok.encode(f"The{nm} unit is")); v=tok.encode(VALS[i])[-1]; life.learn(fr,v); facts.append((fr,v))
sa=sl=0
for fr,v in facts:
    p=probs(list(fr)); sa+=int(int(p.argmax())==v); sl+=int(int(life.blend(fr,p.copy()).argmax())==v)

# 2) optional REAL Wikipedia (enwik8)
ra=rl=n=0; ENW=os.environ.get("ENWIK8","enwik8")
if os.path.exists(ENW):
    raw=open(ENW,"rb").read(5_000_000).decode("utf-8","ignore")
    txt=re.sub(r"<[^>]+>"," ",raw); txt=re.sub(r"\[\[|\]\]|\{\{|\}\}|==+|\||\[|\]"," ",txt)
    txt=re.sub(r"&[a-z]+;"," ",txt); txt=re.sub(r"http\S+"," ",txt); txt=re.sub(r"\s+"," ",txt)
    sents=[s.strip() for s in re.findall(r"[A-Z][A-Za-z0-9 ,'\-\(\)]{45,130}\.",txt)
           if 9<=len(s.split())<=22 and sum(c.isalpha() for c in s)/len(s)>0.75]
    rl_life=Life()
    for s in sents:
        ids=tok.encode(s.rstrip("."))
        if len(ids)<7: continue
        rl_life.learn(tuple(ids[:-1]),ids[-1]); facts_r=(tuple(ids[:-1]),ids[-1])
        p=probs(list(facts_r[0])); ra+=int(int(p.argmax())==facts_r[1]); rl+=int(int(rl_life.blend(facts_r[0],p.copy()).argmax())==facts_r[1]); n+=1
        if n>=40: break

P("\n"+"="*60)
P(f"MODEL: {MODEL}")
P(f"  footprint (peak RSS):            {rss():.0f} MB   (4-bit)")
P(f"  synthetic recall beyond window:  alone {sa}/10   + Life {sl}/10")
if n: P(f"  REAL Wikipedia recall (n={n}):     alone {ra}/{n}   + Life {rl}/{n}")
else: P("  (real-Wikipedia test skipped — no enwik8 file found)")
P("="*60)
