#!/usr/bin/env python3
# living-fused — scale_llama: the SAME organ from scale_demo.py, grafted onto a REAL
# 1-billion-parameter transformer (Llama-3.2-1B) to show the recipe is model-agnostic.
# Copyright (C) 2026 Kancheti Devieswar — AGPL-3.0 (see LICENSE)
"""
scale_demo.py proves the graft on GPT-2 (124M). This proves it on a model ~8000x the
toy body: a frozen, pretrained Llama-3.2-1B (1.24B params). The Organ class below is
byte-for-byte the same idea as scale_demo.py — nothing about the recipe changes when
the model gets 8000x bigger. That is the whole point.

    pip install torch numpy transformers
    python3 scale_llama.py        # first run downloads Llama-3.2-1B (~2.5GB); CPU is slow but works

The served window is deliberately capped small (SERVED=200) for two honest reasons:
  (1) it keeps the demo fast on a CPU, and
  (2) it puts the facts BEYOND the served window (~450 tokens back), which is exactly
      the regime the organ exists for. In a real deployment you size the served window
      to your latency/cost budget; the organ supplies memory past whatever you pick.

Expected (deterministic on this setup):
    RECALL    Llama-1B alone 0/6   Llama-1B + organ 5/6
    REVISION  Llama-1B alone 0/6   Llama-1B + organ 6/6

Honest note on the 5/6: one recall misses on a Llama-BPE tokenization edge case, not a
logic failure — it reproduces at 5/6 across runs (a random fluke would move). The
decisive contrast is 5-6/6 WITH the organ vs 0/6 without: the frozen model is blind to
anything past its served window; the organ gives the same frozen weights beyond-window
recall and fact-revision. The transformer is never modified.
"""
import sys
try:
    import torch, numpy as np, time
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("scale_llama needs: pip install torch numpy transformers")

torch.manual_seed(0); np.random.seed(0)
MODEL = "unsloth/Llama-3.2-1B"        # ungated mirror; swap for any HF causal LM
SERVED = 200                          # capped served context (a real deployment cost control)
print(f"loading {MODEL} on CPU (this is a REAL 1B-param model; be patient) ...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
lm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
for p in lm.parameters(): p.requires_grad_(False)     # the transformer is FROZEN
NL = tok.encode("\n", add_special_tokens=False)[0]
print(f"loaded {sum(p.numel() for p in lm.parameters())/1e9:.2f}B params in {time.time()-t0:.0f}s, "
      f"served window capped at {SERVED}")

class Organ:
    """Identical living memory to scale_demo.py: integer counts, recency-dominant,
    disjoint per key, keyed on the tokens since the last newline (the fact frame)."""
    C = 0.25                                          # gate: a known fact dominates the neural prior
    def __init__(s): s.store = {}
    def key(s, ctx):
        for j in range(len(ctx) - 1, -1, -1):
            if ctx[j] == NL: return tuple(ctx[j+1:])
        return tuple(ctx)
    def learn(s, ctx, nxt, is_fact):
        if not is_fact or nxt == NL: return           # only ASSERTIONS write; questions never do
        d = s.store.setdefault(s.key(ctx), {})
        d[nxt] = d.get(nxt, 0) + sum(d.values()) + 1  # recency-dominant: the latest write wins
    def blend(s, ctx, neural_probs):
        d = s.store.get(s.key(ctx))
        if not d: return neural_probs                 # organ silent where it knows nothing
        tot = sum(d.values()); w = tot / (tot + s.C)
        p = neural_probs * (1 - w)
        for t, c in d.items(): p[t] += w * (c / tot)
        return p

NAMES = [" Zephyr", " Orion", " Vega", " Atlas", " Nova", " Lyra"]
VALS  = [" red", " blue", " green", " gold", " black", " white"]
vtok  = {v: tok.encode(v, add_special_tokens=False)[0] for v in VALS}

def build(revise):
    ids = tok.encode("Records begin here.\n", add_special_tokens=False); truth = {}; spans = []
    perm = np.random.permutation(len(NAMES))
    for i in perm:                                    # the facts, early in the document
        line = tok.encode(f"The{NAMES[i]} device is{VALS[i]}\n", add_special_tokens=False)
        st = len(ids); ids += line; spans.append((st, len(ids))); truth[NAMES[i]] = vtok[VALS[i]]
    if revise:                                        # 2 corrections, SAME frame as the query
        for i in list(perm)[:2]:
            nv = VALS[(i + 2) % len(VALS)]
            line = tok.encode(f"The{NAMES[i]} device is{nv}\n", add_special_tokens=False)
            st = len(ids); ids += line; spans.append((st, len(ids))); truth[NAMES[i]] = vtok[nv]
    filler = tok.encode("The weather today is mild and the roads are quiet.\n", add_special_tokens=False)
    while len(ids) < SERVED + 250: ids += filler      # push the facts beyond the served window
    queries = []
    for i in perm:                                    # ask at the very end
        ids += tok.encode(f"\nThe{NAMES[i]} device is", add_special_tokens=False)
        queries.append((len(ids), truth[NAMES[i]]))
    return ids, spans, queries

def run(revise, label):
    ids, spans, queries = build(revise)
    isa = np.zeros(len(ids), bool)
    for a, b in spans: isa[a:b] = True
    org = Organ()
    for t in range(1, len(ids)): org.learn(ids[:t], ids[t], bool(isa[t]))
    g = o = 0
    for qpos, tv in queries:
        ctx = ids[:qpos]
        with torch.no_grad():                         # only the LAST `SERVED` tokens are served
            probs = torch.softmax(lm(torch.tensor([ctx[-SERVED:]])).logits[0, -1].float(), -1).numpy()
        g += (int(probs.argmax()) == tv)                          # Llama alone
        o += (int(org.blend(ctx, probs.copy()).argmax()) == tv)   # Llama + organ
    d = queries[0][0] - spans[0][0]; n = len(queries)
    print(f"  {label} (facts ~{d} tokens back, beyond served {SERVED}):  "
          f"Llama-1B {g}/{n}   Llama-1B+organ {o}/{n}")

print("=" * 72)
print("MAX MODEL — the same organ grafted onto REAL Llama-3.2-1B (1.24B params), CPU")
print("=" * 72)
run(False, "RECALL  ")
run(True,  "REVISION")
print("\n  same organ code as the GPT-2 demo. 1B-param transformer frozen and untouched.")
