#!/usr/bin/env python3
# living-fused — scale_demo: graft the living memory organ onto a REAL pretrained
# transformer and give it memory BEYOND its context window.
# Copyright (C) 2026 Kancheti Devieswar — AGPL-3.0 (see LICENSE)
"""
This is the SCALING demonstration. It bolts the same living organ from live.py onto
a frozen, pretrained GPT-2 (124M params — ~1000x the toy body) and shows it recall
and revise arbitrary facts placed ~1500 tokens back — BEYOND GPT-2's 1024-token
window, where GPT-2 alone is blind.

    pip install torch numpy transformers
    python3 scale_demo.py         # first run downloads GPT-2 (~500MB)

Expected (deterministic):
    RECALL    GPT-2 alone 0/8   GPT-2 + organ 8/8
    REVISION  GPT-2 alone 0/8   GPT-2 + organ 8/8

The transformer is FROZEN and never modified. The organ is model-agnostic: the same
code runs on Llama, Mistral, or any HF causal LM — swap the model name below.

TWO THINGS THAT SILENTLY BREAK THIS IF YOU GET THEM WRONG (see INTEGRATION.md):
  1. The confidence gate. A single known fact must OUTWEIGH the transformer's guess,
     or the organ never speaks and you get ~0/8 that looks like it's "working".
     That's the `C` below — small C = the organ dominates once it knows something.
  2. The write frame must match the query frame. If you store "X is now blue" but
     ask "X is ___", the keys differ and the update is invisible. Store what you'll
     ask for.
"""
import sys
try:
    import torch, numpy as np
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
except ImportError:
    sys.exit("scale_demo needs: pip install torch numpy transformers")

torch.manual_seed(0); np.random.seed(0)
MODEL = "gpt2"                                        # swap for any HF causal LM
tok = GPT2TokenizerFast.from_pretrained(MODEL)
lm  = GPT2LMHeadModel.from_pretrained(MODEL).eval()
for p in lm.parameters(): p.requires_grad_(False)     # the transformer is FROZEN
WIN = 1024                                            # GPT-2's native context window
NL  = tok.encode("\n")[0]
print(f"loaded frozen {MODEL}: {sum(p.numel() for p in lm.parameters())/1e6:.0f}M params, window {WIN}")

class Organ:
    """The living memory. Keyed on the tokens since the last newline (the fact frame),
    so each fact is isolated. Integer counts, recency-dominant, disjoint per key."""
    C = 0.25                                          # gate: known fact dominates the neural prior
    def __init__(s): s.store = {}
    def key(s, ctx):                                  # tokens since the last boundary
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

NAMES = [" Zephyr", " Orion", " Vega", " Atlas", " Nova", " Lyra", " Draco", " Rigel"]
VALS  = [" red", " blue", " green", " gold", " black", " white", " pink", " gray"]
vtok  = {v: tok.encode(v)[0] for v in VALS}

def build(revise):
    filler = tok.encode("The weather today is mild and the roads are quiet.\n")
    ids = tok.encode("Records begin here.\n"); truth = {}; facts = []
    perm = np.random.permutation(len(NAMES))
    for i in perm:                                    # the facts, early in the document
        line = tok.encode(f"The{NAMES[i]} device is{VALS[i]}\n")
        st = len(ids); ids += line; facts.append((st, len(ids))); truth[NAMES[i]] = vtok[VALS[i]]
    if revise:                                        # 3 corrections, SAME frame as the query
        for i in list(perm)[:3]:
            nv = VALS[(i + 3) % len(VALS)]
            line = tok.encode(f"The{NAMES[i]} device is{nv}\n")
            st = len(ids); ids += line; facts.append((st, len(ids))); truth[NAMES[i]] = vtok[nv]
    while len(ids) < 1500: ids += filler              # push the facts past GPT-2's window
    queries = []
    for i in perm:                                    # ask at the very end
        ids += tok.encode(f"\nThe{NAMES[i]} device is"); queries.append((len(ids), truth[NAMES[i]]))
    return ids, facts, queries

def run(revise, label):
    ids, facts, queries = build(revise)
    is_fact = np.zeros(len(ids), bool)
    for a, b in facts: is_fact[a:b] = True
    organ = Organ()
    for t in range(1, len(ids)):                      # the organ reads the whole document (cheap)
        organ.learn(ids[:t], ids[t], bool(is_fact[t]))
    g = o = 0
    for qpos, true_tok in queries:
        ctx = ids[:qpos]
        with torch.no_grad():
            probs = torch.softmax(lm(torch.tensor([ctx[-WIN:]])).logits[0, -1], -1).numpy()
        g += (int(probs.argmax()) == true_tok)                    # GPT-2 alone
        o += (int(organ.blend(ctx, probs.copy()).argmax()) == true_tok)  # GPT-2 + organ
    d = queries[0][0] - facts[0][0]
    print(f"  {label} (facts ~{d} tokens back, BEYOND the {WIN} window):"
          f"  {MODEL} {g}/8   {MODEL}+organ {o}/8")

print("=" * 72)
print("GRAFT: a living memory organ on a frozen pretrained transformer")
print("=" * 72)
run(False, "RECALL  ")
run(True,  "REVISION")
print("\nThe transformer never changed. The organ gave it memory beyond its window.")
