#!/usr/bin/env python3
"""Logit fusion: fold the Life's counts into a real model's answer distribution.

This is mode (b) from the README — the mechanism behind the measured
bare 0/40 · in-context 36/40 · fused 40/40 result.

Protocol (stated honestly):
  * N facts key->value are stored in the Life. Values form a CLOSED candidate
    set (that is what makes single-forward scoring possible; open-ended
    generation fusion needs a decoding loop, not shown here).
  * BARE       : model scores candidates with no facts anywhere. Expect ~chance.
  * IN-CONTEXT : all facts pasted into the prompt. The fair baseline.
  * FUSED      : bare distribution + life.blend(key, probs). The table's
    counts override the softmax on keys it knows, leave it untouched on
    keys it doesn't (that is the abstention gate).

Needs: pip install transformers torch   (any causal LM; default is small
enough for CPU). Everything else is stdlib + life.py.

Run: python3 examples/fuse_logits.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--n 20]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from life import Life


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("needs: pip install transformers torch")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    # -- the facts: arbitrary key -> value bindings the model cannot know --
    pool = ["crimson", "walnut", "glacier", "ember", "quartz", "meadow",
            "cobalt", "raven", "juniper", "onyx", "saffron", "birch",
            "indigo", "maple", "coral", "slate", "amber", "fern",
            "topaz", "cedar"]
    if args.n > len(pool):
        sys.exit(f"--n caps at {len(pool)} (the closed candidate pool); "
                 f"the parent project's 40-fact run is not shipped here")
    values = pool[: args.n]
    keys = [f"project.codeword{i}" for i in range(len(values))]
    life = Life()
    for k, v in zip(keys, values):
        life.learn(k, v)

    @torch.no_grad()
    def score(prompt, candidates):
        """P(candidate | prompt) over the closed set, via first-token logits."""
        ids = tok(prompt, return_tensors="pt").input_ids
        logits = model(ids).logits[0, -1]
        raw = {}
        for c in candidates:
            t = tok(" " + c, add_special_tokens=False).input_ids[0]
            raw[c] = logits[t].item()
        m = max(raw.values())
        exp = {c: math.exp(v - m) for c, v in raw.items()}
        z = sum(exp.values())
        return {c: v / z for c, v in exp.items()}

    ctx = "\n".join(f"{k} = {v}" for k, v in zip(keys, values))
    bare_ok = incontext_ok = fused_ok = 0
    for k, v in zip(keys, values):
        q = f"Question: what is the value of {k}?\nAnswer:"
        p_bare = score(q, values)
        p_ctx = score(f"Facts:\n{ctx}\n\n{q}", values)
        p_fused = life.blend(k, p_bare)          # <- the fusion, one line
        bare_ok += max(p_bare, key=p_bare.get) == v
        incontext_ok += max(p_ctx, key=p_ctx.get) == v
        fused_ok += max(p_fused, key=p_fused.get) == v

    n = len(values)
    print(f"model: {args.model}   facts: {n}   (closed candidate set)")
    print(f"  BARE        {bare_ok}/{n}   (no facts anywhere — chance-level expected)")
    print(f"  IN-CONTEXT  {incontext_ok}/{n}   (fair baseline: facts in the prompt)")
    print(f"  FUSED       {fused_ok}/{n}   (bare + life.blend — no facts in the prompt)")
    print("honest read: fused should tie-or-beat in-context on EXACT keys;")
    print("the Life contributes exactness+permanence, not intelligence.")
    if incontext_ok <= max(2, n // 10):
        print(f"NOTE: the IN-CONTEXT arm collapsed ({incontext_ok}/{n}) — this model"
              " can't use pasted facts under this prompt. Use an instruct model"
              " (e.g. the default Qwen2.5-0.5B-Instruct) for a meaningful fair"
              " baseline; beating a collapsed baseline proves nothing.")


if __name__ == "__main__":
    main()
