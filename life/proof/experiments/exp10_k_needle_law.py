#!/usr/bin/env python3
"""
EXPERIMENT 10 — Winner-3 claim tested on REAL LLMs (Session 2, prediction from exp4):
"An LLM should start mixing up facts once asked to hold too many at the same time."

Precise, falsifiable form (the part the literature's multi-needle benchmarks do not
state): PER-FACT accuracy should DECLINE as k (facts queried simultaneously) grows —
rejecting the independence null (per-fact accuracy flat in k, total = p^k) — and the
errors should be SWAPS (another context fact's value) rather than fabrications,
because superposition crosstalk predicts binding confusion, not random noise.

Design:
  Context: 32 fact pairs "The code for <Name> is <3-digit>." (unique values -> swaps
  are detectable). Same 32 facts every trial; only the QUERY SIZE k varies.
  k in {1, 2, 4, 8, 16}; T trials per k, deterministic key sampling (LCG).
  temperature 0, fixed seed, via local ollama API. Models: llama3.2:3b, llama3:8b.
Score: per-queried-fact accuracy; error decomposition swap vs fabrication.
"""
import json
import re
import sys
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"
TRIALS = 10
KS = [1, 2, 4, 8, 16]

NAMES = ["Balor", "Cedrin", "Dovak", "Fenwick", "Garrod", "Hestia", "Ilmar", "Jorvun",
         "Kelsit", "Lomand", "Merrin", "Norval", "Opalin", "Prandel", "Quorast", "Rivena",
         "Sortim", "Tuvello", "Umbrik", "Vandor", "Welkin", "Xarnos", "Yelric", "Zombor",
         "Ashvel", "Brontis", "Cinder", "Drossel", "Elvane", "Fumaro", "Gilzen", "Hovart"]


class LCG:
    def __init__(self, seed=13579):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s >> 33

    def sample(self, xs, k):
        pool = list(xs)
        out = []
        for _ in range(k):
            out.append(pool.pop(self.next() % len(pool)))
        return out


rng = LCG()
values = {}
used = set()
for n in NAMES:
    v = 100 + rng.next() % 900
    while v in used:
        v = 100 + rng.next() % 900
    used.add(v)
    values[n] = v

FACTS = "\n".join(f"The code for {n} is {values[n]}." for n in NAMES)


def ask(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "seed": 42, "num_predict": 400}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]


all_values = set(values.values())
print(f"model = {MODEL}; 32 facts in context every trial; {TRIALS} trials per k\n")
print(f"{'k':>3} {'per-fact acc':>13} {'swap errors':>12} {'fabrications':>13} {'n queried':>10}")

results = {}
for k in KS:
    qrng = LCG(seed=1000 + k)
    correct = swaps = fabr = total = 0
    for t in range(TRIALS):
        queried = qrng.sample(NAMES, k)
        qlist = ", ".join(queried)
        prompt = (f"Here is a list of codes.\n\n{FACTS}\n\n"
                  f"What is the code for each of the following: {qlist}?\n"
                  f"Answer with one line per name in the exact format 'Name: number'. "
                  f"No other text.")
        resp = ask(prompt)
        for name in queried:
            total += 1
            m = re.search(rf"{name}\s*[:\-]?\s*(\d{{3}})", resp, re.IGNORECASE)
            if not m:
                fabr += 1  # no parseable answer counts as failure (non-swap)
                continue
            got = int(m.group(1))
            if got == values[name]:
                correct += 1
            elif got in all_values:
                swaps += 1
            else:
                fabr += 1
    acc = correct / total * 100
    results[k] = acc
    print(f"{k:>3} {acc:>12.1f}% {swaps:>12} {fabr:>13} {total:>10}")

print("\nindependence null: per-fact accuracy FLAT in k.")
print(f"measured: k=1 {results[1]:.1f}%  ->  k=16 {results[16]:.1f}%  "
      f"(delta {results[16] - results[1]:+.1f} points)")
print("crosstalk signature: errors dominated by SWAPS (another fact's value) = binding")
print("confusion, exactly what superposition predicts; fabrications = other failure.")
