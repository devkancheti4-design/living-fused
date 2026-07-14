#!/usr/bin/env python3
"""
EXPERIMENT 10b — refinement forced by exp10's talk-back (Rule 17):
At 32 facts, llama3.2:3b was ~flat to k=16 (96.9%, errors = format not swaps) —
load too low to see the cliff the theory predicts. Raise the load and add the
confusability axis exp3 says is the killer:

  N = 128 facts in context.
  k in {4, 16, 32, 64} facts queried simultaneously.
  DISTINCT condition:   128 well-separated names.
  CONFUSABLE condition: 128 names sharing a 5-char stem, differing in 1-2 chars
                        (the c->1 near-duplicate-key regime of exp3/exp4).

Predictions: decline with k appears at this load; confusable names shift errors
toward SWAPS and steepen the decline. Null: flat per-fact accuracy, no swap excess.
"""
import json
import re
import sys
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"
TRIALS = 6
KS = [4, 16, 32, 64]
N = 128


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


ONSETS = ["Ba", "Ce", "Do", "Fe", "Ga", "He", "Ji", "Ko", "Lu", "Me", "No", "Pa"]
RIMES = ["ldur", "rnik", "stel", "vron", "mbly", "xton", "quor", "zmin", "drel", "fost", "gwin", "thal"]


def distinct_names():
    names = []
    for o in ONSETS:
        for r in RIMES:
            names.append(o + r)
            if len(names) == N:
                return names
    return names


def confusable_names():
    # shared stem 'Karv' + systematic tails: Karvban, Karvbar, ... maximally similar
    vowels = "aeiou"
    cons = "bdgklmnprstv"
    finals = "nrl"
    names = []
    for c in cons:
        for v in vowels:
            for f in finals:
                names.append("Karv" + c + v + f)
                if len(names) == N:
                    return names
    return names


def ask(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "seed": 42, "num_predict": 1200}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]


def run_condition(cond_name, names):
    rng = LCG(seed=777)
    values, used = {}, set()
    for n in names:
        v = 100 + rng.next() % 900
        while v in used:
            v = 100 + rng.next() % 900
        used.add(v)
        values[n] = v
    facts = "\n".join(f"The code for {n} is {values[n]}." for n in names)
    all_vals = set(values.values())

    print(f"\n--- condition: {cond_name} (N={N} facts) ---")
    print(f"{'k':>3} {'per-fact acc':>13} {'swaps':>7} {'fabr/miss':>10} {'n':>5}")
    out = {}
    for k in KS:
        qrng = LCG(seed=2000 + k)
        correct = swaps = fabr = total = 0
        for _ in range(TRIALS):
            queried = qrng.sample(names, k)
            prompt = (f"Here is a list of codes.\n\n{facts}\n\n"
                      f"What is the code for each of the following: {', '.join(queried)}?\n"
                      f"Answer with one line per name in the exact format 'Name: number'. "
                      f"No other text.")
            resp = ask(prompt)
            for name in queried:
                total += 1
                m = re.search(rf"\b{name}\b\s*[:\-]?\s*(\d{{3}})", resp, re.IGNORECASE)
                if not m:
                    fabr += 1
                    continue
                got = int(m.group(1))
                if got == values[name]:
                    correct += 1
                elif got in all_vals:
                    swaps += 1
                else:
                    fabr += 1
        acc = correct / total * 100
        out[k] = (acc, swaps, fabr, total)
        print(f"{k:>3} {acc:>12.1f}% {swaps:>7} {fabr:>10} {total:>5}")
    return out


COND = sys.argv[2] if len(sys.argv) > 2 else "both"
d = run_condition("DISTINCT", distinct_names()) if COND in ("both", "distinct") else None
c = run_condition("CONFUSABLE", confusable_names()) if COND in ("both", "confusable") else None

print("\nsummary:")
for label, res in (("DISTINCT", d), ("CONFUSABLE", c)):
    if res:
        print(f"  {label:10} k=4 {res[4][0]:.1f}% -> k=64 {res[64][0]:.1f}%  (delta {res[64][0] - res[4][0]:+.1f})"
              f"   swaps {sum(res[k][1] for k in KS)} / errors {sum(res[k][1] + res[k][2] for k in KS)}")
