#!/usr/bin/env python3
"""
EXPERIMENT 14 — tests Finding 4 (P-4): "capability requiring exactness/honesty lives
in the SYSTEM, not the model" — concretely: a zero-cost exact-store cross-check
catches every silent swap the model makes on batch recall.

Same harness as exp10b (128 facts, k=64, distinct names, llama3.2:3b, temp 0), but
now run as a SYSTEM: the same context lines also populate an exact-key dict (the
Life organ), and every model answer is verified against it before delivery.

  MODEL ALONE : whatever it answers, you get — errors are silent (exp5b/exp10b).
  SYSTEM      : answers cross-checked; mismatches caught and corrected from store.

The point is precisely that the organ is trivial: O(1) per check, built from the
same bytes the model already read. Triviality is the argument, not a weakness.
"""
import json
import re
import sys
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"
TRIALS, K, N = 3, 64, 128

ONSETS = ["Ba", "Ce", "Do", "Fe", "Ga", "He", "Ji", "Ko", "Lu", "Me", "No", "Pa"]
RIMES = ["ldur", "rnik", "stel", "vron", "mbly", "xton", "quor", "zmin", "drel", "fost", "gwin", "thal"]


class LCG:
    def __init__(self, seed=13579):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s >> 33

    def sample(self, xs, k):
        pool = list(xs)
        return [pool.pop(self.next() % len(pool)) for _ in range(k)]


names = []
for o in ONSETS:
    for r in RIMES:
        names.append(o + r)
        if len(names) == N:
            break
    if len(names) == N:
        break
rng = LCG(seed=777)
values, used = {}, set()
for n in names:
    v = 100 + rng.next() % 900
    while v in used:
        v = 100 + rng.next() % 900
    used.add(v)
    values[n] = v
FACTS = "\n".join(f"The code for {n} is {values[n]}." for n in names)

# THE SYSTEM ORGAN: the same lines, parsed once into an exact-key store. O(1) reads.
store = {}
for line in FACTS.split("\n"):
    m = re.match(r"The code for (\w+) is (\d+)\.", line)
    store[m.group(1).lower()] = int(m.group(2))


def ask(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "seed": 42, "num_predict": 1200}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]


model_ok = caught = missing = total = 0
for t in range(TRIALS):
    qrng = LCG(seed=5000 + t)
    queried = qrng.sample(names, K)
    resp = ask(f"Here is a list of codes.\n\n{FACTS}\n\n"
               f"What is the code for each of the following: {', '.join(queried)}?\n"
               f"Answer with one line per name in the exact format 'Name: number'. No other text.")
    for name in queried:
        total += 1
        truth = store[name.lower()]                      # system verification, O(1)
        m = re.search(rf"\b{name}\b\s*[:\-]?\s*(\d{{3}})", resp, re.IGNORECASE)
        if m and int(m.group(1)) == truth:
            model_ok += 1
        elif m:
            caught += 1    # model answered WRONG; store catches + corrects it
        else:
            missing += 1   # model gave nothing parseable; store fills it

print(f"model = {MODEL}, {TRIALS} trials x k={K} simultaneous queries over {N} facts\n")
print(f"  MODEL ALONE : {model_ok}/{total} correct = {model_ok / total * 100:.1f}%  "
      f"({caught} wrong answers delivered SILENTLY, {missing} dropped)")
print(f"  SYSTEM      : {total}/{total} correct = 100.0%  "
      f"({caught} swaps caught+corrected, {missing} gaps filled, all at O(1)/query)")
print(f"\nverification cost: a dict lookup — no second model, no training, no tokens.")
print("the capability gap between the two rows is a SYSTEM property; the model is identical.")
