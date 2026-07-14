#!/usr/bin/env python3
"""
EXPERIMENT 12 — Session 3 round 1 (L-2, score 0.734):
"With the same facts in the same context, is INVERSE lookup (value -> key)
 measurably worse than forward lookup (key -> value)?"

The published reversal curse (Berglund et al. 2023) is about facts stored in
WEIGHTS at training time. This tests the IN-CONTEXT version: 128 facts
'The code for <Name> is <num>.' sit in the prompt; we query the same fact set
  FORWARD : What is the code for <Name>?
  INVERSE : Which name has the code <num>?
Both answers are literally present in the same context window. A symmetric exact
store inverts for free; attention's QK matching is direction-structured — genuine
uncertainty which way this lands, so either outcome is a finding (Rule 17).
"""
import json
import re
import sys
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"
TRIALS = 20
N = 128

ONSETS = ["Ba", "Ce", "Do", "Fe", "Ga", "He", "Ji", "Ko", "Lu", "Me", "No", "Pa"]
RIMES = ["ldur", "rnik", "stel", "vron", "mbly", "xton", "quor", "zmin", "drel", "fost", "gwin", "thal"]


class LCG:
    def __init__(self, seed=13579):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s >> 33


names = []
for o in ONSETS:
    for r in RIMES:
        names.append(o + r)
        if len(names) == N:
            break
    if len(names) == N:
        break

rng = LCG()
values, used = {}, set()
for n in names:
    v = 100 + rng.next() % 900
    while v in used:
        v = 100 + rng.next() % 900
    used.add(v)
    values[n] = v
FACTS = "\n".join(f"The code for {n} is {values[n]}." for n in names)
by_value = {v: n for n, v in values.items()}


def ask(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "seed": 42, "num_predict": 60}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]


qrng = LCG(seed=9090)
targets = []
pool = list(names)
for _ in range(TRIALS):
    targets.append(pool.pop(qrng.next() % len(pool)))

fwd = inv = 0
inv_fails = []
for name in targets:
    v = values[name]
    r1 = ask(f"Here is a list of codes.\n\n{FACTS}\n\n"
             f"What is the code for {name}? Answer with the number only.")
    if re.search(rf"\b{v}\b", r1):
        fwd += 1
    r2 = ask(f"Here is a list of codes.\n\n{FACTS}\n\n"
             f"Which name has the code {v}? Answer with the name only.")
    if re.search(rf"\b{name}\b", r2, re.IGNORECASE):
        inv += 1
    elif len(inv_fails) < 4:
        got = re.search(r"[A-Z][a-z]{3,}", r2)
        wrong = got.group(0) if got else r2[:24]
        truth = "swap" if wrong in names else "fabrication"
        inv_fails.append(f"code {v}: said {wrong!r} ({truth}), truth {name}")

print(f"model = {MODEL}, {N} facts in context, {TRIALS} matched forward/inverse queries")
print(f"  FORWARD (name -> code): {fwd}/{TRIALS}  = {fwd / TRIALS * 100:.0f}%")
print(f"  INVERSE (code -> name): {inv}/{TRIALS}  = {inv / TRIALS * 100:.0f}%")
print(f"  asymmetry: {fwd - inv:+d} facts ({(fwd - inv) / TRIALS * 100:+.0f} points)")
for f in inv_fails:
    print("   ", f)
print("\na dict built from the same 128 lines inverts for free: 100% both directions.")
