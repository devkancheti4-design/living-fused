#!/usr/bin/env python3
"""
EXPERIMENT 15 — control forced by a symptom found in cross-experiment analysis:
exp10b measured k=64 batch recall at 87.8% (6 trials), exp14 at 97.9% (3 trials,
different query seeds) — a 10-point gap far beyond binomial noise. Suspicion:
errors are CORRELATED WITHIN a response (one slip in a 64-line output derails
later lines) — a generation cascade, not (only) retrieval capacity.

Test: identical 64-name query sets, two delivery modes:
  SINGLE : one prompt asking all 64        (as exp10b/exp14)
  CHUNKED: eight prompts asking 8 each     (same facts, same names, same order)
If CHUNKED >> SINGLE, a large share of the k-needle decline is autoregressive
cascade — and query chunking is a free mitigation. If equal, capacity stands.
"""
import json
import re
import sys
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"
TRIALS, K, N, CHUNK = 6, 64, 128, 8

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


def ask(prompt, npred):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "seed": 42, "num_predict": npred}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]


def score(resp, queried):
    ok = 0
    for name in queried:
        m = re.search(rf"\b{name}\b\s*[:\-]?\s*(\d{{3}})", resp, re.IGNORECASE)
        if m and int(m.group(1)) == values[name]:
            ok += 1
    return ok


single_ok = chunked_ok = total = 0
per_trial = []
for t in range(TRIALS):
    qrng = LCG(seed=2064)  # exp10b's exact k=64 seed for trial stream
    for _ in range(t + 1):
        queried = qrng.sample(names, K)   # reproduce exp10b's t-th query set
    total += K

    resp = ask(f"Here is a list of codes.\n\n{FACTS}\n\n"
               f"What is the code for each of the following: {', '.join(queried)}?\n"
               f"Answer with one line per name in the exact format 'Name: number'. No other text.",
               1200)
    s_ok = score(resp, queried)
    single_ok += s_ok

    c_ok = 0
    for i in range(0, K, CHUNK):
        part = queried[i:i + CHUNK]
        resp = ask(f"Here is a list of codes.\n\n{FACTS}\n\n"
                   f"What is the code for each of the following: {', '.join(part)}?\n"
                   f"Answer with one line per name in the exact format 'Name: number'. No other text.",
                   250)
        c_ok += score(resp, part)
    chunked_ok += c_ok
    per_trial.append((s_ok, c_ok))

print(f"model = {MODEL}, {TRIALS} trials, identical 64-name query sets per trial\n")
print(f"{'trial':>6} {'single 1x64':>12} {'chunked 8x8':>12}")
for i, (s, c) in enumerate(per_trial):
    print(f"{i:>6} {s:>9}/64 {c:>9}/64")
print(f"\n  SINGLE  (1x64): {single_ok}/{total} = {single_ok / total * 100:.1f}%")
print(f"  CHUNKED (8x8) : {chunked_ok}/{total} = {chunked_ok / total * 100:.1f}%")
gap = (chunked_ok - single_ok) / total * 100
print(f"  chunking recovered {gap:+.1f} points -> "
      f"{'CASCADE component real: decline is partly generation, not retrieval' if gap > 3 else 'no cascade: capacity interpretation stands'}")
