#!/usr/bin/env python3
"""
EXPERIMENT 4 — mined by the engine in round 4 (AC-2, score 0.790):
"What is the measured capacity of a d-by-d fast-weight associative matrix as
 key-value writes accumulate, and where is the interference cliff relative to d?"

This is the canonical fast-weight / linear-attention memory (also round 2's C-2
claim (b), asserted then untested): M = sum_i v_i k_i^T, retrieve v_hat = M k_cue.
Crosstalk = sum_{j != cue} v_j (k_j . k_cue) — interference grows with load K.

Design (pure Python, deterministic):
  d = 128. Keys: random unit vectors. Values: 16 fixed unit prototype classes.
  Load K pairs into one d x d matrix; retrieval = nearest prototype to M k_i.
  Sweep K = 8 .. 1024. Control: exact dict — 100% at every K, grows O(K).
  Also the eviction-free view: accuracy of the FIRST 16 writes as load grows
  (does new writing corrupt old memories?).
"""
import math

D, NCLS = 128, 16


class LCG:
    def __init__(self, seed=555555):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s >> 33

    def gauss(self):
        return sum(self.next() / float(1 << 31) for _ in range(12)) - 6.0


def normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


rng = LCG()
prototypes = [normalize([rng.gauss() for _ in range(D)]) for _ in range(NCLS)]


def run_load(K, rng):
    keys = [normalize([rng.gauss() for _ in range(D)]) for _ in range(K)]
    labels = [rng.next() % NCLS for _ in range(K)]
    # M = sum v_label k^T   (store as row-major d x d)
    M = [[0.0] * D for _ in range(D)]
    for k, lab in zip(keys, labels):
        v = prototypes[lab]
        for r in range(D):
            vr = v[r]
            row = M[r]
            for c in range(D):
                row[c] += vr * k[c]
    # retrieve
    def predict(k):
        vhat = [sum(M[r][c] * k[c] for c in range(D)) for r in range(D)]
        return max(range(NCLS), key=lambda i: sum(a * b for a, b in zip(vhat, prototypes[i])))
    hits_all = sum(1 for k, lab in zip(keys, labels) if predict(k) == lab)
    first16 = list(zip(keys, labels))[:16]
    hits_first = sum(1 for k, lab in first16 if predict(k) == lab)
    return hits_all / K * 100, hits_first / len(first16) * 100


print(f"d = {D}; fixed cost of M = d^2 = {D * D} floats regardless of load")
print(f"{'K writes':>9} {'K/d':>6} {'acc all':>9} {'acc first-16':>13} {'dict':>6}")
for K in (8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024):
    acc, acc16 = run_load(K, LCG(seed=42 + K))
    print(f"{K:>9} {K / D:>6.2f} {acc:>8.1f}% {acc16:>12.1f}% {'100%':>6}")

print("\nreading: 'acc first-16' shows whether later writes destroy the earliest")
print("memories — fixed-size distributed memory pays for every new write with")
print("interference on ALL old ones; the dict pays with O(K) growth instead.")
