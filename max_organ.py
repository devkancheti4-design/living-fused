#!/usr/bin/env python3
# living-fused — max_organ: push the living memory layer to a machine's ceiling.
# Copyright (C) 2026 Kancheti Devieswar — AGPL-3.0 (see LICENSE)
"""
How many facts can the living organ hold, how fast does it write/recall, and is it
still byte-exactly deterministic at scale? This measures the memory layer alone (no
transformer) so the numbers are about the data structure, not a model.

    python3 max_organ.py          # pure Python + numpy; no GPU, no downloads

What it does:
  * builds 10K -> 5M disjoint facts, reports write throughput, recall accuracy, RAM;
  * hammers 1000 keys with 2,000,000 overwrites to prove last-write-wins at scale;
  * builds 1,000,000 facts TWICE and hashes both stores to prove byte-identity.

Reference run (Apple M4 Pro, 24GB, Python 3.14 — your speeds will vary with CPU load,
but recall%, RAM, the overwrite count, and the determinism hash are exact):

         facts         write   recall%   recall q/s       RAM
        10,000    ~170,000/s      100%    ~1,800,000       6MB
       100,000    ~170,000/s      100%    ~1,750,000      51MB
     1,000,000    ~150,000/s      100%    ~1,450,000     496MB
     5,000,000    ~160,000/s      100%    ~1,550,000    2437MB
  OVERWRITE:   2,000,000 overwrites on 1000 keys -> latest value correct 1000/1000
  DETERMINISM: 1M facts built twice -> BYTE-IDENTICAL hash

The write-throughput figures fluctuate a few percent run to run (wall-clock on a shared
CPU); recall%, RAM, the overwrite result, and the determinism hash do not — those are
the capability, and they are exact.

NOTE: this reference test omits the periodic rescale from INTEGRATION.md #5. Unbounded
integer counts are bounded in the shipped organ by that rescale (a deterministic bulk
down-scale on overflow); correctness holds either way.
"""
import numpy as np, time, hashlib, tracemalloc

def build(N, seed=0):
    rng = np.random.default_rng(seed)
    # random opaque keys standing in for arbitrary contexts; values are the "facts"
    keys = rng.integers(0, 2**31, (N, 3)); vals = rng.integers(0, 4000, N)
    store = {}; t0 = time.time()
    for i in range(N):
        k = (int(keys[i,0]), int(keys[i,1]), int(keys[i,2])); v = int(vals[i])
        d = store.get(k)
        if d is None: d = store[k] = {}
        d[v] = d.get(v, 0) + sum(d.values()) + 1          # recency-dominant
    return store, keys, vals, time.time() - t0

def recall(store, keys, vals, M):
    ok = 0; t0 = time.time()
    for i in range(M):
        d = store.get((int(keys[i,0]), int(keys[i,1]), int(keys[i,2])))
        if d and max(d, key=d.get) == int(vals[i]): ok += 1
    return ok / M * 100, time.time() - t0

def canon_hash(store):
    h = hashlib.sha256()
    for k in sorted(store):
        h.update(str(k).encode()); h.update(str(sorted(store[k].items())).encode())
    return h.hexdigest()[:16]

print("=" * 70)
print("MAX ORGAN — capacity / speed / determinism on this machine")
print("=" * 70)
print(f"  {'facts':>12s} {'write':>13s} {'recall%':>9s} {'recall q/s':>12s} {'RAM':>9s}")
for N in [10_000, 100_000, 1_000_000, 5_000_000]:
    tracemalloc.start()
    store, keys, vals, wt = build(N)
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    M = min(N, 50_000)
    acc, rt = recall(store, keys, vals, M)
    print(f"  {N:>12,} {N/wt:>10,.0f}/s {acc:>8.0f}% {M/rt:>11,.0f} {peak/1e6:>7.0f}MB")

print("\n  OVERWRITE-AT-SCALE (last-write-wins under heavy contention):")
rng = np.random.default_rng(1); NK = 1000; WR = 2_000_000
store = {}; final = {}
t0 = time.time()
for _ in range(WR):
    k = int(rng.integers(0, NK)); v = int(rng.integers(0, 4000))
    d = store.setdefault(k, {}); d[v] = d.get(v, 0) + sum(d.values()) + 1; final[k] = v
wt = time.time() - t0
ok = sum(max(store[k], key=store[k].get) == final[k] for k in range(NK))
print(f"     {WR:,} overwrites on {NK} keys in {wt:.1f}s ({WR/wt:,.0f}/s) -> latest value correct {ok}/{NK}")

print("\n  DETERMINISM AT SCALE (1M facts built twice):")
s1, k1, v1, _ = build(1_000_000, seed=7); s2, k2, v2, _ = build(1_000_000, seed=7)
h1, h2 = canon_hash(s1), canon_hash(s2)
print(f"     {h1} vs {h2} -> {'BYTE-IDENTICAL' if h1 == h2 else 'MISMATCH'}")
print("=" * 70)
