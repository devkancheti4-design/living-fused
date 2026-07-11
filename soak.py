#!/usr/bin/env python3
# living-fused — infinite-stream soak of the REAL Life memory organ (loaded from live.py).
# Copyright (C) 2026 Kancheti Devieswar — AGPL-3.0 (see LICENSE)
"""
Soaks the living memory organ on a never-ending token stream and prints a timestamped
checkpoint every ~40s: elapsed, tokens, tok/s, process RSS, all-passkey recall (with a
FULL-table write-guard), a recency-dominant revision test every 2B tokens, and the organ
hash. Reports only measured numbers.   Usage: python3 soak.py [run_seconds]

Honest scope: this soaks the Life MEMORY organ (a fixed V*V int64 table) on a synthetic
noise stream — NOT the end-to-end fused transformer. It tests: zero memory growth over
billions of tokens, flat throughput, stable recall (100% by construction — passkeys and
filler occupy disjoint rows), and numeric stability of the cap-and-rescale. See
INFINITE_STREAM_SOAK.md for a full run and caveats.
"""
import sys, os, time, subprocess, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "live.py")).read().split("\n")   # load the REAL Life defs
g = {}; exec(compile("\n".join(SRC[:143]), "live.py[defs]", "exec"), g)
V, KEYS, VALS, QUERY = g["V"], g["KEYS"], g["VALS"], g["QUERY"]
NOISE_LO, NOISE_HI = g["NOISE_LO"], g["NOISE_HI"]
Life = g["Life"]

def rss_kb():
    try: return int(subprocess.check_output(["ps","-o","rss=","-p",str(os.getpid())]).strip())
    except Exception: return -1

def log(m): sys.stdout.write(m + "\n"); sys.stdout.flush()

RUN_SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
CKPT_TOKENS = 20_000_000
REVISE_EVERY = 2_000_000_000
UNIFORM = np.full(V, 1.0 / V)

life = Life()
rng = np.random.default_rng(2026)
pk_key = list(range(10)); pk_val = [VALS[(i*3) % len(VALS)] for i in range(10)]
for i in range(10): life.learn(pk_key[i], pk_val[i])          # 10 passkeys, disjoint key rows

def recall_and_guard():
    ok = 0; guard_ok = True
    for i in range(10):
        before = life.table.copy()                            # FULL table, catches a write into any row
        life.learn(QUERY, pk_key[i]); life.learn(pk_key[i], QUERY)   # both guard branches must no-op
        if not np.array_equal(before, life.table): guard_ok = False
        if int(np.argmax(life.blend(pk_key[i], UNIFORM))) == pk_val[i]: ok += 1
    return ok, guard_ok

log(f"# soak: real Life table {V}x{V} int64 = {V*V*8} bytes fixed | run_seconds={RUN_SECONDS}")
log(f"# {'elapsed':>9} {'tokens':>16} {'tok/s':>10} {'RSS_KB':>9} {'recall':>7} {'guard':>6} {'organ_sha':>14} {'revision':>10}")
t0 = time.time(); n = 0; win_t = t0; win_n = 0; last_rev = 0; rev = "-"
prev = int(rng.integers(NOISE_LO, NOISE_HI)); next_ck = CKPT_TOKENS
while True:
    block = rng.integers(NOISE_LO, NOISE_HI, 1_000_000)
    for x in block:
        x = int(x); life.learn(prev, x); prev = x; n += 1
    if n - last_rev >= REVISE_EVERY:                           # recency-dominant overwrite test
        newv = VALS[(pk_val[0]-VALS[0]+1) % len(VALS)]; life.learn(0, newv)
        rev = "OK" if int(np.argmax(life.blend(0, UNIFORM))) == newv else "FAIL"
        pk_val[0] = newv; last_rev = n
    if n >= next_ck:
        now = time.time(); tps = (n - win_n) / max(now - win_t, 1e-9); win_t = now; win_n = n
        ok, guard = recall_and_guard(); el = now - t0
        log(f"  {int(el//3600):02d}:{int(el%3600//60):02d}:{int(el%60):02d} {n:>16,} {tps:>10,.0f} "
            f"{rss_kb():>9,} {ok:>5}/10 {'HELD' if guard else 'FAIL':>6} {life.sha():>14} {rev:>10}")
        next_ck += CKPT_TOKENS
        if el >= RUN_SECONDS:
            log(f"# DONE: {el/3600:.2f}h, {n:,} tokens, recall {ok}/10, guard {'HELD' if guard else 'FAIL'}")
            break
