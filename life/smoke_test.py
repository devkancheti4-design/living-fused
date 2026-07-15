#!/usr/bin/env python3
"""Smoke test for life.py — the core storage claims, measured, pass/fail.
(Model-fusion and paraphrase claims are NOT covered here — see README §6
for each number's provenance.)

Deterministic checks (same result on every machine) are marked [EXACT].
Machine-dependent measurements (latency, RAM) are marked [HW] and are
reported, not gated.

Run:  python3 smoke_test.py          (~10 s, stdlib only)
Exit: 0 = all [EXACT] checks pass, 1 = any failure.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from life import Life

FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


print("=" * 68)
print("1. EXACT RECALL — 10,000 facts stored, 10,000 recalled verbatim [EXACT]")
print("=" * 68)
life = Life()
for i in range(10_000):
    life.learn(f"user.fact{i}", f"value-{i}")
hits = sum(life.recall(f"user.fact{i}") == f"value-{i}" for i in range(10_000))
check("exact recall", hits == 10_000, f"{hits}/10,000")

print("\n" + "=" * 68)
print("2. REVISION — newest wins, history retained (a dict destroys it) [EXACT]")
print("=" * 68)
life.learn("wifi.password", "SunFlower42")
life.learn("wifi.password", "Rose99")
check("newest wins", life.recall("wifi.password") == "Rose99")
hist = [v for v, _ in life.history("wifi.password")]
check("history retained", hist == ["SunFlower42", "Rose99"], str(hist))

print("\n" + "=" * 68)
print("3. STRUCTURAL ABSTENTION — 1,000 unseen keys, 0 guesses [EXACT]")
print("=" * 68)
abstains = sum(life.recall(f"never.seen{i}") is None for i in range(1_000))
check("abstains on every unseen key", abstains == 1_000, f"{abstains}/1,000")

print("\n" + "=" * 68)
print("4. DETERMINISM — twin Lives fed the same stream are byte-identical [EXACT]")
print("=" * 68)
a, b = Life(), Life()
for l in (a, b):
    for i in range(500):
        l.learn(f"k{i}", f"v{i}")
    l.learn("k0", "revised")
    l.link("k1", "k2")
check("twin sha match", a.sha() == b.sha(), a.sha())
GOLDEN = "0ffd7ccc8e97f01b"  # sha of exactly this stream, any machine
check("golden sha (cross-machine)", a.sha() == GOLDEN,
      f"got {a.sha()}, expect {GOLDEN}")

print("\n" + "=" * 68)
print("5. SURVIVES PROCESS DEATH — save, die, fresh process recalls [EXACT]")
print("=" * 68)
tmp = tempfile.mktemp(suffix=".json")
a.save(tmp)
code = (
    "import sys; sys.path.insert(0, %r); from life import Life; "
    "l = Life(%r); print(l.sha()); print(l.recall('k0')); "
    "print(l.recall('nope') is None)"
) % (os.path.dirname(os.path.abspath(__file__)), tmp)
out = subprocess.run([sys.executable, "-c", code],
                     capture_output=True, text=True).stdout.split()
check("cross-process sha byte-exact", out[0] == a.sha(), out[0])
check("cross-process recall verbatim", out[1] == "revised")
check("cross-process abstention", out[2] == "True")
os.unlink(tmp)

print("\n" + "=" * 68)
print("6. RELATIONAL CHAIN — multi-hop retrieval a flat dict can't do [EXACT]")
print("=" * 68)
r = Life()
r.learn("HANDSHAKE_ID", "0x3F9A")
r.link("HANDSHAKE_ID", "RULE_handshake")
r.link("RULE_handshake", "ENV")
r.learn("ENV", "PROD")
ch = r.chain("HANDSHAKE_ID")
check("3-hop chain", ch == ["HANDSHAKE_ID", "RULE_handshake", "ENV"], str(ch))

print("\n" + "=" * 68)
print("7. FUSION MATH — blend() is gated, normalized, and honest [EXACT]")
print("=" * 68)
f = Life()
model_probs = {"paris": 0.6, "london": 0.3, "rome": 0.1}
untouched = f.blend("capital.x", model_probs)
check("absent key leaves model untouched", untouched == model_probs)
f.learn("capital.x", "tokyo")
q = f.blend("capital.x", model_probs)
check("stored value dominates softmax", max(q, key=q.get) == "tokyo",
      f"tokyo={q['tokyo']:.3f}")
check("distribution still sums to 1", abs(sum(q.values()) - 1.0) < 1e-9)

print("\n" + "=" * 68)
print("8. CLI — round-trip behavior and exit codes [EXACT + HW timing]")
print("=" * 68)
cli = os.path.join(os.path.dirname(os.path.abspath(__file__)), "life.py")
db = tempfile.mktemp(suffix=".json")
env = dict(os.environ, LIFE_DB=db)
subprocess.run([sys.executable, cli, "put", "cli.test", "hello world"],
               capture_output=True, env=env)
t0 = time.perf_counter()
r_hit = subprocess.run([sys.executable, cli, "get", "cli.test"],
                       capture_output=True, text=True, env=env)
ms = (time.perf_counter() - t0) * 1e3
r_miss = subprocess.run([sys.executable, cli, "get", "cli.absent"],
                        capture_output=True, text=True, env=env)
check("CLI put->get verbatim", r_hit.stdout.strip() == "hello world"
      and r_hit.returncode == 0)
check("CLI abstain prints ABSTAIN, exit 3",
      r_miss.stdout.strip() == "ABSTAIN" and r_miss.returncode == 3)
print(f"  CLI round-trip {ms:.0f} ms [HW] — interpreter + file load;"
      " the µs figure below is in-process recall()")
os.path.exists(db) and os.unlink(db)
# doctor: red in a bare dir, green in a fused dir with a fact
docdir = tempfile.mkdtemp()
shutil.copy(cli, os.path.join(docdir, "life.py"))
denv = dict(os.environ, LIFE_DB=os.path.join(docdir, "life.json"))
r_bare = subprocess.run([sys.executable, "life.py", "doctor"],
                        capture_output=True, text=True, cwd=docdir, env=denv)
check("doctor: NOT WIRED in a bare dir (exit 1)",
      "NOT WIRED" in r_bare.stdout and r_bare.returncode == 1)
with open(os.path.join(docdir, "CLAUDE.md"), "w") as fh:
    fh.write("<!-- LIFE-MEMORY BEGIN -->\nprotocol\n<!-- LIFE-MEMORY END -->\n")
r_wired = subprocess.run([sys.executable, "life.py", "doctor"],
                         capture_output=True, text=True, cwd=docdir, env=denv)
check("doctor: WIRED, NOT YET USED when fused but no fact (exit 0)",
      "WIRED, NOT YET USED" in r_wired.stdout and r_wired.returncode == 0,
      f"rc={r_wired.returncode}")
subprocess.run([sys.executable, "life.py", "put", "server.ip", "10.0.0.1"],
               capture_output=True, cwd=docdir, env=denv)
r_green = subprocess.run([sys.executable, "life.py", "doctor"],
                         capture_output=True, text=True, cwd=docdir, env=denv)
check("doctor: FUSED AND WORKING once fused + a fact exists (exit 0)",
      "FUSED AND WORKING" in r_green.stdout and r_green.returncode == 0,
      f"rc={r_green.returncode}")
shutil.rmtree(docdir, ignore_errors=True)

print("\n" + "=" * 68)
print("9. SCALE — 100,000 facts: latency + memory [HW — reported, not gated]")
print("=" * 68)
tracemalloc.start()
big = Life()
for i in range(100_000):
    big.learn(f"key.number.{i}", f"value-{i % 1000}")
ram, _ = tracemalloc.get_traced_memory()
tracemalloc.stop()
keys = [f"key.number.{(i * 7919) % 100_000}" for i in range(50_000)]
t0 = time.perf_counter()
for k in keys:
    big.recall(k)
us = (time.perf_counter() - t0) / len(keys) * 1e6
print(f"  100,000 facts: recall {us:.2f} us/lookup (O(1) flat), "
      f"RAM {ram / 1e6:.0f} MB = {ram / 100_000:.0f} B/fact  [HW]")
check("recall stays exact at 100k",
      big.recall("key.number.99999") == f"value-{99999 % 1000}")

print("\n" + "=" * 68)
n = len(FAILS)
print(f"RESULT: {'ALL EXACT CHECKS PASS' if not n else str(n) + ' FAILURES: ' + ', '.join(FAILS)}")
print("=" * 68)
sys.exit(1 if FAILS else 0)
