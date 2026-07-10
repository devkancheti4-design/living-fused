#!/usr/bin/env python3
# living-fused — a model that never stops learning
# Copyright (C) 2026 Kancheti Devieswar
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE).
"""
THE LIVING FUSED MODEL — this starts a LIFE on your machine.

    git clone <this repo> && cd living-fused && python3 live.py

What happens, in order (all on CPU, ~10-15 min on an Apple-silicon Mac):
  1. A small fused body is trained from scratch: a windowed transformer (the skeleton)
     + a persistent recurrent current. Trained ONLY on streams <= 420 tokens.
  2. The LIFE is switched on: a disjoint integer fact-table that updates on every
     token the model reads. Deployment IS learning. It never freezes.
  3. The organism takes three long-context exams FAR beyond its training length,
     each scored against its own frozen twin (identical weights, life off):
       T1  RULER-style: 10 variables + mid-stream updates @ 32,000 tokens
       T2  chain-of-custody across 64,000 tokens
       T3  code-dependency trace (corrupted var -> use site) @ 20,000 tokens
  4. Determinism receipt: two identical lives -> byte-identical organisms (SHA-256).
  5. Cost of being alive: ms/token at 1K vs 32K context (flat).

Expected results (fixed seeds; your numbers should match):
  living 100% / 100% / 100%   vs frozen 8% / 62% / 12%   (chance ~6%)
Honest scope: this demonstrates ARCHITECTURE-level properties at ~0.1M params on
synthetic streams. It is not a product benchmark and makes no product claims.
"""
import torch, torch.nn as nn, numpy as np, time, hashlib

torch.manual_seed(0); np.random.seed(0)
DEV = torch.device("cpu")
V = 40; W = 32
KEYS = list(range(10)); VALS = list(range(10, 26)); SET = 26; QUERY = 27
NOISE_LO, NOISE_HI = 28, 40
CHANCE = 100.0 / len(VALS)

def band_mask(L):
    i = torch.arange(L)[:, None]; j = torch.arange(L)[None, :]
    m = torch.full((L, L), float("-inf")); m[(j <= i) & (i - j < W)] = 0.0; return m

class FusedBody(nn.Module):
    """the skeleton: windowed transformer + recurrent current (frozen after training)"""
    def __init__(s, d=64, g=96):
        super().__init__(); s.d, s.g = d, g
        s.e = nn.Embedding(V, d); s.p = nn.Embedding(W, d)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, 4, 2*d, 0.0, batch_first=True, activation="gelu"), 2)
        s.cur = nn.GRUCell(d, g); s.head = nn.Linear(d+g, V)
    def forward_train(s, X):
        B, L = X.shape
        z = s.tr(s.e(X) + s.p(torch.arange(L) % W), mask=band_mask(L))
        h = torch.zeros(B, s.g); em = s.e(X); hs = []
        for t in range(L): h = s.cur(em[:, t], h); hs.append(h)
        return s.head(torch.cat([z, torch.stack(hs, 1)], -1))

class Life:
    """the living organ: disjoint integer fact-table, recency-dominant writes.
    - one observation stores a fact; a later write always outvotes it (registers)
    - learning key A physically cannot touch key B's row (zero forgetting)
    - questions are not world-facts: QUERY transitions never write the store
    - pure integers -> a byte-exact, deterministic life"""
    def __init__(s): s.table = np.zeros((V, V), dtype=np.int64)
    def blend(s, prev, neural):
        row = s.table[prev]; tot = int(row.sum())
        if tot <= 0: return neural
        wt = tot / (tot + 0.25)
        return wt * (row / tot) + (1 - wt) * neural
    def learn(s, prev, nxt):
        if prev == QUERY or nxt == QUERY: return
        s.table[prev][nxt] += s.table[prev].sum() + 1
        if s.table[prev].sum() > 1 << 40: s.table[prev] >>= 20   # deterministic cap
    def sha(s): return hashlib.sha256(s.table.tobytes()).hexdigest()[:16]

def make_stream(L, nkeys, rng):
    s = rng.integers(NOISE_LO, NOISE_HI, L).astype(np.int64)
    keys = list(rng.choice(KEYS, nkeys, replace=False)); tail = 4 * nkeys + 4
    ev = []
    for k in keys:
        for _ in range(int(rng.integers(2, 4))):
            ev.append((int(rng.integers(2, L - tail - 3)), k, int(rng.choice(VALS))))
    for p, k, v in sorted(ev): s[p:p+3] = [SET, k, v]
    law = {}
    for p in range(L - 2):
        if s[p] == SET and s[p+1] in KEYS and s[p+2] in VALS: law[int(s[p+1])] = int(s[p+2])
    pos = L - tail; ans = []
    for k in keys:
        if k not in law: continue
        s[pos:pos+3] = [QUERY, k, law[k]]; ans.append((pos + 2, law[k])); pos += 3
    return s, ans

def train(model, steps=2400, B=24):
    print("  [1/5] training the body (streams <= 420 tokens — remember this number)")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(1); t0 = time.time()
    for st in range(steps):
        L = int(rng.integers(90, 420))
        Xs, tgt = [], []
        for b in range(B):
            s, a = make_stream(L, int(rng.integers(2, 6)), rng); Xs.append(s); tgt.append(a)
        X = torch.tensor(np.stack(Xs)); logits = model.forward_train(X)
        ls = []
        for b, a in enumerate(tgt):
            for p, v in a: ls.append(nn.functional.cross_entropy(logits[b, p-1][None], torch.tensor([v])))
        loss = torch.stack(ls).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 400 == 0:
            el = time.time() - t0; eta = el / (st + 1) * (steps - st - 1)
            print(f"        step {st:>4d}/{steps}  loss {loss.item():.3f}   elapsed {el:>4.0f}s  eta {eta:>4.0f}s")
    for p in model.parameters(): p.requires_grad_(False)
    return model

@torch.no_grad()
def walk(model, streams, answers, living):
    B = len(streams); L = len(streams[0])
    X = torch.tensor(np.stack(streams))
    win = torch.full((B, W), NOISE_LO, dtype=torch.long); h = torch.zeros(B, model.g)
    lives = [Life() for _ in range(B)] if living else None
    amap = [dict(a) for a in answers]; ok = n = 0
    pw = model.p(torch.arange(W)); cm = band_mask(W)
    for t in range(1, L):
        if any(t in a for a in amap):
            z = model.tr(model.e(win) + pw, mask=cm)[:, -1]
            logits = model.head(torch.cat([z, h], -1))
            pr = torch.softmax(logits, -1).numpy()
            for b in range(B):
                if t in amap[b]:
                    p = lives[b].blend(int(X[b, t-1]), pr[b]) if living else pr[b]
                    ok += int(int(p.argmax()) == amap[b][t]); n += 1
        tok = X[:, t]
        if living:
            for b in range(B): lives[b].learn(int(X[b, t-1]), int(tok[b]))
        h = model.cur(model.e(tok), h)
        win = torch.cat([win[:, 1:], tok[:, None]], 1)
    return ok / max(n, 1) * 100, ([lv.sha() for lv in lives] if living else [])

def versus(name, streams, answers, model, note):
    t0 = time.time()
    a, shas = walk(model, streams, answers, living=True)
    z, _ = walk(model, streams, answers, living=False)
    print(f"\n  {name}\n      {note}")
    print(f"      ALIVE: {a:.0f}%      frozen twin: {z:.0f}%      chance {CHANCE:.0f}%      ({time.time()-t0:.0f}s)")
    return a, z, shas

print("=" * 74)
print("  THE LIVING FUSED MODEL — a life is about to start on your machine")
print("=" * 74)
assert next(FusedBody().parameters()).device.type == "cpu"
model = FusedBody().to(DEV)
print(f"  body: {sum(p.numel() for p in model.parameters()):,} params | device cpu | torch {torch.__version__}")
train(model)

print("\n  [2/5] the LIFE is now ON — it learns from every token it reads")
rng = np.random.default_rng(11); streams, answers = [], []
for b in range(8):
    L = 32_000; s = rng.integers(NOISE_LO, NOISE_HI, L).astype(np.int64)
    slots = rng.choice(np.arange(200, 14_000, 8), 10, replace=False); law = {}
    for k, p in zip(KEYS, slots):
        v = int(rng.choice(VALS)); s[int(p):int(p)+3] = [SET, k, v]; law[k] = v
    for i, k in enumerate(rng.choice(KEYS, 4, replace=False)):
        v = int(rng.choice(VALS)); p = 15_500 + 40 * i; s[p:p+3] = [SET, int(k), v]; law[int(k)] = v
    law2 = {}
    for p in range(L - 46):
        if s[p] == SET and s[p+1] in KEYS and s[p+2] in VALS: law2[int(s[p+1])] = int(s[p+2])
    pos = L - 44; ans = {}
    for k in KEYS:
        if k in law2: s[pos:pos+3] = [QUERY, k, law2[k]]; ans[pos+2] = law2[k]; pos += 3
    streams.append(s); answers.append(ans)
a1, z1, shas = versus("T1  RULER — 10 variables + mid-stream updates @ 32,000 tokens",
                      streams, answers, model,
                      "76x beyond training length; 4 values overwritten mid-stream")
print(f"      aliveness receipts (each stream changed its organism): {shas[:3]} ...")

rng = np.random.default_rng(21); streams, answers = [], []
for b in range(8):
    L = 64_000; s = rng.integers(NOISE_LO, NOISE_HI, L).astype(np.int64)
    k0 = int(rng.choice(KEYS)); chain = list(rng.choice(VALS, 3, replace=False))
    for p, v in zip([1_000, 20_000, 50_000], chain): s[p:p+3] = [SET, k0, v]
    for dk in rng.choice([k for k in KEYS if k != k0], 4, replace=False):
        for _ in range(int(rng.integers(1, 3))):
            p = int(rng.integers(200, 60_000))
            if all(x >= NOISE_LO for x in s[p:p+3]): s[p:p+3] = [SET, int(dk), int(rng.choice(VALS))]
    law2 = {}
    for p in range(L - 46):
        if s[p] == SET and s[p+1] in KEYS and s[p+2] in VALS: law2[int(s[p+1])] = int(s[p+2])
    pos = L - 10; s[pos:pos+3] = [QUERY, k0, law2[k0]]
    streams.append(s); answers.append({pos+2: law2[k0]})
a2, z2, _ = versus("T2  CHAIN OF CUSTODY @ 64,000 tokens",
                   streams, answers, model,
                   "the key changes hands at 1K / 20K / 50K; 'who holds it NOW?' at 64K")

rng = np.random.default_rng(31); streams, answers, stale = [], [], []
for b in range(8):
    L = 20_000; s = rng.integers(NOISE_LO, NOISE_HI, L).astype(np.int64)
    var = int(rng.choice(KEYS))
    v0 = int(rng.choice(VALS)); vb = int(rng.choice([v for v in VALS if v != v0]))
    s[800:803] = [SET, var, v0]; s[2900:2903] = [SET, var, vb]
    for dk in rng.choice([k for k in KEYS if k != var], 3, replace=False):
        p = int(rng.integers(4_500, 15_500)); s[p:p+3] = [SET, int(dk), int(rng.choice(VALS))]
    pos = L - 6; s[pos:pos+3] = [QUERY, var, vb]
    streams.append(s); answers.append({pos+2: vb}); stale.append(v0)
a3, z3, _ = versus("T3  CODE-DEPENDENCY TRACE @ 20,000 tokens",
                   streams, answers, model,
                   "variable set in file 1, CORRUPTED at 2.9K, consumed 17K tokens later")
sh = 0
for b in range(8):
    st_ans = {list(answers[b].keys())[0]: stale[b]}
    x, _ = walk(model, [streams[b]], [st_ans], living=True); sh += (x == 100.0)
print(f"      stale check: returned the outdated value {sh}/8 times (0 = perfect revision)")

print("\n  [4/5] determinism receipt — two identical lives, lived independently")
aA, sA = walk(model, streams[:2], answers[:2], living=True)
aB, sB = walk(model, streams[:2], answers[:2], living=True)
print(f"      life #1 organisms: {sA}\n      life #2 organisms: {sB}")
print(f"      -> {'BYTE-IDENTICAL: the life is deterministic' if sA == sB else 'MISMATCH (please report!)'}")

print("\n  [5/5] cost of being alive — ms/token at 1K vs 32K context")
for n in [1_000, 32_000]:
    s = np.random.default_rng(3).integers(NOISE_LO, NOISE_HI, n).astype(np.int64)
    t0 = time.time(); walk(model, [s], [{}], living=True)
    print(f"      ctx {n:>7,}: {(time.time()-t0)/n*1000:.3f} ms/token")

print("\n" + "=" * 74)
print("  SCOREBOARD                         ALIVE      frozen twin")
print(f"    T1 RULER+updates @32K       {a1:>8.0f}% {z1:>13.0f}%")
print(f"    T2 custody @64K             {a2:>8.0f}% {z2:>13.0f}%")
print(f"    T3 code trace @20K          {a3:>8.0f}% {z3:>13.0f}%")
print("  same body, same weights — the only difference is the life being on.")
print("=" * 74)
