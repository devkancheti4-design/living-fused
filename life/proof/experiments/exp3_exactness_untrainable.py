#!/usr/bin/env python3
"""
EXPERIMENT 3 — mined by the engine in round 3 (AB-2, score 0.810):
"Gradient descent trains attention only through softmax, whose gradient vanishes as
 attention sharpens toward one-hot: is exact lookup UNTRAINABLE by gradient?"

Setup — the BEST case for attention: key geometry is already perfect (query equals the
target key), values distinct, only the sharpness beta must be learned. If gradient
descent cannot reach exactness even here, added capacity (full QK matrices) inherits
the same softmax bottleneck.

  score_i = beta * (q . k_i),  alpha = softmax(score),  L = -log alpha_target
  Closed-form gradient: dL/dbeta = -(d_t - E_alpha[d])  -> 0 as alpha_target -> 1.

Part 1: training dynamics — does (1 - alpha_t) stall as a power law?
Part 2: confusable keys (cosine c to the query) — leaked attention mass after a fixed
        budget, and the beta required for 99.9% exactness, as c -> 1.
An exact-key dict is the control: 0 leak, 0 training, any c.
"""
import math

K, D = 8, 16


class LCG:
    def __init__(self, seed=987654321):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s >> 33

    def gauss(self):
        # sum of 12 uniforms - 6 ~ N(0,1), deterministic
        return sum(self.next() / float(1 << 31) for _ in range(12)) - 6.0


def normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def make_keys(rng, n):
    return [normalize([rng.gauss() for _ in range(D)]) for _ in range(n)]


def dots_for(q, keys):
    return [sum(a * b for a, b in zip(q, k)) for k in keys]


def alpha_target(beta, dots, t):
    m = max(beta * d for d in dots)
    e = [math.exp(beta * d - m) for d in dots]
    return e[t] / sum(e)


def grad_beta(beta, dots, t):
    m = max(beta * d for d in dots)
    e = [math.exp(beta * d - m) for d in dots]
    s = sum(e)
    ealpha = [x / s for x in e]
    exp_d = sum(a * d for a, d in zip(ealpha, dots))
    return -(dots[t] - exp_d)          # dL/dbeta


# ---------- Part 1: dynamics on near-orthogonal keys ----------
rng = LCG()
keys = make_keys(rng, K)
q, t = keys[0], 0                      # perfect geometry: query IS the key
dots = dots_for(q, keys)

beta, lr = 1.0, 1.0
print("=== PART 1: training dynamics (perfect geometry, only sharpness learned) ===")
print(f"{'step':>7} {'beta':>9} {'1-alpha_t':>12} {'|grad|':>12}")
checkpoints = {1, 10, 100, 1000, 10000, 100000}
gap_at = {}
for step in range(1, 100001):
    g = grad_beta(beta, dots, t)
    beta -= lr * g
    if step in checkpoints:
        a = alpha_target(beta, dots, t)
        gap_at[step] = 1 - a
        print(f"{step:>7} {beta:>9.3f} {1 - a:>12.3e} {abs(g):>12.3e}")

r1 = gap_at[1000] / gap_at[10000]
r2 = gap_at[10000] / gap_at[100000]
print(f"\ngap shrink per 10x steps: x{r1:.1f} then x{r2:.1f}"
      f"  (power-law stall: each digit of exactness costs ~10x more steps)")

# ---------- Part 2: confusable keys ----------
print("\n=== PART 2: near-duplicate keys, fixed budget 5000 steps, avg of 20 sets ===")
print(f"{'cos(c)':>7} {'leaked mass':>12} {'beta needed for 99.9%':>22} {'dict leak':>10}")
for c in (0.5, 0.8, 0.9, 0.95, 0.99):
    leaks = []
    for trial in range(20):
        rng2 = LCG(seed=1000 + trial)
        base = make_keys(rng2, K)
        qv, tt = base[0], 0
        conf = []
        for j, k in enumerate(base):
            if j == tt:
                conf.append(k)
            else:
                orth = normalize([kj - c * qj for kj, qj in zip(k, qv)])
                conf.append(normalize([c * qj + math.sqrt(1 - c * c) * oj
                                       for qj, oj in zip(qv, orth)]))
        dd = dots_for(qv, conf)
        b = 1.0
        for _ in range(5000):
            b -= 1.0 * grad_beta(b, dd, tt)
        leaks.append(1 - alpha_target(b, dd, tt))
    beta_star = math.log(999 * (K - 1)) / (1 - c)   # analytic requirement
    print(f"{c:>7.2f} {sum(leaks) / len(leaks):>12.3e} {beta_star:>22.1f} {'0.0':>10}")

print("\nreading: leaked mass = wrong-value contamination per lookup; the dict column")
print("is the exact-key store — zero leak, zero training, at every confusability.")
