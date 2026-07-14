#!/usr/bin/env python3
"""
EXPERIMENT 13 — tests Finding 3 (N-2): "a closed self-improvement loop caps at its
frozen judge's blind spots; reality-grounded verification escapes the cap."

Not exp9 restated: exp9 measured the STATIC geometry (cosine law); this runs the
LOOP DYNAMICS — a generator that iteratively self-improves against three judges:

  FROZEN   judge: linear proxy fit once to 15 noisy samples of true quality
           (rank-deficient + curvature-blind — the RLAIF/constitution template)
  COEVOLVE judge: same proxy REFIT every 100 steps on fresh samples near the
           current solution (the co-evolving-judge proposal)
  REALITY  judge: the true objective itself (executable verification)

True quality V(x) = -||x - x*||^2, hidden target x* in R^8. Hill-climb 600 steps,
accept a perturbation iff the JUDGE says it improved. Track TRUE quality.
Prediction: FROZEN peaks then falls (Goodhart in motion); COEVOLVE partial;
REALITY monotone. Deterministic (LCG), stdlib only.
"""
import math

D, STEPS, REFIT = 8, 600, 100


class LCG:
    def __init__(self, seed=246810):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s >> 33

    def gauss(self):
        return sum(self.next() / float(1 << 31) for _ in range(12)) - 6.0


def true_V(x, xstar):
    return -sum((a - b) ** 2 for a, b in zip(x, xstar))


def fit_linear_judge(rng, xstar, center, spread):
    """Fit judge(x) = a.x + b to 15 noisy samples of V near `center` (least squares
    via simple gradient fit — enough for a linear model)."""
    pts = [[c + spread * rng.gauss() for c in center] for _ in range(15)]
    ys = [true_V(p, xstar) + 2.0 * rng.gauss() for p in pts]
    a, b = [0.0] * D, 0.0
    for _ in range(3000):
        ga, gb = [0.0] * D, 0.0
        for p, y in zip(pts, ys):
            e = (sum(ai * pi for ai, pi in zip(a, p)) + b) - y
            for i in range(D):
                ga[i] += e * p[i]
            gb += e
        for i in range(D):
            a[i] -= 0.0005 / len(pts) * ga[i]
        b -= 0.0005 / len(pts) * gb
    return a, b


def run(judge_kind):
    rng = LCG(seed=99)
    xstar = [rng.gauss() for _ in range(D)]
    x = [0.0] * D
    a, b = fit_linear_judge(rng, xstar, x, 2.0)

    def judge(p):
        if judge_kind == "REALITY":
            return true_V(p, xstar)
        return sum(ai * pi for ai, pi in zip(a, p)) + b

    best_true, traj = true_V(x, xstar), []
    for step in range(1, STEPS + 1):
        cand = [xi + 0.3 * rng.gauss() for xi in x]
        if judge(cand) > judge(x):
            x = cand
        tv = true_V(x, xstar)
        best_true = max(best_true, tv)
        if step % 100 == 0:
            traj.append(tv)
        if judge_kind == "COEVOLVE" and step % REFIT == 0:
            a, b = fit_linear_judge(rng, xstar, x, 1.0)
    return traj, best_true, true_V(x, xstar)


print(f"true quality V(x) = -||x - x*||^2 (0 = perfect); start V = -{sum(1 for _ in range(1)) and ''}{'':s}")
print(f"{'judge':>9} | true V at steps 100..600 {'':>24} | peak | final")
for kind in ("FROZEN", "COEVOLVE", "REALITY"):
    traj, peak, final = run(kind)
    tr = " ".join(f"{v:8.2f}" for v in traj)
    print(f"{kind:>9} | {tr} | {peak:6.2f} | {final:7.2f}")
print("\nreading: the generator and its proposal distribution are IDENTICAL in all")
print("three runs — only the judge differs. Whatever separates the rows is the judge.")
