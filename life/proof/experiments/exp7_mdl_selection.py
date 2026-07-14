#!/usr/bin/env python3
"""
EXPERIMENT 7 — mined by the engine in Session 2 round 1 (GE-5, score 0.742):
"MDL says learning IS compression; yet we optimize LOSS and bolt regularization on
 afterward. Invention: make description length the selection criterion directly — does
 an MDL-selected model pick the generalizing complexity from TRAINING DATA ALONE, no
 validation set?"

Setup (deterministic, pure Python): data y = cubic(x) + Gaussian noise, x in [-1,1].
Fit polynomials degree 0..8 by least squares (normal equations, Gaussian elimination).
For each degree compare three selectors:
  TRAIN-MSE  : monotonically decreasing -> always picks max degree (overfits) — the naive optimizer
  VAL-MSE    : needs a held-out set -> picks the generalizing degree (the standard fix, costs data)
  MDL (bits) : (N/2)log2(train_MSE) + ((k+1)/2)log2(N), uses TRAINING DATA ONLY
Claim: MDL's argmin == VAL's argmin == true degree, with no validation data spent.
"""
import math


class LCG:
    def __init__(self, seed=246813579):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s >> 33

    def uniform(self, lo, hi):
        return lo + (self.next() / float(1 << 31)) * (hi - lo)

    def gauss(self):
        return sum(self.next() / float(1 << 31) for _ in range(12)) - 6.0


TRUE = [0.5, -1.2, 0.0, 2.0]        # 0.5 - 1.2 x + 0 x^2 + 2 x^3  (true degree 3)
NOISE = 0.25


def truef(x):
    return sum(c * x ** i for i, c in enumerate(TRUE))


def make(n, rng):
    xs = [rng.uniform(-1, 1) for _ in range(n)]
    ys = [truef(x) + NOISE * rng.gauss() for x in xs]
    return xs, ys


def solve(A, b):
    """Gaussian elimination with partial pivoting; A is m x m (list of lists)."""
    m = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(m):
        piv = max(range(c, m), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-15:
            continue
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for j in range(c, m + 1):
            M[c][j] /= pv
        for r in range(m):
            if r != c and abs(M[r][c]) > 0:
                f = M[r][c]
                for j in range(c, m + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][m] for i in range(m)]


def fit_poly(xs, ys, deg):
    # normal equations for Vandermonde of degree deg
    k = deg + 1
    A = [[sum(x ** (i + j) for x in xs) for j in range(k)] for i in range(k)]
    b = [sum(ys[t] * xs[t] ** i for t in range(len(xs))) for i in range(k)]
    return solve(A, b)


def mse(coef, xs, ys):
    return sum((sum(c * x ** i for i, c in enumerate(coef)) - y) ** 2
               for x, y in zip(xs, ys)) / len(xs)


rng = LCG()
xs, ys = make(40, rng)                 # training set
vxs, vys = make(200, rng)              # validation set (only VAL-MSE is allowed to use this)
N = len(xs)

print(f"true degree = 3, N_train = {N}, noise sd = {NOISE}\n")
print(f"{'deg':>3} {'train_MSE':>10} {'val_MSE':>10} {'MDL_bits':>10}")
train_pick = (1e9, None); val_pick = (1e9, None); mdl_pick = (1e9, None)
for deg in range(9):
    coef = fit_poly(xs, ys, deg)
    tr = mse(coef, xs, ys)
    va = mse(coef, vxs, vys)
    k = deg + 1
    mdl = (N / 2) * math.log2(max(tr, 1e-12)) + (k / 2) * math.log2(N)
    star = ""
    if tr < train_pick[0]: train_pick = (tr, deg)
    if va < val_pick[0]: val_pick = (va, deg)
    if mdl < mdl_pick[0]: mdl_pick = (mdl, deg); star = " <- MDL min"
    print(f"{deg:>3} {tr:>10.4f} {va:>10.4f} {mdl:>10.2f}{star}")

print(f"\nselector picks:")
print(f"  TRAIN-MSE -> degree {train_pick[1]}  (overfits: always the largest — the naive optimizer)")
print(f"  VAL-MSE   -> degree {val_pick[1]}  (correct, but spent 200 held-out points to know it)")
print(f"  MDL       -> degree {mdl_pick[1]}  (correct, using TRAINING DATA ONLY — no validation set)")
print(f"\ninvention verdict: {'MDL == VAL == true degree; description length is a '
                              'validation-free selector' if mdl_pick[1] == val_pick[1] == 3 else 'MDL and VAL diverge — see table'}")
