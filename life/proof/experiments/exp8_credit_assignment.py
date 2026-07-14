#!/usr/bin/env python3
"""
EXPERIMENT 8 — mined by the engine in Session 2 round 2 (E-2, score 0.709):
"Is backprop the unique efficient credit-assignment algorithm, or merely the
 differentiable one — what is the minimum signal credit assignment needs, and
 what does each relaxation cost?"

Four learners, identical net (one-hot(18) -> ReLU hidden -> 4-way softmax),
identical FORWARD-PASS budget (fair compute), same data, same deterministic noise:

  BP  exact backprop                      (full structural signal: W2^T)
  FA  feedback alignment                  (hidden feedback through a FIXED RANDOM matrix
                                           — no weight symmetry, Lillicrap et al. 2016)
  NP  node perturbation                   (no feedback path at all: jitter each hidden
                                           unit, read the loss change — per-UNIT signal)
  WP  weight perturbation (antithetic ES) (one global scalar per trial — per-TRIAL signal)

Task: class = shape % 4 over (shape,color) one-hot keys; 48 train / 16 heldout.
Run at hidden = 16 and hidden = 48 to expose how each rule scales with dimension.
Prediction (conservation sniff): the per-parameter signal must come from somewhere —
structure (BP/FA) or sampling variance (NP/WP, paying with dimension).
"""
import math

SHAPES, COLORS, NCLS = 10, 8, 4
DIN = SHAPES + COLORS
BUDGET = 60000  # forward passes, all methods


class LCG:
    def __init__(self, seed=777):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s >> 33

    def uniform(self, lo, hi):
        return lo + (self.next() / float(1 << 31)) * (hi - lo)

    def gauss(self):
        return sum(self.next() / float(1 << 31) for _ in range(12)) - 6.0

    def shuffle(self, xs):
        for i in range(len(xs) - 1, 0, -1):
            j = self.next() % (i + 1)
            xs[i], xs[j] = xs[j], xs[i]


def make_data():
    rng = LCG(4242)
    keys = [(s, c) for s in range(SHAPES) for c in range(COLORS)]
    rng.shuffle(keys)
    return keys[:48], keys[48:64]


def onehot(k):
    v = [0.0] * DIN
    v[k[0]] = 1.0
    v[SHAPES + k[1]] = 1.0
    return v


label = lambda k: k[0] % 4


class Net:
    def __init__(self, hid, rng):
        self.h = hid
        self.W1 = [[rng.uniform(-0.3, 0.3) for _ in range(DIN)] for _ in range(hid)]
        self.b1 = [0.0] * hid
        self.W2 = [[rng.uniform(-0.3, 0.3) for _ in range(hid)] for _ in range(NCLS)]
        self.b2 = [0.0] * NCLS

    def forward(self, x, hnoise=None):
        pre = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(self.W1, self.b1)]
        if hnoise:
            pre = [p + n for p, n in zip(pre, hnoise)]
        h = [max(0.0, p) for p in pre]
        z = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return pre, h, [v / s for v in e]

    def loss(self, p, y):
        return -math.log(max(p[y], 1e-12))


def acc(net, keys):
    return sum(1 for k in keys if max(range(NCLS), key=lambda i: net.forward(onehot(k))[2][i]) == label(k)) / len(keys) * 100


def train(method, hid, lr, sigma=0.05):
    rng = LCG(1000 + hid)
    net = Net(hid, rng)
    B = [[rng.uniform(-0.3, 0.3) for _ in range(NCLS)] for _ in range(hid)]  # FA feedback
    train_keys, held = make_data()
    fwd = 0
    i = 0
    order = list(train_keys)
    while fwd < BUDGET:
        k = order[i % len(order)]
        i += 1
        x, y = onehot(k), label(k)
        if method in ("BP", "FA"):
            pre, h, p = net.forward(x)
            fwd += 2  # count backward as a forward-equivalent
            dz = [(p[c] - (1.0 if c == y else 0.0)) for c in range(NCLS)]
            if method == "BP":
                dh = [sum(net.W2[c][j] * dz[c] for c in range(NCLS)) for j in range(hid)]
            else:
                dh = [sum(B[j][c] * dz[c] for c in range(NCLS)) for j in range(hid)]
            dh = [d if pre[j] > 0 else 0.0 for j, d in enumerate(dh)]
            for c in range(NCLS):
                for j in range(hid):
                    net.W2[c][j] -= lr * dz[c] * h[j]
                net.b2[c] -= lr * dz[c]
            for j in range(hid):
                if dh[j] == 0.0:
                    continue
                for a in range(DIN):
                    if x[a] != 0.0:
                        net.W1[j][a] -= lr * dh[j] * x[a]
                net.b1[j] -= lr * dh[j]
        elif method == "NP":
            _, h, p = net.forward(x)
            L0 = net.loss(p, y)
            noise = [sigma * rng.gauss() for _ in range(hid)]
            _, _, p1 = net.forward(x, hnoise=noise)
            L1 = net.loss(p1, y)
            fwd += 2
            scale = (L1 - L0) / (sigma * sigma)
            # hidden units' incoming weights via perturbation signal
            for j in range(hid):
                g = scale * noise[j]
                for a in range(DIN):
                    if x[a] != 0.0:
                        net.W1[j][a] -= lr * g * x[a]
                net.b1[j] -= lr * g
            # output layer gets the exact local delta rule (observable layer)
            dz = [(p[c] - (1.0 if c == y else 0.0)) for c in range(NCLS)]
            for c in range(NCLS):
                for j in range(hid):
                    net.W2[c][j] -= lr * dz[c] * h[j]
                net.b2[c] -= lr * dz[c]
        elif method == "WP":
            # antithetic weight perturbation on ALL parameters
            eps1 = [[sigma * rng.gauss() for _ in range(DIN)] for _ in range(hid)]
            eps2 = [[sigma * rng.gauss() for _ in range(hid)] for _ in range(NCLS)]
            for j in range(hid):
                for a in range(DIN):
                    net.W1[j][a] += eps1[j][a]
            for c in range(NCLS):
                for j in range(hid):
                    net.W2[c][j] += eps2[c][j]
            _, _, pp = net.forward(x)
            Lp = net.loss(pp, y)
            for j in range(hid):
                for a in range(DIN):
                    net.W1[j][a] -= 2 * eps1[j][a]
            for c in range(NCLS):
                for j in range(hid):
                    net.W2[c][j] -= 2 * eps2[c][j]
            _, _, pm = net.forward(x)
            Lm = net.loss(pm, y)
            fwd += 2
            g = (Lp - Lm) / (2 * sigma * sigma)
            for j in range(hid):
                for a in range(DIN):
                    net.W1[j][a] += eps1[j][a] - lr * g * eps1[j][a]
            for c in range(NCLS):
                for j in range(hid):
                    net.W2[c][j] += eps2[c][j] - lr * g * eps2[c][j]
    train_keys, held = make_data()
    return acc(net, train_keys), acc(net, held)


print(f"budget = {BUDGET} forward passes for every method (fair compute)\n")
print(f"{'method':>7} {'signal':>28} | {'hid=16 train/held':>18} | {'hid=48 train/held':>18}")
LRS = {"BP": 0.5, "FA": 0.5, "NP": 0.05, "WP": 0.02}
SIGNAL = {"BP": "exact per-weight (W2^T)", "FA": "random-matrix per-unit", "NP": "loss-delta per-unit", "WP": "loss-delta per-TRIAL"}
for m in ("BP", "FA", "NP", "WP"):
    t16, h16 = train(m, 16, LRS[m])
    t48, h48 = train(m, 48, LRS[m])
    print(f"{m:>7} {SIGNAL[m]:>28} | {t16:>7.1f}% / {h16:<6.1f}% | {t48:>7.1f}% / {h48:<6.1f}%")

print("\nreading: params hid16 ~350, hid48 ~1050. Structural signals (BP/FA) are")
print("width-indifferent; sampled signals pay variance that grows with what one")
print("scalar must explain — the credit signal is conserved: compute it or sample it.")
