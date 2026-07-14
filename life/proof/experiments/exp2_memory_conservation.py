#!/usr/bin/env python3
"""
EXPERIMENT 2 — mined by the engine in round 1 (C-4 + B-2 triples):
C-4: "Is there a conservation law of memory: any store that generalizes must interfere,
      and any store that never interferes cannot generalize?"
B-2: "Is catastrophic interference the destruction caused by averaging shared parameters?"

Design (pure Python, deterministic, no deps):
  Facts are (shape, color) -> class, where class = shape_id % 4  (a latent RULE).
  5 sequential tasks, 12 facts each, disjoint keys. After each task, test:
    - RETENTION: accuracy on task 1's facts (seen once, never revisited)
    - GENERALIZATION: accuracy on 20 held-out unseen (shape, color) keys obeying the rule
  Learner A: tiny 2-layer softmax net, shared params, online SGD (the gradient paradigm)
  Learner B: exact count table keyed by (shape, color)          (the Life paradigm)

  Conservation-law prediction: A generalizes but forgets; B never forgets, never generalizes.
"""
import math

SHAPES, COLORS, CLASSES = 10, 8, 4
HID, LR, EPOCHS = 24, 0.5, 40


# --- deterministic pseudo-random (LCG), no random module ---
class LCG:
    def __init__(self, seed=1234567):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s >> 33

    def uniform(self, lo, hi):
        return lo + (self.next() / float(1 << 31)) * (hi - lo)

    def shuffle(self, xs):
        for i in range(len(xs) - 1, 0, -1):
            j = self.next() % (i + 1)
            xs[i], xs[j] = xs[j], xs[i]


rng = LCG()

# --- build tasks: 5 tasks x 12 disjoint keys; rule: class = shape % 4 ---
all_keys = [(s, c) for s in range(SHAPES) for c in range(COLORS)]
rng.shuffle(all_keys)
tasks = [all_keys[i * 12:(i + 1) * 12] for i in range(5)]
heldout = all_keys[60:80]  # never trained by either learner
label = lambda k: k[0] % 4


def onehot(k):
    v = [0.0] * (SHAPES + COLORS)
    v[k[0]] = 1.0
    v[SHAPES + k[1]] = 1.0
    return v


# --- Learner A: 2-layer net, manual backprop ---
class Net:
    def __init__(self):
        d = SHAPES + COLORS
        self.W1 = [[rng.uniform(-0.3, 0.3) for _ in range(d)] for _ in range(HID)]
        self.b1 = [0.0] * HID
        self.W2 = [[rng.uniform(-0.3, 0.3) for _ in range(HID)] for _ in range(CLASSES)]
        self.b2 = [0.0] * CLASSES

    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        z = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return h, [v / s for v in e]

    def train_task(self, keys):
        for _ in range(EPOCHS):
            for k in keys:
                x, y = onehot(k), label(k)
                h, p = self.forward(x)
                dz = [(p[i] - (1.0 if i == y else 0.0)) for i in range(CLASSES)]
                dh = [sum(self.W2[i][j] * dz[i] for i in range(CLASSES)) * (1.0 if h[j] > 0 else 0.0)
                      for j in range(HID)]
                for i in range(CLASSES):
                    for j in range(HID):
                        self.W2[i][j] -= LR * dz[i] * h[j]
                    self.b2[i] -= LR * dz[i]
                for j in range(HID):
                    for i2 in range(SHAPES + COLORS):
                        self.W1[j][i2] -= LR * dh[j] * x[i2]
                    self.b1[j] -= LR * dh[j]

    def predict(self, k):
        _, p = self.forward(onehot(k))
        return max(range(CLASSES), key=lambda i: p[i])


# --- Learner B: exact count table ---
class Table:
    def __init__(self):
        self.t = {}

    def train_task(self, keys):
        for k in keys:
            self.t[k] = label(k)

    def predict(self, k):
        return self.t.get(k, None)  # abstains on unseen


def acc(learner, keys):
    hits = sum(1 for k in keys if learner.predict(k) == label(k))
    return hits / len(keys) * 100


net, table = Net(), Table()
print(f"{'after task':>10} | {'NET t1-retention':>16} {'NET heldout-gen':>15} | "
      f"{'TABLE t1-retention':>18} {'TABLE heldout-gen':>17}")
for i, task in enumerate(tasks):
    net.train_task(task)
    table.train_task(task)
    print(f"{i + 1:>10} | {acc(net, tasks[0]):>15.1f}% {acc(net, heldout):>14.1f}% | "
          f"{acc(table, tasks[0]):>17.1f}% {acc(table, heldout):>16.1f}%")

print("\nconservation-law check:")
print(f"  net    : generalizes (heldout {acc(net, heldout):.0f}%) but task-1 retention fell to {acc(net, tasks[0]):.0f}%")
print(f"  table  : task-1 retention {acc(table, tasks[0]):.0f}%, heldout {acc(table, heldout):.0f}% (abstains — no rule, no guess)")
