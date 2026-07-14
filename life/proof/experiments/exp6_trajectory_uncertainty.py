#!/usr/bin/env python3
"""
EXPERIMENT 6 — mined by the engine in Session 2 round 1 (FE-5, score 0.790):
"SGD discards the cloud of near-equal models it walked through, but that cloud IS an
 unnormalized posterior. Invention: read uncertainty/abstention from the SPREAD of the
 optimization trajectory instead of from softmax magnitude."

This directly attacks Session-1 exp5b, where a single net was 97.4% confident on GARBAGE.
The Bayesian-optimization cross (F x E) says: don't trust one point; look at the ensemble.

Two spreads, both free from optimization:
  TRAJECTORY = late checkpoints of ONE SGD run (cheapest — one training run)
  MULTI-INIT = a handful of runs from different seeds (deep-ensemble, more independent)
Signal = predictive DISAGREEMENT (entropy of the mean prediction + vote split) on:
  TRAINED / HELDOUT (in-distribution) vs GARBAGE (off-manifold).
Abstain when disagreement is high. Compare against exp5b's single-softmax confidence.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp2_memory_conservation as base

NCLS = base.CLASSES
DIN = base.SHAPES + base.COLORS


def train_net(seed, snapshots):
    rng_saved = base.rng
    base.rng = base.LCG(seed=seed)
    net = base.Net()
    base.rng = rng_saved
    snaps = []
    order = [k for t in base.tasks for k in t]
    step = 0
    total = base.EPOCHS * len(order)
    for _ in range(base.EPOCHS):
        for k in order:
            x, y = base.onehot(k), base.label(k)
            h, p = net.forward(x)
            dz = [(p[i] - (1.0 if i == y else 0.0)) for i in range(NCLS)]
            dh = [sum(net.W2[i][j] * dz[i] for i in range(NCLS)) * (1.0 if h[j] > 0 else 0.0)
                  for j in range(base.HID)]
            for i in range(NCLS):
                for j in range(base.HID):
                    net.W2[i][j] -= base.LR * dz[i] * h[j]
                net.b2[i] -= base.LR * dz[i]
            for j in range(base.HID):
                for i2 in range(DIN):
                    net.W1[j][i2] -= base.LR * dh[j] * x[i2]
                net.b1[j] -= base.LR * dh[j]
            step += 1
            if step in snapshots:
                snaps.append(clone(net))
    snaps.append(clone(net))
    return net, snaps


def clone(net):
    import copy
    n = base.Net.__new__(base.Net)
    n.W1 = copy.deepcopy(net.W1); n.b1 = list(net.b1)
    n.W2 = copy.deepcopy(net.W2); n.b2 = list(net.b2)
    return n


def probs_raw(net, x):
    _, p = net.forward(x)
    return p


def ensemble_stats(nets, x):
    ps = [probs_raw(n, x) for n in nets]
    mean = [sum(p[i] for p in ps) / len(ps) for i in range(NCLS)]
    ent = -sum(m * math.log2(max(m, 1e-12)) for m in mean)      # predictive entropy (bits)
    votes = [max(range(NCLS), key=lambda i: p[i]) for p in ps]
    disagree = 1.0 - max(votes.count(v) for v in set(votes)) / len(votes)  # vote split
    return ent, disagree


# --- build one trajectory ensemble and one multi-init ensemble ---
total_steps = base.EPOCHS * 60
snap_steps = {int(total_steps * f) for f in (0.6, 0.7, 0.8, 0.9)}
main_net, traj_snaps = train_net(42, snap_steps)

init_nets = [train_net(1000 + s, set())[0] for s in range(8)]

# --- probes ---
trained = [k for t in base.tasks for k in t]
heldout = base.heldout
grng = base.LCG(seed=31337)
garbage = [[grng.uniform(-2, 2) for _ in range(DIN)] for _ in range(40)]


def mean_over(fn, items, to_x):
    vals = [fn(to_x(it)) for it in items]
    return sum(v for v, _ in vals) / len(vals), sum(d for _, d in vals) / len(vals)


def single_conf(x):
    return max(probs_raw(main_net, x))


print("SINGLE MODEL max-softmax confidence (the exp5b failure, reproduced):")
print(f"  trained {sum(single_conf(base.onehot(k)) for k in trained) / len(trained):.3f}"
      f"   heldout {sum(single_conf(base.onehot(k)) for k in heldout) / len(heldout):.3f}"
      f"   GARBAGE {sum(single_conf(x) for x in garbage) / len(garbage):.3f}   <- confident on garbage\n")

for label_e, nets in [("TRAJECTORY (1 run, 5 checkpoints)", traj_snaps),
                       ("MULTI-INIT (8 runs)", init_nets)]:
    et, dt = mean_over(lambda x: ensemble_stats(nets, x), trained, base.onehot)
    eh, dh_ = mean_over(lambda x: ensemble_stats(nets, x), heldout, base.onehot)
    eg, dg = mean_over(lambda x: ensemble_stats(nets, x), garbage, lambda x: x)
    print(f"{label_e}")
    print(f"  predictive entropy (bits):  trained {et:.3f}  heldout {eh:.3f}  GARBAGE {eg:.3f}")
    print(f"  vote disagreement (0..1) :  trained {dt:.3f}  heldout {dh_:.3f}  GARBAGE {dg:.3f}")
    sep = eg / max(eh, 1e-9)
    # abstention: threshold at midpoint of heldout and garbage entropy
    thr = (eh + eg) / 2
    caught = sum(1 for x in garbage if ensemble_stats(nets, x)[0] > thr)
    kept = sum(1 for k in heldout if ensemble_stats(nets, base.onehot(k))[0] <= thr)
    print(f"  entropy garbage/heldout ratio: {sep:.2f}x   "
          f"abstain@thr caught {caught}/40 garbage, kept {kept}/20 valid\n")

print("verdict: softmax magnitude cannot flag garbage (exp5b); trajectory/ensemble")
print("DISAGREEMENT can — uncertainty read from optimization spread, not readout height.")
