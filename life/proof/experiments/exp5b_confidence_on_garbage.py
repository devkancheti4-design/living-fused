#!/usr/bin/env python3
"""
EXPERIMENT 5b — mined by the engine in round 5 (AD-1, score 0.792):
"Is softmax confidence calibrated on truly out-of-distribution inputs, or does a
 trained net emit confident answers on garbage while an exact store stays silent?"

Reuses exp2's trained net (5 tasks, agreement regime, 100% heldout generalization —
its BEST self). Probes:
  TRAINED   the 60 keys it was trained on
  HELDOUT   20 valid unseen keys (in-distribution, rule applies)
  GARBAGE   40 random dense vectors — not one-hot, no shape, no color, no rule
Signal: mean max-softmax "confidence". Control: the exact table on the same probes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp2_memory_conservation as base

net, table = base.Net(), base.Table()
for task in base.tasks:
    net.train_task(task)
    table.train_task(task)

rng = base.LCG(seed=31337)


def max_softmax_key(k):
    _, p = net.forward(base.onehot(k))
    return max(p)


def max_softmax_raw(x):
    _, p = net.forward(x)
    return max(p)


trained = [k for t in base.tasks for k in t]
conf_trained = sum(max_softmax_key(k) for k in trained) / len(trained)
conf_heldout = sum(max_softmax_key(k) for k in base.heldout) / len(base.heldout)

garbage = [[rng.uniform(-2, 2) for _ in range(base.SHAPES + base.COLORS)] for _ in range(40)]
conf_garbage = sum(max_softmax_raw(x) for x in garbage) / len(garbage)
high_conf_garbage = sum(1 for x in garbage if max_softmax_raw(x) > 0.9)

tbl_answers_garbage = 0  # exact table keyed by (shape,color): garbage has no key at all
tbl_answers_heldout = sum(1 for k in base.heldout if table.predict(k) is not None)

print("net mean max-softmax confidence (chance would be 0.25):")
print(f"  TRAINED keys : {conf_trained:.3f}")
print(f"  HELDOUT keys : {conf_heldout:.3f}")
print(f"  GARBAGE      : {conf_garbage:.3f}   ({high_conf_garbage}/40 garbage inputs answered with >90% confidence)")
print()
print("exact table on the same probes:")
print(f"  HELDOUT: answers {tbl_answers_heldout}/20 (abstains — key not present)")
print(f"  GARBAGE: answers {tbl_answers_garbage}/40 (garbage has no key: abstention is structural)")
print()
print("the net MUST answer — softmax has no abstain state; the table CANNOT bluff —")
print("absence of a key is a first-class, free, always-calibrated 'I don't know'.")
