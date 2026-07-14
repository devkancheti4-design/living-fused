#!/usr/bin/env python3
"""
EXPERIMENT 2b — the control exp2 demands:
exp2 used ONE latent rule shared by all tasks -> net retention recovered (agreement regime).
Here the rule CHANGES each task: task t labels its keys class = (shape + t) % 4.
Keys stay disjoint, so the count table is untouched; but the net's shared feature->class
weights receive directly conflicting gradients across tasks (the B-2 averaging-conflict regime).

Prediction (conservation law, corrected form):
  gradient memory forgets exactly what the stream DISAGREES about;
  table memory does not care whether the stream agrees.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp2_memory_conservation as base  # reuse Net/Table/LCG/keys — deterministic


def label_for_task(k, t):
    return (k[0] + t) % 4


# rebuild learners fresh
net, table = base.Net(), base.Table()
tasks = base.tasks

# monkey-patch-free training: retrain with per-task labels
def train_net(net, keys, t):
    saved = base.label
    base.label = lambda k: label_for_task(k, t)
    net.train_task(keys)
    base.label = saved


def train_table(table, keys, t):
    for k in keys:
        table.t[k] = label_for_task(k, t)


def acc_task1(learner):
    hits = sum(1 for k in tasks[0] if learner.predict(k) == label_for_task(k, 0))
    return hits / len(tasks[0]) * 100


print(f"{'after task':>10} | {'NET task1-retention':>19} | {'TABLE task1-retention':>21}")
for t, task in enumerate(tasks):
    train_net(net, task, t)
    train_table(table, task, t)
    print(f"{t + 1:>10} | {acc_task1(net):>18.1f}% | {acc_task1(table):>20.1f}%")

print("\nconflict-regime verdict:")
print(f"  net   : task-1 retention {acc_task1(net):.0f}% (chance = 25%)")
print(f"  table : task-1 retention {acc_task1(table):.0f}%")
