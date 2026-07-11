# Infinite-stream soak — the living memory doesn't leak or drift

A common objection to "it's flat at 32K/64K tokens" is: *sure, but over a long enough
stream it'll drift, blow up, or forget.* This is the measured answer.

We ran the **real `Life` memory organ** — the exact class in [`live.py`](live.py), loaded
from source, not a reimplementation — on a never-ending token stream, checkpointing every
~40 seconds.

## Result — single continuous run (Apple M4 Pro, CPU, one core)

| metric | value |
|---|---|
| duration | **10h 55m** continuous (no sleep, no restart) |
| tokens processed | **19,940,000,000** (~20 billion) |
| process RSS | **223,584 KB — a single value for the entire run** (997 checkpoints) |
| memory growth | **0 KB** — RSS never changed once |
| throughput | 504,810 → 498,260 tok/s (flat, ~1.3% drift) |
| passkey recall | **10/10 at every one of 997 checkpoints**, 0 dips (100% *by construction* — disjoint rows; see scope) |
| write-guard (full-table) | HELD at every checkpoint |
| recency-dominant revision test | **0 failures** |
| organ table footprint | 12,800 bytes, fixed by construction |

Every ~40s the run re-verifies: process memory, throughput, that all 10 stored passkeys
still recall, that a query cannot write the store (a full-table guard check), and — every
2B tokens — that a later write cleanly overrules an older one (recency-dominant revision).

We stopped it at ~11h / ~20B tokens; nothing was trending, so more wall-clock adds no new
information for this component.

## Why the memory is flat *by construction*

The `Life` organ is a fixed `V×V` integer table (here 40×40 int64 = 12,800 bytes),
pre-allocated once. Processing more tokens writes into existing cells — it never allocates.
So "no leak" isn't luck; the data structure cannot grow. The ~223 MB process baseline is the
Python / NumPy / PyTorch interpreter, constant and excluded from the growth figure. The one
numeric risk over billions of writes — integer overflow — is handled by a deterministic
cap-and-rescale, which the 0 revision failures confirm stayed stable.

## Honest scope — read before citing

- **This soaks the memory organ, not the end-to-end fused model.** It measures the living
  memory's stability (no leak, flat compute, stable recall, numeric stability over ~20B
  writes). The fused transformer path is [`live.py`](live.py); a continuous *fused-model*
  soak is a separate experiment.
- **Recall is 100% by construction, not a hard-won result.** The 10 passkeys occupy disjoint
  table rows (keys 0–9); the filler stream occupies other rows (noise 28–39), so filler
  *architecturally cannot* touch a passkey row. This validates the disjoint-row
  zero-interference design — it does **not** claim recall under adversarial same-row
  contention.
- **The stream is synthetic** (uniform noise over a 40-symbol alphabet), not real text.
- Numbers are one CPU core on an M4 Pro; your throughput will differ, but the
  memory-flatness, recall, and determinism are structural, not hardware-dependent.

## Reproduce

```bash
python3 soak.py 3600      # run for 1 hour (arg = seconds); prints a checkpoint every ~40s
```

`soak.py` loads the real `Life` class from `live.py` and streams tokens through it, logging
memory, throughput, passkey recall, the full-table write-guard, and a two-lives determinism
check.
