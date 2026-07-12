# apps — two working demos of the fused memory

Two small, honest, self-contained programs that show what the flat integer memory does on a real
laptop. Both **adapt to your machine** (detect RAM, suggest a model tier) and both have a **core
that runs with nothing but Python** — no GPU, no model download required to see the point.

## `researcher_bench.py` — context-memory decoupling, run locally

Reproduces the cost/scaling numbers on *your* machine: the KV-cache growth table (O(N)) vs the flat
integer store (O(1)), a flat-memory proof (stream millions of tokens — the store does not grow), and
lookup latency. No model needed.

```bash
python3 apps/researcher_bench.py
```

**Shows:** long-context *memory* is decoupled from VRAM — a million-token memory fits in flat KB on a
standard computer, no multi-GPU grant. **Does not show:** cheaper generation — that runs at
transformer speed regardless. This fixes the context-*memory* cost, not inference cost.

## `personal_brain.py` — a private, on-device fact memory

Tell it things, ask in plain language. Everything is local and persistent (saved to
`~/.personal_brain.json`). Uses a semantic embedder if `transformers` is installed (handles reworded
questions), otherwise keyword matching; uses a local model to phrase answers if one is available,
otherwise it returns the matching facts.

```bash
python3 apps/personal_brain.py remember "the project deadline is March 15"
python3 apps/personal_brain.py ask "when is my project due?"
python3 apps/personal_brain.py            # interactive
```

Example (semantic recall on deliberately reworded queries):

```
"what happened to that property paperwork after Artie got it?" -> "Arthur gave the asset deed to Beatrice."
"why did the sign-in break this morning?"                      -> "The login gateway failed ... null pointer."
```

**Honest scope:** the memory is exact, persistent, and flat (it grows with #facts, not with how long
you use it). Semantic recall is a float embedder — strong on rewording but with real holes (negation,
rare synonyms, thin margins when many facts compete). Any reasoning/phrasing is the optional model's
job at the model's quality; the memory layer adds **recall, not intelligence**.

## Adapting to your laptop and model

Both scripts detect RAM and suggest a model tier (~0.5B → ~14B by available memory). The memory is
**model-agnostic** — swap in any model your machine can run; `researcher_bench.py` lists several
config presets, and you can add your own `(layers, kv_heads, head_dim, bytes)`. The core memory +
retrieval runs even with **no model at all**.
