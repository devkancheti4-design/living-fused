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

## `personal_brain.py` — a private, on-device conversational memory

Just talk to it. It figures out whether you're **telling** it something (it remembers) or **asking**
(it answers from what it knows), holds the **last few turns** so follow-ups like *"and where is
that?"* work, and replies in a warm, concise voice. Everything is local and persistent (saved to
`~/.personal_brain.json`). `forget <text>` / `forget all` to manage; `AUTO_REMEMBER=0` to only save
on an explicit `remember …`.

Recall uses a semantic embedder if `transformers` is installed (handles reworded questions),
otherwise keyword matching. The **talking** is your local model (Apple MLX by default) grounded in
the recalled facts — with no model, it returns the matching facts directly.

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

## `webui.py` — a Claude-style chat app for your brain

Run it, open the printed URL in a browser tab, and it's an app. No extra dependencies (Python
standard library + `personal_brain.py`). Everything stays on your machine.

```bash
python3 apps/webui.py                 # -> http://127.0.0.1:8765
PORT=9000 python3 apps/webui.py       # custom port
HOST=0.0.0.0 python3 apps/webui.py    # share on your LAN (trusted networks only)
```

In the chat, ask anything in plain language; start a message with `remember …` to teach it a fact.
It shows your fact count and recall mode (semantic / keyword) in the header. Same honest scope as
`personal_brain.py`. Each person runs their **own** instance — private, on-device, no cloud.

## Adapting to your laptop and model

Both scripts detect RAM and suggest a model tier (~0.5B → ~14B by available memory). The memory is
**model-agnostic** — swap in any model your machine can run; `researcher_bench.py` lists several
config presets, and you can add your own `(layers, kv_heads, head_dim, bytes)`. The core memory +
retrieval runs even with **no model at all**.
