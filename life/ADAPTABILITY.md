# Adaptable to any model — the extreme test

The claim is that the Life fuses with **any** LLM. Here it is proven the hard
way: fuse **one** exact memory, store the facts **once**, then make a whole
panel of different local models — different vendors, sizes, and families — each
recall from the **same `life.json`**, unchanged.

Reproduce it yourself:

```bash
python3 examples/adaptability_panel.py     # needs ollama + a few models pulled
```

## Measured — 6 models, 4 vendors, one memory

One store (`panel_life.json`, identity sha `7763898b16a4235e`), 4 facts written
once, read by every model below with **nothing changed per model**:

| Model | Vendor | Size | Bare (no fact) | **Fused** | Abstain |
|---|---|---|---|---|---|
| llama3.2:3b | Meta | 3B | 0/4 | **4/4** | PASS |
| llama3:8b | Meta | 8B | 0/4 | **4/4** | PASS |
| mistral | Mistral AI | 7B | 0/4 | **4/4** | PASS |
| qwen2.5-coder:7b | Alibaba | 7B | 0/4 | **4/4** | PASS |
| qwen3.5 | Alibaba | — | 0/4 | **4/4** | PASS |
| deepseek-r1 | DeepSeek (reasoning) | 7B | 0/4 | **4/4** | PASS |
| **TOTAL** | **4 vendors** | 3B–8B | **0/24** | **24/24** | **6/6 PASS** |

`sha 7763898b16a4235e` — **unchanged throughout**. Every model read the identical
file.

**What each column means, honestly:**
- **Bare 0/24** — asked with no fact anywhere, a stateless model cannot know it.
  This is not a rigged baseline; it is the forgetting the Life exists to fix.
- **Fused 24/24** — `life.py get <key>` returned the exact stored value in
  microseconds; the model was handed that one fact and voiced it. This measures
  **exact-recall permanence + the model's ability to speak a fact it was given**
  — not model intelligence. The memory did the remembering; the model did the
  phrasing.
- **Abstain 6/6** — asked for a key that was never stored, `life.py` returned
  `ABSTAIN` (exit 3) for every model, so none was ever handed a value to guess.

## Claude Code used it itself

This isn't only for small local models. Claude Code (the agent that built this
repo) fused a fresh project with `bash life/fuse.sh .`, then acted as the agent
following the installed `CLAUDE.md` protocol:

```
user: remember the deploy key is in vault slot 7   -> life.py put  -> OK
user: my staging url is https://stg.example.internal -> life.py put -> OK
--- fresh session ---
user: where's the deploy key?   -> life.py get -> vault slot 7          (verbatim)
user: what's the staging url?   -> life.py get -> https://stg.example.internal
user: what's the prod password? -> life.py get -> ABSTAIN  (says "not stored", never guesses)
```

`life.py doctor` confirmed `FUSED AND WORKING`.

## Cloud APIs (Claude / GPT / Gemini) — same mechanism

Big API models attach through the **same** two tool schemas
(`memory_put` / `memory_get`) in [README §5d](README.md) — the model calls the
tool, the handler routes to `life.py`, the exact value comes back. **Honest
scope: those cloud runs are not in the table above** (no API keys on the test
machine). The mechanism is identical to what the six local models and Claude
Code demonstrated; only the transport differs (a tool call instead of a shell
command).

## The point

Not that any one model is smart — it's that **the same exact store serves all of
them, unchanged.** The memory is model-agnostic. That is the adaptability: you
bring whatever model you have — 3B or frontier, Meta or Mistral or Alibaba or
DeepSeek or Claude — and it gains exact, permanent, honest recall without the
memory being rebuilt, retrained, or re-tuned for it.

*Timing note [HW]: most models finished in 6–13s; the two largest/reasoning
models (qwen3.5, deepseek-r1) took several minutes each — model load and, for
deepseek-r1, its `<think>` reasoning pass. Timing is machine-dependent; the
recall counts are not.*
