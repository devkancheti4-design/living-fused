# Benchmark — the honest numbers

A skeptic asked three fair questions: how is a fact stored and can it derive
new facts (age → birth year)? how is this different from a dictionary? and where
is the actual benchmark behind "doesn't forget, doesn't hallucinate"? This is
the measured answer. Reproduce it:

```bash
python3 benchmark.py     # dict + life.py need only stdlib; RAG arm needs transformers+torch
```

## First, a retraction

If any description of this project said memory comes from **"floating-point
attention"** — that's wrong, and the skeptic is right to call it out. Attention
is not memory. The actual mechanism is boring on purpose: an **exact key→value
store kept *outside* the model's float weights** — which is exactly where
hallucination comes from. No attention, no floats, no magic. A dict on disk with
two extra properties. Say that, not the other thing.

## The measured comparison — 10 facts, 3 systems, 4 axes

| System | Exact recall | Paraphrase | No-hallucination | Deterministic |
|---|---|---|---|---|
| plain `dict` | 10/10 | 0/10 | 5/5 | byte-exact |
| **`life.py`** | 10/10 | 0/10 | 5/5 | byte-exact |
| RAG (MiniLM) | 10/10 | **6/10** | 4/5 | n/a (float) |

- **Exact recall** — the fact you asked for, verbatim.
- **Paraphrase** — the same fact asked in different words.
- **No-hallucination** — of 5 keys that were **never stored**, how many the
  system correctly returned *nothing* for (instead of a wrong fact).
- **Deterministic** — rebuild from the same facts → byte-identical store (SHA).

## What the numbers actually say (no spin)

**"How is this different from a dictionary?" — On exact recall, it isn't.**
Measured: `life.py` 10/10 = `dict` 10/10. For plain store-and-recall, a Python
dict is its equal, and pretending otherwise would be dishonest. The differences
are narrow and specific, and none of them is "better recall":

1. **Abstention is a contract, not a `None`.** A dict returns `None` on a miss
   and nobody wired that to the model. `life.py` returns `ABSTAIN` with **exit
   code 3**, and the fuse protocol binds the model to say "not stored" instead
   of guessing. The behavior is the same; the *enforcement* is the point.
2. **It survives process death, byte-exact.** A dict lives in RAM and dies with
   the process. `life.py`'s store reloads with an identical SHA across a reboot
   (verified in `smoke_test.py`). That's the "doesn't forget over time" claim,
   and it's real — but it's a property of *persistence + determinism*, not
   intelligence.
3. **It can fuse into the model's logits** (`blend()`), which a dict can't —
   the counts override the next-token distribution on keys it knows.

If you only need in-process store-and-recall, **use a dict.** `life.py` earns
its place when you need the abstention contract, cross-process determinism, or
logit fusion — not before.

**"Doesn't forget / doesn't hallucinate" — true, but unremarkable, and NOT
"better than existing."** `life.py` gets 5/5 on the hallucination probe — but so
does a dict. The interesting comparison is **RAG**, the system people actually
reach for: it hallucinated on **1 of 5** never-stored probes (returned a
plausible wrong fact despite a similarity threshold), and it is not byte-exact
deterministic. **But RAG crushes both exact stores on paraphrase (6/10 vs
0/10)** — because semantic matching is its whole job and exact-key stores don't
do it at all. So:

> **`life.py` is not "better than" RAG. It is a different point on the trade-off:
> exact + deterministic + guaranteed-abstention, at the cost of zero semantic
> reach.** RAG is fuzzy + semantic, at the cost of determinism and the occasional
> confident wrong retrieval. The honest design uses **both** — exact store for
> IDs/credentials/config, RAG for "which note was about X."

## The "age 42 → year of birth" question, answered exactly

```
you tell it: "I am 42"
life.py get user.age         -> 42        (exact recall)
life.py get user.birth.year  -> ABSTAIN   (the STORE does no arithmetic; it refuses to guess)
```

The store **cannot** derive the birth year, and correctly doesn't try. The
*fused* answer = the store hands back `42`, the **model** computes 2026 − 42 =
~1984. Memory remembers; model reasons. A store that invented "1984" from "42"
would be doing exactly the hallucination this is meant to prevent — so the
`ABSTAIN` is the correct behavior, not a missing feature.

## Scope — what this benchmark does and doesn't cover

- It compares the **primitives**: an exact key store (`life.py`), its honest
  baseline (a `dict`), and a real semantic retriever (MiniLM RAG). Those are the
  mechanisms underneath the product-level systems.
- It does **not** yet run Mem0 / Letta / Zep as products. Those are largely
  RAG + LLM-summarisation wrappers, so on these four axes they behave like the
  RAG row plus a summarisation step; a proper product-level bench-off is future
  work, and this file will say so until it's done.
- The cross-model recall result (one memory, 6 models, bare 0/24 → fused 24/24)
  is in [ADAPTABILITY.md](ADAPTABILITY.md); the mechanism proof is in
  [proof/PROOF.md](proof/PROOF.md).

**Bottom line for a skeptic:** yes, it's a key-value store with deterministic
retrieval and an abstention contract. That's the whole claim, it's measured
here, and it does not beat semantic RAG — it trades semantics for exactness and
determinism. Use it where that trade is the one you want.
