# The Required-Fusion Proof

**Claim.** A gradient-trained transformer LLM cannot provide exact, permanent, honest
memory — not because nobody has built it in yet, but because the mechanisms that make
the LLM trainable exclude it. An exact-key store (here: *Life*) provides precisely the
excluded properties and cannot provide what the LLM does. Therefore an LLM+Life fusion
is not an engineering convenience — it is the design forced by the measurements below.

**Status of this document.** Every quantitative statement links to a deterministic,
dependency-free experiment in [`experiments/`](experiments/) that reproduces it
bit-for-bit (fixed LCG seeds, stdlib only). Two experiments additionally run against
real local LLMs via ollama. Nothing in this file is asserted without a measurement or
an explicitly-marked literature anchor. Scope limits are in §8 — read them before
quoting the claim.

Reproduce the core in one command:

```
cd proof && bash run_proof.sh          # pillars 1-5, ~2 minutes, no deps
bash run_proof.sh --llm                          # + real-LLM behavioral pillar (needs ollama)
```

---

## 1. Exact recall is optimizer-inaccessible (Pillar I)

*Experiment:* [`experiments/exp3_exactness_untrainable.py`](experiments/exp3_exactness_untrainable.py)

Give attention its best case: perfect key geometry, distinct values, one scalar
(sharpness β) to learn. Even there:

- The training gradient shrinks **proportionally to the remaining error** — measured
  across five orders of magnitude. Consequence: a power-law stall; the exactness gap
  shrinks **9.9× per 10× training steps**. Every extra digit of lookup precision costs
  10× the compute, with no floor.
- On near-duplicate keys (cosine c to the query), the sharpness needed for 99.9%
  exactness diverges as **1/(1−c)**: at c = 0.99 it is β ≈ 885, while 100,000 steps of
  training reach only β ≈ 18. After a fixed 5,000-step budget the model leaks **2% of
  attention mass at c = 0.90, 17% at c = 0.95, 85% at c = 0.99** to wrong values.
- The exact-key dict control: **zero leak, zero training, at every confusability.**

The literature supplies the matching upper wall: pushing attention sharp destabilizes
training (attention entropy collapse; σReparam/QK-norm exist to prevent it). Exact
lookup is squeezed from both sides — no gradient signal to reach it, instability if
forced there. **The optimizer cannot park where exact memory lives.**

## 2. No fixed-size differentiable memory can substitute (Pillar II)

*Experiment:* [`experiments/exp4_fastweight_capacity.py`](experiments/exp4_fastweight_capacity.py)

The standard in-architecture writable memory (fast weights / linear-attention state — a
d×d associative matrix) was loaded with one-shot key→value writes, d = 128:

| writes K | K/d | recall | first-16 memories |
|---|---|---|---|
| 32 | 0.25 | 100% | 100% |
| 128 | 1.0 | 98.4% | 100% |
| 192 | 1.5 | 87.5% | 87.5% |
| 512 | 4.0 | 58.6% | 81.2% |
| 1024 | 8.0 | 37.0% | **31.2%** |

The cliff sits at **K = d**, and interference is **retroactive** — writing new memories
physically degrades old ones (first-written facts fall 100% → 31%). The exact table:
100% at every load, paying O(K) growth instead. This is the measured **memory
trilemma**: {fixed-size, interference-free, growing-knowledge} — at most two. The LLM's
weights and any bolted-on differentiable memory sit in the fixed-size corner; *permanent*
memory under continuous writing is excluded there.

## 3. The LLM cannot be honest about what it doesn't know (Pillar III)

*Experiments:* [`experiments/exp5b_confidence_on_garbage.py`](experiments/exp5b_confidence_on_garbage.py),
[`experiments/exp6_trajectory_uncertainty.py`](experiments/exp6_trajectory_uncertainty.py)

A trained net at its generalizing best (100% held-out) was probed with pure garbage
inputs (no structure, no valid key):

- Mean max-softmax confidence: **0.983 trained / 0.985 held-out / 0.974 garbage** —
  36/40 garbage inputs answered with >90% confidence. Softmax is a **normalizer**: it
  reports relative preference among a menu and structurally cannot represent absence of
  evidence. Confident hallucination is the readout working as designed.
- The exact-key table on the same probes: **abstains 60/60** — key-absence is a free,
  structural, always-calibrated "I don't know."
- The gradient-side fix exists but is priced (exp6): ensemble disagreement separates
  garbage 29.5× — at the cost of N training runs. Honesty is free on the table side,
  expensive on the gradient side, and impossible at the single-softmax readout.

## 4. Writing new facts into weights is conditionally destructive — and the condition is decidable (Pillar IV)

*Experiments:* [`experiments/exp2_memory_conservation.py`](experiments/exp2_memory_conservation.py),
[`experiments/exp2b_conflict_control.py`](experiments/exp2b_conflict_control.py)

Sequential training, two regimes, same net:

- **Agreement regime** (tasks share one latent rule): retention dips then **self-heals**
  (75% → 92%) while held-out generalization reaches 100% — later data is evidence, not
  interference.
- **Conflict regime** (rule changes per task; keys disjoint): retention collapses through
  **33% → 25% → 8%** at the trough, ending at 42% (chance = 25%).
- The count table: **100% retention in both regimes**, 0% generalization, abstains on
  unseen keys.

**The Routing Law:** gradient memory forgets exactly what the stream *disagrees* about
and keeps what it agrees about; table memory is indifferent to agreement. This makes
"what may safely consolidate into weights" a *decidable, measurable* property of content
— the consolidation gate of the fusion. It also explains the field's own behavior:
i.i.d. shuffling is agreement manufacturing — offline training terraforms reality's
correlated stream into the only regime gradient memory survives. An online system gets
no shuffle; it needs the table organ or replay. There is no fourth option in the data.

## 5. The failure is behaviorally real in production LLMs today (Pillar V)

*Experiments:* [`experiments/exp10_k_needle_law.py`](experiments/exp10_k_needle_law.py),
[`experiments/exp10b_k_needle_load.py`](experiments/exp10b_k_needle_load.py)
(llama3.2:3b and llama3:8b, local ollama, temperature 0, 128 facts in context)

- **Load law:** per-fact recall accuracy falls as k facts are queried simultaneously —
  3b: 100% → 87.8%; 8b: 100% → 88.5% (k = 4 → 64). Recall budget is per-question, not
  per-fact: the independence null is rejected on both models.
- **Swap signature:** the errors are another fact's value — 82% of errors under
  near-duplicate keys (3b), 78% even with distinct keys (8b). The model confidently
  hands you the neighbor's fact (Pillars I + III, in the wild).
- **Similarity tax:** near-duplicate keys cost ~4–8 points even at k = 4 — the 1/(1−c)
  leak of Pillar I surfacing behaviorally.
- **Scale is not the fix:** the 8b model was *not* shallower than the 3b (−11.5 vs
  −12.2 points; more swaps). (Caveat: cross-generation pair, 1.33× width — see §8.)

## 6. Life holds exactly the complementary corner — and only that corner

Measured previously in the parent project (red-teamed there; **not reproducible from this repo alone** — quoted with that provenance):

- Exact-key store: **0 collisions at 10M/20M/30M facts**; ~410 bytes and ~1 µs per
  fact, O(1) flat — exact and permanent where Pillars I–II are leaky and mortal.
- **Byte-exact determinism across process death** (cross-process SHA match) — memory
  that outlives the context window and the process.
- Structural abstention (§3) — honest where the readout cannot be.
- **And the other direction:** Life alone is a failed generator (0 valid generated
  programs) and does not generalize (0% held-out on unseen keys, by design). The
  gradient organ wins statistics and generalization (exp2's 100% held-out; measured
  organ-specialization on enwik8). Fused recall was measured end-to-end earlier:
  **bare LLM 0/40, in-context 36/40, fused 40/40.**

## 7. The argument, assembled

1. Exact recall cannot be *trained into* the LLM (I), cannot be *bolted on* as any
   fixed-size differentiable memory (II), and its failures are *silent* (III) — so an
   LLM-only system can be neither exact, nor permanent, nor honest about which.
2. Continuous writing into weights is safe only for agreement-bearing content (IV), so
   an online system must route conflicting/arbitrary/exact content elsewhere — and the
   routing condition is measurable.
3. These are not hypothetical failure modes: they are measurable in today's production
   models from the query side alone (V) — and more parameters did not fix them.
4. A component holding the excluded properties exists and is measured (VI Life), but it
   cannot generate or generalize — the LLM's corner.
5. Two components, each mathematically excluded from the other's corner, plus a
   decidable routing law between them: **the fusion is the design the measurements
   force.** Biology reached the same partition (complementary learning systems:
   hippocampus + neocortex + gated consolidation) — an independent prior, not proof.

## 8. Scope — what this does and does not prove

- "Never" means: within gradient-trained softmax-readout architectures, as measured at
  the mechanism level (vanishing signal ∝ error; normalizing readout; K = d cliff) and
  confirmed behaviorally on real models. It is **not** a claim about all conceivable
  future architectures — it is a claim about the paradigm's own mathematics.
- Pillars I–IV are mechanism isolations at small scale by design (Rule 12: prove the
  limit in the mechanism's *best* case; richer variants inherit the failure mode).
  Pillar V is the bridge to production scale; its size-scaling arm used a dirty model
  pair (cross-generation, 1.33× width) and awaits a clean same-family ladder.
- Individual components here overlap known results (feedback alignment, deep ensembles,
  BIC, fast-weight capacity, CLS theory, multi-needle degradation) — cited in the
  experiment files. What is claimed as ours is the **assembled, quantified, reproducible
  chain** from mechanism to behavior to design conclusion, plus the specific
  measurements (per-fact independence-null rejection with swap decomposition; the
  (sustain, residual) curiosity coordinates; the routing-law regime split).
- No commercial-value claim is made or implied. Four earlier "billion-dollar" claims in
  this project died under fair baselines; this document survives *because* it makes no
  such claim.

## 9. Reproduction map

| Pillar | Experiment | Runtime | Needs |
|---|---|---|---|
| I | exp3_exactness_untrainable.py | ~10 s | python3 only |
| II | exp4_fastweight_capacity.py | ~60 s | python3 only |
| III | exp5b_confidence_on_garbage.py, exp6_trajectory_uncertainty.py | ~30 s | python3 only |
| IV | exp2_memory_conservation.py, exp2b_conflict_control.py | ~20 s | python3 only |
| V | exp10_k_needle_law.py, exp10b_k_needle_load.py | ~30–60 min | ollama + llama3.2:3b / llama3:8b |
| supporting | exp1, exp5, exp7, exp8, exp9 | seconds–minutes | python3 only |

All pure-Python experiments are deterministic (fixed LCG seeds, no `random`, no
network) — same numbers on every run, every machine.

---

*Produced by the Reasoning Engine loop (see [README.md](README.md)): sector-window
question mining → answer → adversarial critique → deterministic experiment → commit.
The engine's evolved rulebook and full round history live in the parent project
(research_miner/memory and research_miner/rounds — not shipped in this repo).*
