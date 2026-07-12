# What the living memory does — measured, at scale, with the failures shown

Everything here was measured on one Apple M4 Pro (24 GB, CPU) and reproduces from the
harnesses in this repo. Fixed seeds; your numbers should match. The second half — **how we
mistuned and lost** — matters as much as the results: the mechanism is ~15 lines, but every
one of those numbers required getting the tuning right, and a naive graft silently returns
~0 *while looking like it works*.

## 1. The graft scales — 124M to 7B, three model families, fp32 and 4-bit

The living memory only consumes a model's next-token distribution, so it is model-agnostic.
We grafted the *same* memory onto four frozen models and asked each to recall facts placed
**beyond its context window** — where the model alone is blind.

| Model | Family | Precision | Footprint | Synthetic recall (alone → +Life) | Real Wikipedia (alone → +Life) |
|---|---|---|---|---|---|
| GPT-2 124M | GPT-2 | fp32 | ~0.5 GB | 0/8 → **8/8** | — |
| Llama-3.2-1B | Llama | fp32 | ~5.5 GB | 0/10 → **10/10** | 19/40 → **40/40** |
| Llama-3.2-3B | Llama | 4-bit | ~2.5 GB | 0/10 → **10/10** | 19/40 → **40/40** |
| Qwen2.5-7B | Qwen | 4-bit | ~4.7 GB | 0/10 → **10/10** | 17/40 → **40/40** |

A **7B model (4-bit) fits in ~4.7 GB** on a laptop, and the deterministic memory adds only
**megabytes** on top (14.7 MB for 30,000 facts — see §2), growing ~0.5 KB/fact. Reproduce:
`python3 scale_mlx.py mlx-community/Qwen2.5-7B-Instruct-4bit`.

## 2. Retention — 30,000 real Wikipedia facts, zero forgetting

Store N real Wikipedia facts in the Life and it retains **all** of them as N grows — because
each fact lives in its own row, a new fact *physically cannot* disturb an old one.

| N real facts | retained | forgotten | Life memory |
|---|---|---|---|
| 1,000 | **100%** | 0 | 0.5 MB |
| 10,000 | **100%** | 0 | 4.8 MB |
| 30,000 | **100%** | 0 | 14.7 MB |

That is retention a fine-tune cannot match — a fine-tune *forgets* as you cram more in.
Reproduce: `python3 retention.py`.

## 3. Routing — the memory answers with the model idle

The confidence gate is also a router. For a query the Life knows, it answers in ~0.1 µs with
**zero model forwards**; the big model only wakes for what the Life doesn't know (a real
counter proves the zero).

```
answered by the LIFE (0 model calls): 20/40   ~0.1 µs   correct 20/20
routed to the MODEL (Llama forward):  20/40   ~173 ms
```
So the footprint isn't "5 GB every time" — the always-on part is the tiny Life; the model is
an on-demand fallback. The Life-handled fraction = your workload's lookup share.
Reproduce: `python3 routing.py`.

## 4. Streaming — flat forever, while the naive way OOMs

A transformer fed a *growing* context climbs ~237 KB/token and **OOMs a 24 GB machine at
~79,000 tokens**. The same model on a fixed window + the Life sits **flat at its resident
size forever**, ingesting at ~500K tok/s and recalling beyond the window. (And the memory
layer alone survives a **20-billion-token** soak at 0 memory growth — see
[INFINITE_STREAM_SOAK.md](INFINITE_STREAM_SOAK.md).)

---

# How we mistuned and lost — and why tuning is everything

Every result above has a failed first attempt behind it. Most failures **looked like
success** until adversarially checked. This is the real lesson: the mechanism is trivial;
the tuning is not, and a wrong knob degrades silently.

**1. We tested it with the life switched OFF — and "lost" three exams.**
Early long-context tests scored 8% / 38% / 0%. The cause: the eval ran the model frozen
(`@torch.no_grad`, the memory never updating). Switching the life **on** — updating on every
token — turned the same three exams into **100% / 100% / 100%**. *Lesson: the entire point is
that it learns at inference; a frozen eval measures the wrong object.*

**2. A tokenizer mismatch dropped recall to 7/8 — and it wasn't the memory's fault.**
On Llama, re-tokenizing the query string produced a *different* token frame than the stored
fact line for one name, so the key missed. Keying on the **exact stored frame** → **10/10**.
*Lesson: the memory keys on tokens; the query must tokenize to the same frame. A "miss" is
usually a key mismatch, not a memory failure.*

**3. We ran the transformer every token — and hid the memory's speed (17.6 tok/s).**
Running a full model forward per token made throughput reflect the *transformer*, not the
Life. Firing the model only at query time (the Life ingests every token) → **~500,000 tok/s**
ingest. *Lesson: fuse the efficient way — the memory ingests cheaply; the model runs on
demand. Otherwise you measure the wrong component.*

**4. Unbounded filler values made it slow down and creep in RAM (85K → 16K tok/s).**
A stress test fed 128K *distinct* values into each row, so the rows grew without bound.
Bounding the value set (and the deterministic cap-rescale on counts) → flat **500K tok/s**,
flat RAM. *Lesson: the store is bounded only if what you write to it is bounded; unbounded
distinct writes are storage growth, not a leak, but they will slow you down.*

**5. Memory-mapped weights faked the OOM number (975 KB/tok → really 237 KB/tok).**
The first growing-context run counted the model's weights *paging into RAM* as KV-cache
growth, inflating the per-token cost 4× and the OOM point the wrong way. Warming up the
weights first and measuring the *linear-regime* slope → **237 KB/token, OOM ~79K tokens**.
*Lesson: separate one-time load from per-token growth; measure after weights are resident.*

**6. Measurement ordering inflated a footprint (6,368 MB → really 5,503 MB).**
Running the fixed-window test *after* the growing-context test reused a large allocator pool,
inflating the "flat" number. A fresh process → **5,503 MB**. *Lesson: measure each footprint
in isolation.*

**7. A naive fixed blend makes prose *worse* than the model alone.**
On soft text, a hand-tuned blend can outvote a *better* neural prediction (we lost a
perplexity test 3.35 vs 2.53 bpc this way). The fix is the **confidence gate** so the memory
only speaks when it's sure; at scale you replace the hand-tuned gate with a **learned** one.
*Lesson: organs specialize — gate the memory so it stays silent when it would hurt.*

**Plus the six documented mistunes** in [INTEGRATION.md](INTEGRATION.md): gate too high/low,
writing questions into the store, averaging instead of overwriting, shared vs disjoint rows,
unbounded integer growth, and float in the store (which kills determinism).

## Why tuning is important — the one-paragraph version

The living memory is about fifteen lines of integer arithmetic. But each knob — the
confidence gate, the key frame, assertion-only writes, recency-dominant overwrites, disjoint
rows, the cap-rescale, matched keys, integer-only storage — silently degrades or **fakes**
the result if it's wrong. Half of the failures above returned a plausible-looking number
(a clean 0%, a slow throughput, an inflated footprint) that was an artifact, not a finding.
The numbers in the first half of this document exist because each of those knobs was tuned
*and then adversarially re-checked*. A naive graft doesn't crash — it quietly does nothing,
or hurts, while looking like it works. That is why the tuning, and the red-teaming of every
number, is the actual work.

## Honest caveats (read before citing)

- Model-alone recall on real Wikipedia is ~40–48% because many continuations are trivially
  predictable (spaces, common words); the Life's edge is the *specific* facts the model can't
  predict but retains perfectly.
- The real-data tests use real **public** text (Wikipedia), not any user's private data.
- 7B was tested at **4-bit** (the practical size for a 24 GB laptop); fp16/fp32 would be a
  bigger footprint, same recall.
- Recall is deterministic matched-key retrieval — it needs the query to tokenize to the same
  frame that was stored (consistent tokenization).
- These are architecture-level demonstrations, not product benchmarks; the memory is a *fact*
  memory and does not add reasoning.
