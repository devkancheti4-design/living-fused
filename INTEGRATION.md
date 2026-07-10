# Scaling this up — and the mistunes that will wreck it if you don't

The demo in `live.py` is tiny on purpose. To fuse the living organ onto a real
transformer (yours, an open model, or via an LLM's help), the architecture is
simple — but a handful of small mistakes each cause **major** degradation, and
they cost us a failed run apiece to find. Read this before you scale.

## The architecture in one paragraph

Freeze your transformer. Add ONE living organ beside it: a fact-table
`store[context] -> counts over next-tokens`, updated on every token at inference.
At each step, get the transformer's next-token distribution (the prior), and if
the current context has entries in the store, blend the store's distribution over
the prior by confidence. That's it. The transformer computes; the organ remembers;
a confidence gate decides who speaks. Nothing about the transformer changes — you
are adding a read/write memory beside a frozen brain.

## The mistunes — ranked by how badly each one degrades you

**1. Blending the organ when it barely knows anything (the #1 killer).**
A near-empty fact row is noise, not knowledge. If you let it speak, it *drowns a
good transformer*. This is exactly why our own text-perplexity test with a naive
table LOST (3.35 vs 2.53 bpc) — the crude organ outvoted a better neural predictor.
FIX: confidence weight `w = total_count / (total_count + K)`. The organ only earns
its vote as evidence accumulates. Tune K on held-out data; too low = organ shouts
from one example, too high = organ never contributes. `K ≈ 0.25–4` in our runs.
**A fixed hand-picked blend is the single biggest source of "it got worse." The
real fix at scale is a LEARNED gate** — a tiny trained layer that outputs the
blend weight from (transformer confidence, organ confidence). Do not ship the
fixed blend to production; it is a demo shortcut.

**2. Writing questions into the fact store (silent, cost us exactly 50%).**
Assertions are world-facts; queries are not. If the token stream that ASKS a
question gets written to memory the same way as the token stream that STATES a
fact, the query syntax poisons the row. Our nested test scored a suspicious exactly-
50% until we added: "query/prompt tokens never write the store." Separate episodic
input (what happened) from interrogative input (what's being asked). This is the
episodic-vs-semantic split every real memory system needs.

**3. Averaging counts instead of overwriting them (breaks updates).**
If a fact changes ("Banana is now 20"), a plain running average blends old and new
forever and never fully updates. Use **recency-dominant writes**: a new observation
outweighs the accumulated past (e.g. `row[next] += row.sum() + 1`), so the latest
write wins — that is what makes it a *register* and not a *histogram*. Without this,
the RULER-update and chain-of-custody tests collapse.

**4. Shared storage instead of disjoint rows (brings back forgetting).**
The whole zero-forgetting property comes from each context owning its own row —
learning fact A physically cannot touch fact B. If you hash contexts into a shared
low-dimensional space with heavy collisions, you reintroduce catastrophic
forgetting through the back door. Keep the store sparse and disjoint; grow it, don't
compress it, unless you measure that collisions stay rare.

**5. Unbounded integer growth (numerically bites at long life).**
Recency-dominant writes double counts; over millions of tokens they overflow.
Cap-and-halve deterministically (`if row.sum() > CAP: row >>= s`) — deterministic so
the byte-exact reproducibility survives. Ours overflowed on unqueried noise rows
before we capped it; harmless there, fatal if it hits a live row.

**6. Floating-point in the organ (kills determinism across machines).**
The organ's crown property — identical life → byte-identical organism on any
hardware — only holds if the *store* is integer. Keep counts in int; do the
float blend only at the read, and if you need cross-machine bit-identity in the
blend too, quantize it. The transformer can be float; the memory must be integer.

## Fusing it with an LLM's help (what people actually want)

You can ask an LLM to graft this onto a big model, but the LLM will get the six
points above *wrong by default* — it will write a fixed 50/50 blend, write every
token to the store, and use a running average. Give it this file. The prompt that
works: *"Freeze the transformer. Add a per-context integer count-table memory,
updated only on assertion tokens, with recency-dominant writes and a confidence
gate `w=n/(n+K)`; blend over the frozen model's logits at read time. Keep the store
integer and disjoint-per-context."* Then **test small first** — reproduce the
`live.py` scoreboard (100/100/100 vs the frozen twin) on YOUR fused model before
scaling context. If your fused model doesn't beat its own frozen twin at 1K tokens,
it will not at 1M; fix the gate, not the size.

## What to measure before you believe it

Always run the **frozen twin**: identical model, organ off. The claim is never
"high score" — it is "living beats its own frozen self." If living ≈ frozen, the
organ isn't contributing; if living < frozen, you hit mistune #1. And always print
the **determinism receipt** (two identical lives → same hash) — if it breaks, you
have float in the store (#6).
