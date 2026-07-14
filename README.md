# living-fused

TL;DR: a small deterministic memory you bolt onto a local LLM. You tell it a fact, it stores that
fact exactly (as integers, not floats), and when you ask, it gives the model the exact fact instead
of letting the model guess. So the model won't drift or invent things about stuff you've actually
told it. Runs on your laptop, stays private, and the memory stays tiny no matter how long the input
gets (there's no growing KV-cache).

Why I built it: LLMs hallucinate and forget because everything they "know" is either frozen into the
weights or crammed into a context window that costs memory and gets ignored anyway. I wanted a memory
that sits outside the model, is exact, and is reproducible. Same input, same output, every time. If
it saved a fact, it hands back that fact. It doesn't bluff.

What it does, plainly:
- you give it facts, it stores them in a flat integer table;
- when you ask, it finds the relevant ones and either hands them to the model to phrase (the
  assistant app), or blends them straight into the model's next-token probabilities so the stored
  fact wins over a guess (the research core);
- that table doesn't grow with how many tokens you stream past it, only with how many facts you save.

"How is this different from a markdown file or RAG?" Honest answer: for the assistant demo, not that
different. It retrieves facts and asks a model, which plenty of tools do, and yes you could rig
something similar with markdown plus a retriever. What's actually new is two things. One, the memory
is deterministic and exact, so recall never drifts and you can force the model to use the stored
value instead of a hallucinated one. Two, it lives outside the context window, so a million tokens of
history cost kilobytes instead of gigabytes of KV-cache. That second part is the real result. Judge
it by the benchmark, which runs in one command.

Status, so nobody's misled: this is a research demo, not a product. Some of the code is rough and got
built fast. But the numbers are real and you can reproduce them yourself. `researcher_bench.py` needs
nothing installed. If a number doesn't reproduce, open an issue and I'll fix it or take it down.

Fastest way to see it is the apps below. To watch the core memory train from scratch and beat its own
frozen copy:

```bash
git clone https://github.com/devkancheti4-design/living-fused.git
cd living-fused && pip install -r requirements.txt
python3 live.py
```

That's the whole setup. In ~10–15 minutes on a laptop CPU (no GPU, no internet after
install), you will watch:

1. a small **fused body** train from scratch — a windowed transformer (skeleton)
   plus a persistent recurrent current — on streams **no longer than 420 tokens**;
2. the **life switch on**: a disjoint integer fact-table that updates on *every token
   the model reads*. Deployment is learning. It never freezes;
3. the fused model takes three exams **up to 150× beyond its training length**, scored
   live against its own **frozen twin** (identical weights, life off):

| exam | scale | ALIVE | frozen twin | chance |
|---|---|---|---|---|
| RULER-style: 10 variables + mid-stream **updates** | 32,000 tokens | **100%** | 8% | ~6% |
| chain of custody ("who holds it *now*?") | 64,000 tokens | **100%** | 62% | ~6% |
| code-dependency trace (corrupted var → use-site) | 20,000 tokens | **100%** | 12% | ~6% |

4. a **determinism receipt**: two identical lives produce **byte-identical**
   models (SHA-256 of the fact-tables printed);
5. the **cost of being alive**: ms/token measured at 1K vs 32K context — flat.

Fixed seeds — your numbers should match the table. If they don't, open an issue.

## 📦 The Memory Box — [`life/`](life/) — fuse any model, one command

The memory organ, packaged standalone: one stdlib file, zero dependencies,
works with **any** model on **any** device — Claude, GPT, a local ollama
model, anything. One command fuses it into any project:

```bash
bash life/fuse.sh ~/your/project
```

That single command: runs the 16-check smoke test (refuses to fuse a failing
organ) → installs `life.py` → writes the memory protocol into `CLAUDE.md` and
`AGENTS.md` (so **Claude Code and Cursor sessions in that directory are fused
automatically — nothing to paste**) → detects every model backend on your
device and prints how each one connects → proves the put/get/ABSTAIN
round-trip live. From then on, the agent in that directory stores facts
exactly, recalls them verbatim, and says "not stored" instead of guessing.

Big API models (Claude/GPT/Gemini): two function-calling schemas in
[`life/README.md`](life/README.md) §5d route to the same file.
Critics asking what this is: [`life/WHAT_IS_LIFE.md`](life/WHAT_IS_LIFE.md) —
every answer is a command, not a claim. The box ships its own proof
(`life/proof/`, reproduces in ~2 min) and honest limits
([`life/README.md`](life/README.md) §7): exact-key only, doesn't reason,
pairs with RAG for paraphrases.

## Quick start — the apps

Clone it and run. The apps **adapt to your laptop** (detect RAM, pick a model tier) and **degrade
gracefully**, so they run on any machine:

```bash
git clone https://github.com/devkancheti4-design/living-fused.git
cd living-fused

python3 apps/researcher_bench.py     # KV-cache / cost benchmark — needs NOTHING installed
python3 apps/personal_brain.py       # your private conversational memory (terminal)
python3 apps/webui.py                # a Claude-style chat app -> http://127.0.0.1:8765
```

Optional, for the full experience:

```bash
pip install transformers torch   # -> semantic recall (understands reworded questions)
pip install mlx-lm               # -> a local model phrases replies (Apple Silicon)
```

No `transformers`? recall falls back to keyword. No local model? it returns the matching facts
instead of phrasing them. `researcher_bench.py` needs nothing at all.

**What talking to your brain looks like** (all local, saved to `~/.personal_brain.json`):

```
you   > my daughter Mia just turned six
brain > Happy 6th birthday, Mia!                         (it remembered)
you   > I have a dentist appointment tomorrow at 9am in room 214
brain > Got it — see you at 9am in room 214.             (remembered)
you   > when is it?
brain > The dentist appointment is tomorrow at 9am.      (follow-up resolved)
you   > and where?
brain > It's in room 214.                                (multi-turn)
you   > can I eat a peanut butter sandwich?
brain > Not if you're allergic to peanuts.              (recalled)
```

Just talk to it — it works out whether you're telling it something or asking. Everything stays on
your machine; each person runs their own. Full usage + honest scope in [`apps/README.md`](apps/README.md).

## The three apps in one line each

- **`apps/researcher_bench.py`** — reproduce the context-memory decoupling on your machine (KV-cache
  O(N) table vs the flat O(1) store, a flat-memory proof, lookup latency). No model needed.
- **`apps/personal_brain.py`** — a private, on-device, conversational memory: talk to it, it
  remembers facts and answers from them, with multi-turn follow-ups.
- **`apps/webui.py`** — a local, Claude-style chat UI for the brain; open it in a browser tab.

## What it is · what it's not · how to use it (full clarity)

**What it is**
- A **memory layer** for local LLMs: a flat, integer, deterministic store of facts that lives
  *outside* the model's context window, so long-context memory costs kilobytes instead of a growing
  KV-cache.
- A **private, on-device assistant** (`personal_brain.py` + `webui.py`): you talk to it, it remembers
  your facts and answers from them, entirely on your machine and persistent across sessions.
- A **reproducible benchmark** (`researcher_bench.py`): shows the KV-cache `O(N)` vs flat `O(1)`
  scaling on your own laptop, no GPU cluster needed.
- **Model-agnostic** — the memory works alongside any local model you can run.

**What it is not**
- **Not a new foundation model or a reasoning engine.** The intelligence — language, reasoning,
  analysis — is your *local model's*. The memory only supplies the right facts.
- **Not a replacement for a vector DB / semantic search.** The reworded-query matching *is* a
  sentence-embedder (float vectors + cosine) — the same tech a vector DB uses. It doesn't replace
  that; it uses it.
- **Not cheaper generation.** It does not speed up or reduce the cost of the model *generating* text
  (that's transformer-speed). What it removes is the *context-memory* cost (the KV-cache), not inference.
- **Not a cloud service.** There is no hosted version. Each person runs their own instance locally;
  nothing is uploaded.
- **Not guaranteed correct.** Semantic recall can pull the wrong fact (negation, rare synonyms, thin
  margins when many facts compete), and the local model can misphrase or hallucinate. Treat answers
  as helpful, not authoritative.

**How to use it**
1. Clone and run (see **Quick start** above). `researcher_bench.py` needs nothing installed; the
   assistant uses `transformers` for recall and a local model (Apple MLX by default) for replies, and
   degrades gracefully to keyword + raw facts without them.
2. Talk to it in plain language — it detects whether you're *telling* it a fact or *asking*.
   `remember …`, `forget …`, `forget all` are explicit controls; `AUTO_REMEMBER=0` disables auto-saving.
3. Your data lives in `~/.personal_brain.json` on your machine — a plain JSON file you can back up,
   edit, or delete. To open the web app to a trusted LAN: `HOST=0.0.0.0 python3 apps/webui.py`.
4. To use your own model, edit one function: `_model_chat()` in `apps/personal_brain.py`.

**Terms & disclaimer**
- License: **AGPL-3.0** — use, modify, and self-host freely; any network-served modification must be
  open-sourced under the same terms. See [LICENSE](LICENSE).
- Provided **as-is, with no warranty.** Do **not** rely on it for medical, legal, financial, or any
  safety-critical decisions — it is a personal memory tool, not a source of truth.
- **Your data stays yours and local.** The project never transmits your data anywhere. If you enable
  LAN sharing, doing so safely on a trusted network is your responsibility.
- You are responsible for whichever model and data you choose to run with it.

## The idea in four sentences

A frozen model can only use what fits in its attention window; everything else is
gone. This architecture adds a **living organ**: a deterministic integer fact-table
with recency-dominant writes — one observation stores a fact, a later write always
overrules it, and learning fact A physically cannot disturb fact B (disjoint rows —
zero forgetting by construction). Questions never write the store (assertions are
world-facts; queries are not). The neural body supplies the prior; the living organ
supplies memory that survives any context length, at **constant cost per token**.

## Scaling it up — proven on a real pretrained transformer

`live.py` trains its own tiny body. But the organ is **model-agnostic** — you can
bolt it onto a *frozen, pretrained* transformer of any size and give it memory it
structurally cannot have. `scale_demo.py` does exactly this on **GPT-2 (124M — ~1000×
the toy body)**:

```bash
pip install torch numpy transformers
python3 scale_demo.py         # first run downloads GPT-2 (~500MB)
```

It places arbitrary facts (`The Zephyr device is red`) **~1,500 tokens back — beyond
GPT-2's 1024-token window** — then queries them at the end:

| test | GPT-2 alone | GPT-2 + living organ |
|---|---|---|
| **recall** a fact beyond the window | **0 / 8** | **8 / 8** |
| **revise** a fact that was updated | **0 / 8** | **8 / 8** |

GPT-2 alone is blind (the facts scrolled out of its window and are arbitrary). The
organ gives the *same frozen weights* beyond-window recall and fact-revision.

### It is not just GPT-2 — the same organ scales to a 1B-param model

`scale_llama.py` runs the **identical organ** on a frozen, pretrained **Llama-3.2-1B
(1.24B params — ~8,000× the toy body)**. Here the served window is **deliberately
capped at 200 tokens** — Llama-3.2-1B natively handles 128K, so we serve only 200 to
keep CPU inference fast *and* to model a cost-capped deployment — with the facts placed
~450 tokens back, beyond that served slice:

```bash
python3 scale_llama.py        # first run downloads Llama-3.2-1B (~2.5GB); CPU is slow but works
```

| test | Llama-1B (served 200) | Llama-1B + living organ |
|---|---|---|
| **recall** a fact beyond the served window | **0 / 6** | **5 / 6** |
| **revise** a fact that was updated | **0 / 6** | **6 / 6** |

**Read this table honestly:** the `0/6` baseline reflects the *imposed* 200-token cap,
not a limit of Llama — served its full 128K window, Llama could hold these facts itself.
What this demonstrates is different from the GPT-2 case (where ~1,500 tokens back is a
*real* 1024-window limit): it shows the **same organ code runs unchanged on a 1B-param
model** and supplies memory beyond *whatever* window you choose to serve — the point
being that in a real deployment you serve a bounded window for cost, and the organ
carries the rest. The Organ class here is the same one in `scale_demo.py`; nothing in
the recipe changes when the model gets 8,000× bigger. (Recall is **5/6, not a clean
sweep** — one fact misses on a Llama-BPE tokenization edge case, and it reproduces at
5/6 across runs, so it's a stable quirk, not noise.)

The organ only needs a next-token distribution, so the **code** is model-agnostic — swap
`MODEL` for any HuggingFace causal LM. We have verified GPT-2 and Llama-3.2-1B; others
(Mistral, Qwen, …) should slot in unchanged but are **untested**, and tokenizer
differences can shift the numbers (see the 5/6 note). The transformer is never modified.

### How far the memory itself scales

`max_organ.py` pushes the memory layer alone (no transformer) to a machine's ceiling —
capacity, speed, and byte-exact determinism:

```bash
python3 max_organ.py          # pure Python + numpy; no GPU, no downloads
```

| facts held | recall accuracy | RAM |
|---|---|---|
| 1,000,000 | **100%** | 496 MB |
| **5,000,000** | **100%** | 2.4 GB |

At 5 million facts it still recalls **100%** and answers **~1.5 million lookups/sec**,
with flat per-fact cost — the limit hit is the machine's RAM, not the algorithm. Two
more receipts the script prints: **2,000,000 overwrites on 1,000 keys → latest value
correct 1,000/1,000** (last-write-wins holds under heavy contention), and **1M facts
built twice → byte-identical SHA-256** (the same reproducibility the demo shows, at
scale). Write throughput fluctuates a few percent run to run (wall-clock on a shared
CPU); the accuracy, RAM, overwrite, and determinism results are exact.

### Exactly how the graft works (5 lines of real logic)

1. **Freeze the transformer.** Read its next-token distribution as the *prior*.
2. **Key a memory on the fact frame.** For each token, `key = tokens since the last
   newline` — so `The Zephyr device is` is one isolated key, and `→ red` is its value.
3. **Write only on assertions, recency-dominant.** `store[key][next] += sum(store[key])
   + 1`. A later write always outweighs the past, so facts *update*, not average.
   Questions never write.
4. **Blend by confidence at read time.** `w = n / (n + C)`, then
   `p = (1-w)·prior + w·memory`. If the key is unknown, the organ stays silent and the
   transformer speaks.
5. That's it. No training, no fine-tuning, no touching the transformer.

### Two mistunes that silently make it do *nothing* (we hit both, live)

Getting the graft to 8/8 took fixing two things that otherwise return ~0/8 *while
looking like they work*:

- **The gate `C` was too high** (a known fact got only 0.2 weight and GPT-2's generic
  guess drowned it). A single, definite fact must **dominate** the prior — small `C`
  (we use `0.25`). This is mistune #1.
- **The write frame must match the query frame.** We stored `X is now blue` but asked
  `X is ___` — different keys, invisible update. Store what you'll ask for.

Read [INTEGRATION.md](INTEGRATION.md) before scaling — it lists all six mistunes, each
from a run that actually failed.

## Fusing onto bigger models — and letting an LLM wire it up

`live.py` is tiny on purpose; `scale_demo.py` and `scale_llama.py` show the *same* organ
on frozen 124M and 1.24B models with **zero code changes**. On a bigger machine (more
RAM/VRAM, a 7B–70B model — **untested at this size, so this is an architectural argument,
not a measured result**) the recipe still does not change — the organ has **constant
per-token cost, independent of context length and model size, and is model-agnostic, so
you scale the transformer, not the organ**:

1. Load your model **frozen** (any HF causal LM). Read its next-token logits as the prior.
2. Add one `Organ` beside it — the ~15-line class in `scale_demo.py`. Keep it **integer
   and disjoint-per-context**.
3. Write to it on **assertion tokens only**; blend at read time with the confidence gate.
4. **Reproduce the `live.py` frozen-twin scoreboard on YOUR model at ~1K tokens before
   scaling context.** If living doesn't beat its frozen twin small, it won't big — fix
   the gate, not the size.

**You can hand this to an LLM (Claude, GPT-4, etc.) to graft onto your model** — but it
will get the details wrong *by default*: it writes a fixed 50/50 blend, writes every
token to the store, and averages instead of overwriting (three of the six mistunes).
Give it [INTEGRATION.md](INTEGRATION.md) and this prompt verbatim:

> *"Freeze the transformer. Add a per-context integer count-table memory, updated only on
> assertion tokens, with recency-dominant writes (`row[next] += row.sum()+1`) and a
> confidence gate `w = n/(n+K)`; blend over the frozen model's logits at read time. Keep
> the store integer and disjoint-per-context. Then reproduce the frozen-twin scoreboard at
> 1K tokens before scaling context."*

For **production** at scale, replace the fixed gate with a **learned** gate — a tiny
trained layer that outputs the blend weight from (transformer confidence, memory
confidence). The hand-picked blend is a demo shortcut and is the #1 source of "it got
worse" (mistune #1). Two practical notes for bigger machines: RAM is the ceiling for the
memory (~0.5 GB per 1M facts here), while the **transformer's own weights + KV-cache
dominate at 7B+**; and the organ adds **~0 to per-token latency** (flat with context), so
its cost is memory, not compute.

## Cost & scaling — the real math

The one property worth citing, with the formula and the measurement behind it. **Context
length is decoupled from memory:** a transformer's KV-cache grows linearly with context; the
integer store does not.

**KV-cache a transformer needs, per token** (7B with GQA — 28 layers, 4 KV heads, 128 head-dim, fp16):

```
KV-cache = 2 × layers × KV-heads × head-dim × bytes
         = 2 × 28 × 4 × 128 × 2  =  57,344 bytes/token  ≈  56 KB/token
```

| context length | transformer KV-cache | integer store (fixed #facts) |
|---|---|---|
| 1K   | 57 MB  | ~KB *(constant)* |
| 100K | 6 GB   | ~KB |
| 1M   | 57 GB  | ~KB |
| 15M  | **860 GB** | ~KB |

That is **O(N) vs O(1)** in context length — the store scales with number of facts, not tokens.

**Measured operation costs** (10× each, one consumer laptop):

| operation | cost |
|---|---|
| exact-key lookup | 0.33 µs, zero model calls |
| recency-dominant integer write | 0.36 µs |
| fused blend over the full vocab (per fire) | ~14 µs |
| fused **generation** (transformer forward + blend) | **~120 ms/token, flat as memory grows 1K → 1M facts** |
| transformer forward vs context length | 144 → 804 → 6,937 ms/token at 64 → 512 → 4,096 tokens |

**A representative task, measured end-to-end** — a 15M-token stream where a cheap filter flagged
~13 relational sentences for the model, then one multi-hop query:

```
15M-token stream (filter/count, no model)     :  0.2 s
13 transformer extractions (only heavy calls) :  7.7 s
1 query                                       :  0.9 s
total on a laptop                             :  8.8 s   ≈ $0.000015 electricity
```

The transformer-only alternative for the same 15M-token context would need **~860 GB of
KV-cache** — ~11× an 80 GB GPU just to *hold* the context (tens of $/hr of HBM), or it simply
does not fit; most models cap context far below 15M anyway.

### The honest scope of that cost — read before quoting it

- The saving is **not cheaper generation.** The fused system generates at **transformer speed
  (~120 ms/token)**, unchanged. It is flat with *stream length* only because the transformer is
  fed a *bounded short context* (retrieved facts + query) while the memory lives in the flat store.
- The saving comes from two things: **the model is invoked only on flagged sentences** (it sleeps
  through filler), and **there is no KV-cache.** This applies to **sparse-fact long-context** work.
  A task that must densely read or generate over *every* token gets none of it.
- **RSS is a range, not a fixed number** — ~2.6–4.8 GB depending on phase and GPU-buffer state
  (store is flat KB, weights ~1.3 GB fixed, total process RSS swings ~2×).
- **Semantic retrieval is not replaced** — fuzzy/paraphrase matching needs an embedder (float
  vectors + cosine), which *is* a vector DB. The integer store replaces an *exact-key* store (a
  dict already is one), not semantic search.
- **Reasoning is the model's** — the multi-hop "resolve" is a deterministic graph traversal over
  extracted facts; the transformer does the extraction and the language.

**The claim that survives every rerun:**

> Context length is decoupled from memory — 0-byte KV-cache growth, O(1) integer lookups,
> ~120 ms/token generation flat with stream length — so a million-token sparse-fact query runs in
> flat kilobytes on a laptop for a fraction of a cent, where a transformer would need hundreds of
> GB of KV-cache. The transformer weights and generation compute are unchanged; this fixes the
> **context-memory** cost, not inference cost.

## "It only works if you ask the exact question" — the honest answer

This is the most common — and most correct — objection, so it gets a direct answer.

**For the integer memory alone, it is true.** The count-table is keyed on the exact token
sequence. Ask it a reworded question and it correctly returns *nothing* — it never guesses.
That is a deliberate property (deterministic, never-bluffing recall you can audit), and on
its own it does demand the exact question. If that were the whole system, the objection
would be fatal.

**It isn't the whole system.** Exact-key recall and messy-language matching cannot come
from the *same* component — an exact key is lossless but rigid; a semantic key is flexible
but a float with failure modes. So you keep both, as three layers:

1. **Integer memory** — exact, deterministic, O(1), never bluffs. *(this repo)*
2. **A semantic retriever** — a small sentence-embedding model that maps a *messy* query to
   the *meaning* of a stored fact.
3. **The language model** — reads the retrieved fact(s) and answers in natural language,
   including multi-hop reasoning.

### Measured, on deliberately broken queries

*(Separate harness from `live.py`: a 7B model + a sentence embedder, target facts stored
among 14 distractors. Reworded/indirect/multi-hop queries — never the "expected" one.)*

| Stored fact | Query (never the expected phrasing) | Result |
|---|---|---|
| *Arthur gave the asset deed to Beatrice.* | "walk me through what happened to that real estate paperwork after **Artie** got his hands on it — did it change owners?" | ✅ "…changed hands to **Beatrice**" |
| *…update failed at 04:00 — unhandled null pointer in the **authentication gateway** module.* | "the **login screen** broke this morning, why did it go down?" | ✅ "**null pointer** in the **authentication gateway**" |
| *Charles travels only on vehicles built by **Boeing**.* + *the airline bought a fleet of **737**s.* | "can **Charles** book a flight with that new airline?" | ✅ "**Yes**" (deduced 737 → Boeing → Charles flies Boeing) |

Exact-key memory alone: **0/3**. Fused (retriever + model): **3/3**.

### What this does *not* claim

- The messy-query matching is done by the **embedder + language model**, **not** the
  integer memory. The deterministic core contributes **zero** to fuzzy matching; it is the
  auditable store, not the semantic bridge. Credit goes to the right layer.
- The retriever is a **float similarity with thin margins.** In the login example the
  correct fact scored **0.25 — tied with a distractor**; it ranked first by a hair. With
  more distractors or a worse-phrased query it can be outranked and silently return the
  wrong fact. It also has known failure modes (negation: "the capital is *not* X" can fire
  the wrong match).
- So the honest claim is: **reworded, indirect, and multi-hop queries work — reliably when
  retrieval fires, which is most of the time but not always.** Strong, not infallible.

**In one line:** the objection is right about the *core* — and the *system* is built
precisely to fix it, by putting a semantic retriever and a language model in front of the
deterministic memory, at the cost of that front layer being a float with real failure
modes rather than the lossless integer core.

## Honest scope — read before citing

- These are **architecture-level demonstrations on synthetic streams** — not product
  benchmarks. No product claims are made. The core demo (`live.py`) trains ~0.1M
  parameters from scratch; the scale demos bolt the *same* organ onto frozen, pretrained
  124M–1.24B models without training any new weights.
- The living organ is a *fact memory*. It does not add reasoning, and on soft
  statistical prediction (prose) a naive fixed blend can *hurt* — organs specialize.
- Single-pass nested composition (answer f(g(x)) in one step) is **not** in this
  model's repertoire; it composes by chaining lookups across steps.
- Per-token cost is flat because the attention window is fixed and the organ is O(1)
  per update. The frozen twin's failures are structural (facts leave its window),
  which is exactly the point being demonstrated.

## Is this a world-first? — the honest answer

Short version: **the goal is not new; this particular combination might be.** I would
rather tell you the truth than sell you a "world-first."

Memory-augmented language models already exist, and several are close in spirit:

- **kNN-LM** (Khandelwal et al., 2020) — a frozen LM interpolated with a nearest-neighbor
  datastore of context→next-token. The closest prior art to what this does.
- **Memorizing Transformers** (Wu et al., 2022) — kNN lookup into a memory of past
  keys/values to extend effective context.
- **RAG / vector-DB retrieval** (Lewis et al., 2020) — retrieve relevant text and
  condition generation on it.
- **Memory Networks** (Weston/Sukhbaatar, 2015), **Neural Turing Machines / DNC**
  (Graves et al., 2014/2016), **fast-weights** (Ba et al., 2016) — external or
  fast-changing memory, going back years.
- **Knowledge-editing** (ROME/MEMIT/SERAC) *does* revise facts, and NTM/DNC have
  overwrite gates — but they edit float weights or use learned retrieval, are not
  integer-deterministic, and are not online O(1) per token.

So *"a frozen model + external memory blended at read time"* is **not** first-of-its-kind.
If a post tells you otherwise — including mine — be skeptical.

What I have **not** found in any single existing system is a drop-in memory that is *all
four of these at once*:

1. **Deterministic and integer** — the fact-table is byte-reproducible *by construction*
   (integer arithmetic is machine-independent, unlike float); demonstrated here as
   byte-identical across independent runs on the same machine. (kNN-LM and vector DBs use
   float similarity search; they are not built to be bit-reproducible.)
2. **Register semantics** — a later write cleanly *overrules* an older one, so a fact can
   be **revised**, not just retrieved. Pure similarity retrieval cannot do this (the stale
   neighbor still votes); float weight-editing (ROME/MEMIT) can, but not deterministically
   or online-per-token.
3. **Zero-forgetting by construction** — disjoint per-context rows; learning fact A
   *physically cannot* touch fact B. Not "rarely" — structurally.
4. **Online at inference, constant per-token cost, CPU-only** — no retraining, no
   re-indexing, flat cost as context grows.

In three words, it is an **adaptive, deterministic, recurrent** memory — and the
load-bearing pair is the first two. Online *adaptivity* (learning at inference) is almost
always stochastic and float, so it is not reproducible; making it *deterministic*
(bit-exact) at the same time is the unusual move. *Recurrent* is the shape: it carries and
updates a persistent state token by token, rather than re-reading a fixed window.

Each property exists somewhere on its own. The part I haven't seen in one system is the
**intersection**: adaptive **+** deterministic **+** clean-revision **+** structural-no-forget
**+** online-on-CPU. That — not "world-first" — is the honest claim. **If you know of a
system that does all four, please open an issue with the link; I'll add it here.** That is
not rhetorical: being wrong about this is worth knowing, and I would rather find out than
pretend.

## License

[AGPL-3.0](LICENSE). You may use, modify, and serve this freely — but any modified
or network-served version must be open-sourced under the same terms. For a
commercial license without AGPL obligations, contact the author.
