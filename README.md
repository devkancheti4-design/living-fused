# living-fused

**A model that never stops learning. One command starts a life on your machine.**

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
