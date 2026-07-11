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

## License

[AGPL-3.0](LICENSE). You may use, modify, and serve this freely — but any modified
or network-served version must be open-sourced under the same terms. For a
commercial license without AGPL obligations, contact the author.
