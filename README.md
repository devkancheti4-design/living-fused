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
3. the organism take three exams **up to 150× beyond its training length**, scored
   live against its own **frozen twin** (identical weights, life off):

| exam | scale | ALIVE | frozen twin | chance |
|---|---|---|---|---|
| RULER-style: 10 variables + mid-stream **updates** | 32,000 tokens | **100%** | 8% | ~6% |
| chain of custody ("who holds it *now*?") | 64,000 tokens | **100%** | 62% | ~6% |
| code-dependency trace (corrupted var → use-site) | 20,000 tokens | **100%** | 12% | ~6% |

4. a **determinism receipt**: two identical lives produce **byte-identical**
   organisms (SHA-256 of the fact-tables printed);
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

## Scaling it up

To fuse the living organ onto a real transformer (or via an LLM's help), read
[INTEGRATION.md](INTEGRATION.md) FIRST — six small mistunes each cause major
degradation, and each one is documented there from a failed run we actually hit.

## Honest scope — read before citing

- These are **architecture-level demonstrations at ~0.1M parameters on synthetic
  streams** — not product benchmarks. No product claims are made.
- The living organ is a *fact memory*. It does not add reasoning, and on soft
  statistical prediction (prose) a naive fixed blend can *hurt* — organs specialize.
- Single-pass nested composition (answer f(g(x)) in one step) is **not** in this
  organism's repertoire; it composes by chaining lookups across steps.
- Per-token cost is flat because the attention window is fixed and the organ is O(1)
  per update. The frozen twin's failures are structural (facts leave its window),
  which is exactly the point being demonstrated.

## License

[AGPL-3.0](LICENSE). You may use, modify, and serve this freely — but any modified
or network-served version must be open-sourced under the same terms. For a
commercial license without AGPL obligations, contact the author.
