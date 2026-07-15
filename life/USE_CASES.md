# Where this is actually useful

Two use-cases with measured backing and their honest limits. No dollar figures,
no "trillion-dollar" — those died four times in this project's history under
fair testing, and the [proof](proof/PROOF.md) makes no commercial claim on
purpose. What follows is only what the code demonstrates.

## 1. A silent-error checker (the strongest one)

**The failure it targets is measured.** Under multi-fact load a model returns
another fact's value — confidently, with no signal it's wrong. Reproduced in
[`proof/experiments/exp14`](proof/experiments/exp14_system_catches_swaps.py) on
llama3.2:3b: **188/192 correct = 3 silent, confident swaps.** Softmax can't flag
them (that's [Pillar III](proof/PROOF.md)).

**The fix, measured.** Where you hold a ground-truth value, an exact store
cross-checks every answer in O(1) and corrects the deviation — no second model,
no extra tokens:

```
exp14:          MODEL ALONE 188/192  ->  + exact cross-check 192/192   (3 swaps caught)
examples/checker.py (8 similar-key facts, run live):
                MODEL ALONE   3/8    ->  + CHECKER 8/8   (5 deviations caught + corrected)
```

Run it: `python3 examples/checker.py` (needs ollama). It stores invoice amounts,
ICD codes, and case citations, asks the model all at once, and catches every
answer that doesn't match the stored truth — swaps *and* dropped/reformatted
values.

**What it is:** a verification layer between the model and the user that turns a
confident-wrong answer into a correct one for the cost of a dict lookup.

**The bill (honest, and it's the whole limit):** it only works **where a
ground-truth value exists to check against** — IDs, codes, settled numbers,
canonical facts. It is a **checker, not a brain.** It cannot judge an answer it
has no truth for (an opinion, a novel inference, a summary). Where there's no
truth source, it does nothing. So this is not "make the LLM correct" — it's
"eliminate the silent-wrong tail on the subset of answers you can verify."

## 2. A persistence organ for agents

**Measured, red-teamed properties** (see [`smoke_test.py`](smoke_test.py) and
the project dossiers): byte-exact across process death (cross-process SHA match),
O(1) recall, ~330–410 B/fact, and — from earlier federation work — count-merge
across devices is bit-exact and order-independent.

**What it is:** the state layer under an agent that survives context end, a
reboot, and a machine move without drift.

**The bill (honest):** this is a **crowded lane** — Mem0, Letta, Zep, LangMem
all sell "agent memory," and most are RAG + summarisation. On semantic recall
they beat this ([BENCHMARK.md](BENCHMARK.md): RAG 6/10 paraphrase vs 0/10 here).
The edge here is **not "memory" generically** — it's the specific guarantees:
**exactness, byte-exact determinism, and cross-device merge.** You'd win on
"provably the same bits everywhere, forever," not on being a better memory in
general. Where those guarantees aren't required, use the incumbent.

## Not making these repo claims (yet)

Two further ideas came up — a *verified-knowledge ledger* (verify a fact once,
reuse forever) and *owning a private verifiable stream* (a domain where reality
auto-checks and you own the checking loop). They are **strategy directions, not
capabilities** — there is no code or measurement behind them here, and each has
a fatal-until-solved bill (a public ledger copies perfectly → no moat; a private
stream requires actually holding one). They are deliberately **not** on the repo
as product claims. If they ever get a running check and a measurement, they earn
a section like the two above — and not before.
