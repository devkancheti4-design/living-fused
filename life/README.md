# Life — exact, permanent, honest memory for any LLM

One Python file. Zero dependencies. Works with whatever model you already have —
Claude, GPT, Llama, Qwen, anything local or cloud. Your model keeps the reasoning;
the Life keeps the facts: **exact** (verbatim, not paraphrased), **permanent**
(survives the conversation, the context window, and the process), and **honest**
(on a fact it doesn't have, it says `ABSTAIN` — it cannot guess).

**Is it just a dict? Better than RAG?** → [BENCHMARK.md](BENCHMARK.md) — measured: ties a dict on exact recall, LOSES to RAG on paraphrase (0/10 vs 6/10), wins on determinism + guaranteed abstention. No spin.
**Works with any model?** → [ADAPTABILITY.md](ADAPTABILITY.md) — one memory, 6 models, 4 vendors, measured: bare 0/24 → fused 24/24.

**"Isn't it just a dict?"** → [WHAT_IS_LIFE.md](WHAT_IS_LIFE.md) — the straight
answer, including Claude's own honest comparison of itself against this file.
Every difference on that page is a command you can run, not a claim.

## 1. The prompt — this is the product

Get [`life.py`](life.py) into your working directory, then paste this into any
assistant that can run commands (Claude Code, Cursor, Codex, Aider, an agent
wrapper around a local model — anything):

```
You now have an exact external memory: ./life.py (one stdlib Python file, no install).
Follow these rules exactly:

1. STORE — when the user states a fact worth keeping (a name, number, path,
   decision, preference — anything they'd be annoyed to repeat), run:
     python3 life.py put "subject.attribute" "exact value"
   Use lowercase dot-notation keys, e.g. wifi.password, server.ip, mom.birthday.

2. RECALL — before answering anything that depends on a stored fact, run:
     python3 life.py get "subject.attribute"
   • It prints a value  -> use it VERBATIM. Never paraphrase a stored value.
   • It prints ABSTAIN  -> say you don't have that stored. NEVER guess a stored fact.

3. REVISE — store again with the same key; the newest value wins and the old
   ones stay inspectable via: python3 life.py history "key"

4. RELATE — python3 life.py link "a" "b" connects facts;
   python3 life.py chain "a" follows the whole dependency chain.

5. The memory is ./life.json. It outlives this conversation, your context
   window, and your process. Never edit it by hand, never delete it.

If you cannot run commands, print the exact command for the user to run and
ask them to paste the output back. The protocol is the same either way.
```

That's the whole integration. The prompt adapts to whatever model reads it:
a tool-running agent executes the commands itself; a chat-only model falls back
to printing them for you. Either way the model now has memory that never
paraphrases and never bluffs.

### Which "Claude" do you have? (this decides the setup)

| You have… | Can it run `life.py` itself? | What to do |
|---|---|---|
| **Claude Code** (terminal) | **Yes** — and it auto-reads `CLAUDE.md` | `bash life/fuse.sh .` (in living-fused) or drop `life.py` + the prompt above into your project's `CLAUDE.md`. Fully automatic after that. |
| **Cursor / Codex / Aider** | **Yes** | Same — the prompt in `CLAUDE.md`/`AGENTS.md`, or paste it once. |
| **Claude API** (your own code) | **Yes**, via tools | Register the two tool schemas in §5d; route them to `life.py`. |
| **claude.ai in a browser** | **No** — it can't touch your files | **Manual only:** you run `python3 life.py get "key"` in your terminal and paste the value into the chat; run `put` yourself when you want to store something. The exactness/abstention still hold — but you are the hands. |

### Is it actually working? — `python3 life.py doctor`

Run it inside your project to get a green/red checklist:

```bash
python3 life.py doctor
```
```
  [OK ] protocol installed in CLAUDE.md  -> Claude Code / Cursor here auto-use the Life
  [OK ] memory file life.json  2 facts, last written 12s ago
        newest-looking sample: 'server.ip' -> '192.168.7.203'
  [OK ] live round-trip  put/get verbatim=True, abstain-on-unknown=True
  RESULT: FUSED AND WORKING — the agent here can store & recall exactly.
```

If the protocol line is red, the agent won't use the Life yet — fuse it. The
**decisive** test that it's live: tell your agent a fact, run `python3 life.py
stats`, and watch the fact count go **up** — then ask for it back in a brand-new
session. If the count grew and the new session recalls it verbatim, it's working.
See [§9 How you KNOW it's working](#9-how-you-know-its-working-as-the-creator).

## 2. Install & smoke test

```bash
git clone https://github.com/devkancheti4-design/life-memory
cd life-memory
python3 smoke_test.py        # ~10 s, stdlib only, deterministic pass/fail
```

The smoke test proves the following claims on your machine: 10,000/10,000
exact recalls, revision with retained history, 1,000/1,000 abstentions on
unseen keys, byte-identical twins (golden SHA `0ffd7ccc8e97f01b` — the same on
your machine as on ours), recall across process death, 3-hop relational chains,
gated fusion math, and 100k-fact scale. (The larger-scale and model-fusion
numbers in §6 have their own provenance — labeled there.)

Try it by hand:

```bash
python3 life.py put "wifi.password" "SunFlower42"
python3 life.py get "wifi.password"     # -> SunFlower42  (in-process recall ~1 µs [HW];
                                        #    CLI round-trip ~30–100 ms: interpreter + file load)
python3 life.py get "wifi.pasword"      # -> ABSTAIN, exit 3 (exact-key: typos abstain, not guess)
python3 life.py sha                     # -> identity hash; twins match byte-exactly
```

## 3. Why it's not "just a dict"

Honest answer first: the storage core **is** an exact-key count table — on
purpose. The proof below shows exact lookup is precisely the property gradient
training cannot deliver, so the winning move is to keep it exact, not to make
it clever. What a plain dict does **not** give you:

| Property | dict | Life |
|---|---|---|
| Revision keeps full history (`history`) | ✗ overwrites | ✓ newest wins, history retained |
| Abstention wired into the model contract | `None`, unused | ✓ `ABSTAIN` → the model must say "not stored" instead of guessing |
| **Confidence-gated logit fusion** (`blend`) | ✗ | ✓ counts fold into the model's softmax; unseen keys leave the model untouched |
| Relational multi-hop retrieval (`chain`) | ✗ one value | ✓ follows dependency chains |
| Canonical identity (`sha`) | ✗ | ✓ byte-exact twins, byte-exact across process death |

The fusion row is the one that matters. Measured end-to-end with buried facts
in the parent project (this repo ships a 20-fact reproduction,
[`examples/fuse_logits.py`](examples/fuse_logits.py), not the original 40):
**bare model 0/40 · facts-in-context 36/40 · fused 40/40.** The fused system
ties the fair baseline and adds an exactness edge — because the table's counts
override the softmax on keys it knows and leave it alone on keys it doesn't.

## 4. Why an LLM needs it — the Required-Fusion Proof

[`proof/PROOF.md`](proof/PROOF.md) argues from measurement that an LLM+exact-store
fusion is not a convenience but the design the mathematics forces. Reproduce it:

```bash
bash proof/run_proof.sh          # pillars I–IV, ~2 min, stdlib only, deterministic
bash proof/run_proof.sh --llm    # pillar V on real local LLMs (needs ollama)
```

The five pillars, one line each (pillars 1–4 are deterministic and reproduce
bit-for-bit; pillar 5 is measured on real local LLMs via ollama [HW]):

1. **Exact recall is optimizer-inaccessible.** The training gradient vanishes
   proportionally to remaining error (measured over 5 orders of magnitude);
   on near-duplicate keys attention leaks 85% of its mass at cosine 0.99.
   The exact table: zero leak, zero training.
2. **No fixed-size differentiable memory substitutes.** Fast-weight recall
   cliffs at K = d writes (100% → 37%), and the damage is retroactive —
   new writes physically degrade the oldest memories (100% → 31%).
3. **Softmax cannot abstain.** A net at its generalizing best answers pure
   garbage with 0.974 mean confidence (36/40 above 90%). The table abstains
   60/60 — "I don't know" is structural, not trained.
4. **Writing facts into weights is conditionally destructive — decidably.**
   Agreement-bearing streams self-heal (75%→92%); conflicting streams collapse
   to 8% retention at the trough, ending at 42% (chance = 25%). The table holds
   100% in both regimes. That decidable split is the routing law of the fusion.
5. **It's behaviorally real in production models.** llama3.2:3b and llama3:8b
   at temperature 0 drop from 100% to ~88% per-fact recall as queried facts
   grow 4 → 64, and ~80% of the errors are another fact's value, confidently
   delivered. More parameters did not fix it.

The Life holds exactly the complementary corner — and *only* that corner
(see Honest limits).

## 5. Three ways to fuse

**a) Prompt fusion — any model (the prompt in §1).** Zero code. The model
runs `put`/`get` and obeys `ABSTAIN`. Works with cloud APIs and local models
alike; nothing about your facts enters any training set or context you didn't
choose.

**b) Logit fusion — local models.** [`examples/fuse_logits.py`](examples/fuse_logits.py)
shows `blend()` folding the table into a real model's next-token distribution —
the mechanism behind the 40/40. Needs a local model you can read logits from
(transformers, MLX, llama.cpp).

**c) Life + RAG + model — the full memory system.** The Life is exact-key
*only*. Measured in the parent project (no embedder ships in this repo): on 20
buried facts, entity-named queries hit 10/10 through the Life, but paraphrased
queries hit **0/10** — while embedding-based RAG got those 10/10. They're different organs: Life = exact/permanent/honest, RAG =
fuzzy/semantic, model = reasoning. [`examples/agent_ollama.py`](examples/agent_ollama.py)
runs the two-session demo (store facts, kill the process, fresh session
recalls) against any local ollama model.

**d) Tool calling — big API models (Claude, GPT, Gemini).** The same protocol
as §1, delivered as two tools instead of a prompt. Register these schemas with
any function-calling API:

```json
[{"name": "memory_put",
  "description": "Store or revise an exact fact the user states. Newest value wins; history is kept.",
  "input_schema": {"type": "object",
    "properties": {"key":   {"type": "string", "description": "lowercase dot-notation, e.g. wifi.password"},
                   "value": {"type": "string", "description": "the exact value, verbatim"}},
    "required": ["key", "value"]}},
 {"name": "memory_get",
  "description": "Exact recall before answering anything that depends on a stored fact. Returns the verbatim value, or ABSTAIN — if ABSTAIN, say the fact is not stored; never guess.",
  "input_schema": {"type": "object",
    "properties": {"key": {"type": "string"}},
    "required": ["key"]}}]
```

and route the calls to the file (whole handler, any language):

```python
import subprocess
def handle(tool, args):
    if tool == "memory_put":
        return subprocess.run(["python3", "life.py", "put", args["key"], args["value"]],
                              capture_output=True, text=True).stdout.strip()
    if tool == "memory_get":
        return subprocess.run(["python3", "life.py", "get", args["key"]],
                              capture_output=True, text=True).stdout.strip()  # value or "ABSTAIN"
```

The model does the reasoning; the file does the remembering. Your facts stay
in `life.json` on your disk — the API only ever sees the one fact the model
asked for.

## 6. Measured numbers

`[EXACT]` = deterministic, reproduces bit-for-bit on your machine.
`[HW]` = measured on an Apple M4 laptop; yours will differ.

| Measurement | Value | Label |
|---|---|---|
| Recall correctness, 10k facts | 10,000/10,000 verbatim | [EXACT] |
| Abstention on unseen keys | 1,000/1,000 (0 guesses) | [EXACT] |
| Twin determinism | SHA `0ffd7ccc8e97f01b` on any machine | [EXACT] |
| Recall across process death | byte-exact (cross-process SHA match) | [EXACT] |
| Fused recall vs fair baseline | bare 0/40 · in-context 36/40 · fused 40/40 | [EXACT protocol, HW model, parent project — repo ships a 20-fact version] |
| Recall latency @100k facts | ~1 µs in-process, O(1) flat | [HW] |
| Recall latency @15M facts | 0.2–1.4 µs (still flat) | [HW, measured in parent project — script not shipped] |
| Memory cost | ~330–410 B/fact RAM, ~35–42 B/fact on disk | [HW] |
| Store scale tested | 30M facts, 0 key collisions | [HW, parent project; note: exact string keys are collision-free *by construction*] |
| Paraphrase recall | Life 0/10 vs semantic RAG 10/10 | [EXACT protocol, HW embedder, parent project — Life loses this one; see §7] |
| Context-window arithmetic | 1M facts ≈ 6M tokens | [EXACT arithmetic] |
| Same facts as a Life store | ~330–410 MB RAM, one laptop | [HW projection from B/fact] |
| Exact recall at that scale | context windows that large exist, but per-fact recall degrades under multi-fact load (Pillar 5); the Life stays exact | [HW, Pillar 5] |

## 7. Honest limits — read before you rely on it

- **Exact-key only.** `wifi.pasword` (typo) abstains. A paraphrase ("what's
  that plant-named password?") scores **0/10** where semantic RAG scores 10/10.
  Pair with RAG if you need fuzzy retrieval.
- **It does not generalize and does not reason.** 0% held-out on unseen keys —
  by design. The model is the reasoner; the Life is the memory.
- **It ties, not crushes, the fair baseline.** A model *given* the facts
  in-context already gets ~90% (36/40). The Life's win is the last 10%
  exactness, plus permanence past the context window, plus honest abstention —
  not a new kind of intelligence.
- **Growth is O(K).** Exactness is bought with ~330–410 B/fact RAM [HW]. 1M
  facts ≈ 330–410 MB. That's the trade — the proof's Pillar II shows every
  fixed-size alternative pays in silent forgetting instead.
- **The CLI is the slow path.** Every CLI call is a fresh interpreter plus a
  whole-file load, and every CLI write rewrites the whole file — O(N) per
  operation (~110 ms per put at 100k facts [HW]). The µs numbers are
  in-process `recall()`. At large stores, use the `Life` class in-process;
  the CLI is for zero-integration convenience.
- **Some §6 numbers are inherited.** The 15M-latency, 30M-scale, 40-fact
  fused-recall, and paraphrase-vs-RAG rows were measured in the parent
  project; this repo's own smoke test stops at 100k and its fusion example
  at 20 facts. They're labeled as such — don't quote them as reproduced-here.
- **Determinism means byte-exact replay,** not correctness: store a wrong
  fact and it will recall the wrong fact, exactly, forever (until you revise).
- **No commercial-value claim.** This repo claims exactly what its tests
  measure, nothing more.

## 9. How you KNOW it's working (as the creator)

You built it — so don't trust it, *observe* it. Three signals, in order of
strength:

1. **`python3 life.py doctor`** — the checklist above. Green protocol line =
   the agent is wired to use it. Green memory line with a rising fact count =
   it's actually being used.
2. **The fact count grows as you talk.** Store nothing manually. Just use your
   agent normally, then run `python3 life.py stats` before and after a session.
   If the agent is honoring the protocol, `keys N` climbs — each new fact the
   user stated is now on disk. If it never moves, the agent isn't calling
   `put` (protocol not loaded, or the model ignored it — check `doctor`).
3. **The cross-session recall test** — the one that can't be faked:

   ```bash
   # session 1
   python3 life.py put "launch.date" "March 14"
   # now literally quit / close the terminal / reboot, then:
   # session 2 (fresh process, days later)
   python3 life.py get "launch.date"     # -> March 14   (verbatim, or it's broken)
   python3 life.py get "launch.tim"      # -> ABSTAIN     (typo abstains, never guesses)
   ```

   If session 2 returns `March 14` and the typo returns `ABSTAIN`, the Life is
   doing its whole job: exact recall across process death, and honest silence
   on what it doesn't have. If a browser-based assistant instead *guesses* a
   date when you didn't wire the Life in — that's the exact failure the Life
   exists to remove, and proof you need the wiring, not just the file.

**One honest caveat for the browser case:** in **claude.ai** the model can't run
`life.py`, so "working" means *you* ran `get` and pasted the value — the model
can't prove it used the Life because it never touched it. Only Claude Code /
Cursor / API can make the fusion automatic and self-evident in `doctor`.

## 8. License

MIT. One file. Take it, fuse it.
