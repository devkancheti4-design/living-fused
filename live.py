#!/usr/bin/env python3
# living-fused — a model that never stops learning
# Copyright (C) 2026 Kancheti Devieswar
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE).
"""
Train a tiny model, switch on an integer memory that keeps learning at inference,
and watch it beat a frozen copy of itself on long-context tasks far beyond its
training length.

    git clone <this repo> && cd living-fused && python3 live.py

Runs on CPU, ~10-15 min on an Apple-silicon Mac. What it does, in order:

  1. Train a small "body" from scratch: a sliding-window transformer plus a
     recurrent state. Trained ONLY on streams of <= 420 tokens.
  2. Switch on the memory: an integer count-table that updates on every token the
     model reads. There is no separate train/serve step — it keeps learning.
  3. Three long-context tests, each scored against a frozen twin (same weights,
     memory off):
       T1  10 variables + mid-stream updates, at 32,000 tokens
       T2  a value that changes hands three times, asked at 64,000 tokens
       T3  a variable set, corrupted, then read 17,000 tokens later, at 20,000
  4. Determinism check: two independent runs -> a byte-identical table (SHA-256).
  5. Cost: ms/token at 1K vs 32K context (should be flat).

Fixed seeds, so your numbers should match:
  memory on:  100% / 100% / 100%     frozen twin: ~8% / 62% / 12%   (chance ~6%)

Honest scope: this shows architecture-level properties at ~0.1M params on synthetic
streams. It is not a product benchmark and makes no product claims.
"""
import time
import hashlib
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("cpu")

# --- token vocabulary (40 symbols) --------------------------------------------
VOCAB = 40                     # total number of token ids
WINDOW = 32                    # the transformer only attends this many tokens back
KEYS = list(range(10))         # ids 0-9   : "variable names"
VALUES = list(range(10, 26))   # ids 10-25 : the values a variable can hold
SET_TOKEN = 26                 # marks "the next two tokens are (key, value)"
QUERY_TOKEN = 27               # marks "the next token asks for a key's current value"
NOISE_LO, NOISE_HI = 28, 40    # ids 28-39 : filler / distractor tokens
CHANCE = 100.0 / len(VALUES)   # accuracy a blind guesser would get (~6%)


def sliding_window_mask(length):
    """Attention mask: each position sees itself and the WINDOW-1 tokens before it."""
    rows = torch.arange(length)[:, None]
    cols = torch.arange(length)[None, :]
    mask = torch.full((length, length), float("-inf"))
    visible = (cols <= rows) & (rows - cols < WINDOW)
    mask[visible] = 0.0
    return mask


class FusedBody(nn.Module):
    """The trainable body: a sliding-window transformer plus a recurrent state.
    After training we freeze it — every bit of adaptivity at inference then comes
    from the memory below, not from these weights."""

    def __init__(self, d_model=64, d_state=96):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.embed = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(WINDOW, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=2 * d_model,
            dropout=0.0, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.recurrent = nn.GRUCell(d_model, d_state)
        self.head = nn.Linear(d_model + d_state, VOCAB)

    def forward_train(self, tokens):
        """Full-sequence forward, used only during training."""
        batch, length = tokens.shape
        positions = torch.arange(length) % WINDOW
        attended = self.transformer(self.embed(tokens) + self.pos(positions),
                                    mask=sliding_window_mask(length))
        state = torch.zeros(batch, self.d_state)
        embedded = self.embed(tokens)
        states = []
        for t in range(length):
            state = self.recurrent(embedded[:, t], state)
            states.append(state)
        return self.head(torch.cat([attended, torch.stack(states, 1)], dim=-1))


class Life:
    """The living memory: an integer count-table over (previous token -> next token).

    It keeps learning at inference — every token the model reads updates it — and:
      - a write outvotes everything before it, so the newest fact wins (revision);
      - one key's row is never touched by writes to another key (no forgetting);
      - a question (QUERY_TOKEN) is not a fact, so it never writes;
      - it is all integer, so two identical runs give a byte-identical table.
    """

    def __init__(self):
        self.counts = np.zeros((VOCAB, VOCAB), dtype=np.int64)

    def blend(self, prev_token, model_probs):
        """Mix the memory's vote for the next token into the model's probabilities.
        Confidence grows with how much evidence the row holds; an empty row defers
        entirely to the model."""
        row = self.counts[prev_token]
        total = int(row.sum())
        if total <= 0:
            return model_probs
        weight = total / (total + 0.25)
        return weight * (row / total) + (1 - weight) * model_probs

    def learn(self, prev_token, next_token):
        """Record that next_token followed prev_token. Adding (row.sum() + 1) makes
        the newest write dominate the row, so a later value overrides an earlier one."""
        if prev_token == QUERY_TOKEN or next_token == QUERY_TOKEN:
            return  # questions are not facts, so they never write
        self.counts[prev_token][next_token] += self.counts[prev_token].sum() + 1
        if self.counts[prev_token].sum() > (1 << 40):
            self.counts[prev_token] >>= 20  # rescale to avoid overflow (still deterministic)

    def sha(self):
        return hashlib.sha256(self.counts.tobytes()).hexdigest()[:16]


def make_stream(length, num_keys, rng):
    """Build one training stream: filler noise with (SET key value) facts planted in
    it, then (QUERY key) questions at the end. Returns the stream and, for each
    question, the position to score and the correct answer."""
    stream = rng.integers(NOISE_LO, NOISE_HI, length).astype(np.int64)
    keys = list(rng.choice(KEYS, num_keys, replace=False))
    tail = 4 * num_keys + 4

    # plant a few SETs per key at random positions
    events = []
    for k in keys:
        for _ in range(int(rng.integers(2, 4))):
            pos = int(rng.integers(2, length - tail - 3))
            val = int(rng.choice(VALUES))
            events.append((pos, k, val))
    for pos, k, val in sorted(events):
        stream[pos:pos + 3] = [SET_TOKEN, k, val]

    # a key's true value is its LAST SET before the questions
    truth = {}
    for pos in range(length - 2):
        if stream[pos] == SET_TOKEN and stream[pos + 1] in KEYS and stream[pos + 2] in VALUES:
            truth[int(stream[pos + 1])] = int(stream[pos + 2])

    # append the questions in the tail
    ask_at = length - tail
    answers = []
    for k in keys:
        if k not in truth:
            continue
        stream[ask_at:ask_at + 3] = [QUERY_TOKEN, k, truth[k]]
        answers.append((ask_at + 2, truth[k]))
        ask_at += 3
    return stream, answers


def train(model, steps=2400, batch_size=24):
    print("  [1/5] training the body (streams <= 420 tokens — remember this number)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(1)
    start = time.time()
    for step in range(steps):
        length = int(rng.integers(90, 420))
        streams, targets = [], []
        for _ in range(batch_size):
            num_keys = int(rng.integers(2, 6))
            stream, answers = make_stream(length, num_keys, rng)
            streams.append(stream)
            targets.append(answers)
        tokens = torch.tensor(np.stack(streams))
        logits = model.forward_train(tokens)

        losses = []
        for b, answers in enumerate(targets):
            for pos, value in answers:
                losses.append(nn.functional.cross_entropy(logits[b, pos - 1][None],
                                                          torch.tensor([value])))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 400 == 0:
            elapsed = time.time() - start
            eta = elapsed / (step + 1) * (steps - step - 1)
            print(f"        step {step:>4d}/{steps}  loss {loss.item():.3f}"
                  f"   elapsed {elapsed:>4.0f}s  eta {eta:>4.0f}s")

    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def walk(model, streams, answers, alive):
    """Read each stream one token at a time. When `alive`, blend the memory into the
    model's prediction at each question and learn from every non-question token."""
    batch = len(streams)
    length = len(streams[0])
    tokens = torch.tensor(np.stack(streams))
    window = torch.full((batch, WINDOW), NOISE_LO, dtype=torch.long)
    state = torch.zeros(batch, model.d_state)
    memories = [Life() for _ in range(batch)] if alive else None
    answer_at = [dict(a) for a in answers]
    correct = total = 0
    pos_emb = model.pos(torch.arange(WINDOW))
    win_mask = sliding_window_mask(WINDOW)

    for t in range(1, length):
        if any(t in a for a in answer_at):
            attended = model.transformer(model.embed(window) + pos_emb, mask=win_mask)[:, -1]
            logits = model.head(torch.cat([attended, state], dim=-1))
            probs = torch.softmax(logits, dim=-1).numpy()
            for b in range(batch):
                if t in answer_at[b]:
                    p = memories[b].blend(int(tokens[b, t - 1]), probs[b]) if alive else probs[b]
                    correct += int(int(p.argmax()) == answer_at[b][t])
                    total += 1

        current = tokens[:, t]
        if alive:
            for b in range(batch):
                memories[b].learn(int(tokens[b, t - 1]), int(current[b]))
        state = model.recurrent(model.embed(current), state)
        window = torch.cat([window[:, 1:], current[:, None]], dim=1)

    accuracy = correct / max(total, 1) * 100
    fingerprints = [m.sha() for m in memories] if alive else []
    return accuracy, fingerprints


def versus(name, streams, answers, model, note):
    """Score the same streams twice: memory on, then the frozen twin with it off."""
    start = time.time()
    alive_acc, fingerprints = walk(model, streams, answers, alive=True)
    frozen_acc, _ = walk(model, streams, answers, alive=False)
    print(f"\n  {name}\n      {note}")
    print(f"      ALIVE: {alive_acc:.0f}%      frozen twin: {frozen_acc:.0f}%"
          f"      chance {CHANCE:.0f}%      ({time.time() - start:.0f}s)")
    return alive_acc, frozen_acc, fingerprints


# =====================================================================================
print("=" * 74)
print("  THE LIVING FUSED MODEL — a life is about to start on your machine")
print("=" * 74)
assert next(FusedBody().parameters()).device.type == "cpu"
model = FusedBody().to(DEVICE)
print(f"  body: {sum(p.numel() for p in model.parameters()):,} params"
      f" | device cpu | torch {torch.__version__}")
train(model)

# --- T1: 10 variables, four of them overwritten mid-stream, asked at 32K -------------
print("\n  [2/5] the LIFE is now ON — it learns from every token it reads")
rng = np.random.default_rng(11)
streams, answers = [], []
for _ in range(8):
    length = 32_000
    stream = rng.integers(NOISE_LO, NOISE_HI, length).astype(np.int64)
    slots = rng.choice(np.arange(200, 14_000, 8), 10, replace=False)
    for key, pos in zip(KEYS, slots):
        val = int(rng.choice(VALUES))
        stream[int(pos):int(pos) + 3] = [SET_TOKEN, key, val]
    # overwrite four of the variables later in the stream
    for i, key in enumerate(rng.choice(KEYS, 4, replace=False)):
        val = int(rng.choice(VALUES))
        pos = 15_500 + 40 * i
        stream[pos:pos + 3] = [SET_TOKEN, int(key), val]
    # the answer for each key is its last SET
    truth = {}
    for pos in range(length - 46):
        if stream[pos] == SET_TOKEN and stream[pos + 1] in KEYS and stream[pos + 2] in VALUES:
            truth[int(stream[pos + 1])] = int(stream[pos + 2])
    ask_at = length - 44
    answer = {}
    for key in KEYS:
        if key in truth:
            stream[ask_at:ask_at + 3] = [QUERY_TOKEN, key, truth[key]]
            answer[ask_at + 2] = truth[key]
            ask_at += 3
    streams.append(stream)
    answers.append(answer)
a1, z1, fingerprints = versus("T1  RULER — 10 variables + mid-stream updates @ 32,000 tokens",
                              streams, answers, model,
                              "76x beyond training length; four values overwritten mid-stream")
print(f"      aliveness receipts (each stream changed its memory): {fingerprints[:3]} ...")

# --- T2: one value handed on at 1K / 20K / 50K, asked at 64K --------------------------
rng = np.random.default_rng(21)
streams, answers = [], []
for _ in range(8):
    length = 64_000
    stream = rng.integers(NOISE_LO, NOISE_HI, length).astype(np.int64)
    key = int(rng.choice(KEYS))
    chain = list(rng.choice(VALUES, 3, replace=False))
    for pos, val in zip([1_000, 20_000, 50_000], chain):
        stream[pos:pos + 3] = [SET_TOKEN, key, val]
    # distractor SETs for other keys
    for other in rng.choice([k for k in KEYS if k != key], 4, replace=False):
        for _ in range(int(rng.integers(1, 3))):
            pos = int(rng.integers(200, 60_000))
            if all(x >= NOISE_LO for x in stream[pos:pos + 3]):
                stream[pos:pos + 3] = [SET_TOKEN, int(other), int(rng.choice(VALUES))]
    truth = {}
    for pos in range(length - 46):
        if stream[pos] == SET_TOKEN and stream[pos + 1] in KEYS and stream[pos + 2] in VALUES:
            truth[int(stream[pos + 1])] = int(stream[pos + 2])
    ask_at = length - 10
    stream[ask_at:ask_at + 3] = [QUERY_TOKEN, key, truth[key]]
    streams.append(stream)
    answers.append({ask_at + 2: truth[key]})
a2, z2, _ = versus("T2  CHAIN OF CUSTODY @ 64,000 tokens",
                   streams, answers, model,
                   "the key changes hands at 1K / 20K / 50K; 'who holds it NOW?' at 64K")

# --- T3: a variable set, corrupted at 2.9K, read 17K tokens later ---------------------
rng = np.random.default_rng(31)
streams, answers, stale = [], [], []
for _ in range(8):
    length = 20_000
    stream = rng.integers(NOISE_LO, NOISE_HI, length).astype(np.int64)
    var = int(rng.choice(KEYS))
    first = int(rng.choice(VALUES))
    corrupted = int(rng.choice([v for v in VALUES if v != first]))
    stream[800:803] = [SET_TOKEN, var, first]
    stream[2900:2903] = [SET_TOKEN, var, corrupted]
    for other in rng.choice([k for k in KEYS if k != var], 3, replace=False):
        pos = int(rng.integers(4_500, 15_500))
        stream[pos:pos + 3] = [SET_TOKEN, int(other), int(rng.choice(VALUES))]
    ask_at = length - 6
    stream[ask_at:ask_at + 3] = [QUERY_TOKEN, var, corrupted]
    streams.append(stream)
    answers.append({ask_at + 2: corrupted})
    stale.append(first)
a3, z3, _ = versus("T3  CODE-DEPENDENCY TRACE @ 20,000 tokens",
                   streams, answers, model,
                   "variable set in file 1, CORRUPTED at 2.9K, consumed 17K tokens later")

# how often it wrongly returns the OLD value instead of the corrected one (0 = perfect)
returned_stale = 0
for b in range(8):
    stale_answer = {list(answers[b].keys())[0]: stale[b]}
    acc, _ = walk(model, [streams[b]], [stale_answer], alive=True)
    returned_stale += (acc == 100.0)
print(f"      stale check: returned the outdated value {returned_stale}/8 times (0 = perfect revision)")

# --- determinism: two independent lives over the same streams ------------------------
print("\n  [4/5] determinism receipt — two identical lives, lived independently")
_, sha_a = walk(model, streams[:2], answers[:2], alive=True)
_, sha_b = walk(model, streams[:2], answers[:2], alive=True)
print(f"      life #1 model: {sha_a}\n      life #2 model: {sha_b}")
print(f"      -> {'BYTE-IDENTICAL: the life is deterministic' if sha_a == sha_b else 'MISMATCH (please report!)'}")

# --- cost: per-token time should not grow with context length ------------------------
print("\n  [5/5] cost of being alive — ms/token at 1K vs 32K context")
for context in [1_000, 32_000]:
    noise = np.random.default_rng(3).integers(NOISE_LO, NOISE_HI, context).astype(np.int64)
    start = time.time()
    walk(model, [noise], [{}], alive=True)
    print(f"      ctx {context:>7,}: {(time.time() - start) / context * 1000:.3f} ms/token")

# --- scoreboard ----------------------------------------------------------------------
print("\n" + "=" * 74)
print("  SCOREBOARD                         ALIVE      frozen twin")
print(f"    T1 RULER+updates @32K       {a1:>8.0f}% {z1:>13.0f}%")
print(f"    T2 custody @64K             {a2:>8.0f}% {z2:>13.0f}%")
print(f"    T3 code trace @20K          {a3:>8.0f}% {z3:>13.0f}%")
print("  same body, same weights — the only difference is the life being on.")
print("=" * 74)
