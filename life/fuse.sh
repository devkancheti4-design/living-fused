#!/bin/bash
# fuse.sh — ONE command to fuse the Life with whatever model this device has.
#
#   bash life/fuse.sh [TARGET_DIR]     # default target: the directory you run it from
#
# What it does, in order (all visible, nothing silent):
#   1. SMOKE TEST the memory organ (16 deterministic checks). Abort on any failure.
#   2. Install life.py into TARGET_DIR.
#   3. Write the memory protocol into TARGET_DIR/CLAUDE.md and TARGET_DIR/AGENTS.md
#      (append-only, marker-guarded, never clobbers your existing content).
#      -> Claude Code / Cursor / Codex sessions in that directory are fused
#         AUTOMATICALLY: the agent reads the protocol on startup. No pasting.
#   4. Detect the model backends on this device and print how each one fuses.
#   5. Prove the round-trip: put a fact, get it back verbatim, show ABSTAIN.
#
# Honest scope: "fused" means the agent in TARGET_DIR now stores facts exactly,
# recalls them verbatim, and abstains instead of guessing — via the protocol.
# Logit-level fusion (blend into the softmax) is examples/fuse_logits.py and
# needs a local model with readable logits. This script never installs anything,
# never touches the network, and only writes: life.py, CLAUDE.md, AGENTS.md,
# life.json inside TARGET_DIR.
set -e
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then echo "ERROR: Python 3 not found. Install python.org/downloads then re-run."; exit 1; fi
SRC="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$PWD}"
mkdir -p "$TARGET"
TARGET="$(cd "$TARGET" && pwd)"

echo "==================================================================="
echo " THE LIFE — one-command fuse        target: $TARGET"
echo "==================================================================="

# ---- 1. smoke test the organ (never fuse an organ that fails) -------------
echo ""
echo "[1/5] smoke test (16 deterministic checks)..."
if "$PY" "$SRC/smoke_test.py" > /tmp/life_smoke_$$.log 2>&1; then
  grep -E "^RESULT" /tmp/life_smoke_$$.log | sed 's/^/      /'
else
  echo "      SMOKE TEST FAILED — refusing to fuse. Full log:"
  cat /tmp/life_smoke_$$.log
  rm -f /tmp/life_smoke_$$.log
  exit 1
fi
rm -f /tmp/life_smoke_$$.log

# ---- 2. install the organ --------------------------------------------------
echo ""
echo "[2/5] installing life.py -> $TARGET/life.py"
if [ "$SRC/life.py" != "$TARGET/life.py" ]; then cp "$SRC/life.py" "$TARGET/life.py"; fi

# ---- 3. write the protocol (Claude Code + Cursor/Codex auto-load these) ----
echo ""
echo "[3/5] writing the memory protocol (marker-guarded, append-only)"
PROTO='<!-- LIFE-MEMORY BEGIN (installed by fuse.sh — edit between markers only if you know why) -->
# Exact memory — the Life

This project has an exact, permanent, honest external memory: `./life.py`
(stdlib Python, zero install). The memory file is `./life.json`. Rules:

1. **STORE** — when the user states a fact worth keeping (name, number, path,
   decision, preference), run:  `python3 life.py put "subject.attribute" "exact value"`
   (lowercase dot-notation keys, e.g. `wifi.password`, `server.ip`)
2. **RECALL** — before answering anything that depends on a stored fact, run:
   `python3 life.py get "subject.attribute"`
   - prints a value -> use it **VERBATIM**, never paraphrase a stored value
   - prints `ABSTAIN` (exit 3) -> say it is not stored. **NEVER guess a stored fact.**
3. **REVISE** — `put` again with the same key; newest wins, history kept
   (`python3 life.py history "key"`).
4. **RELATE** — `python3 life.py link "a" "b"`, follow with `python3 life.py chain "a"`.
5. `./life.json` outlives every conversation, context window, and process.
   Never edit it by hand, never delete it.
<!-- LIFE-MEMORY END -->'
for F in CLAUDE.md AGENTS.md; do
  if [ -f "$TARGET/$F" ] && grep -q "LIFE-MEMORY BEGIN" "$TARGET/$F"; then
    echo "      $F: protocol already present — left untouched (idempotent)"
  else
    printf '\n%s\n' "$PROTO" >> "$TARGET/$F"
    echo "      $F: protocol installed"
  fi
done

# ---- 4. detect what this device can fuse with ------------------------------
echo ""
echo "[4/5] model backends on this device:"
if command -v claude >/dev/null 2>&1; then
  echo "      Claude Code        FUSED — any 'claude' session started in $TARGET"
  echo "                         auto-reads CLAUDE.md. Nothing more to do."
else
  echo "      Claude Code        not found — install it, or paste life/README.md §1"
  echo "                         into any agent; the protocol is the same."
fi
if curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "      ollama             UP — two-session demo:"
  echo "                         python3 $SRC/examples/agent_ollama.py --model <your-model>"
else
  echo "      ollama             not running — local-model demo unavailable right now"
fi
if python3 -c "import transformers" >/dev/null 2>&1; then
  echo "      transformers       present — logit-level fusion (blend into softmax):"
  echo "                         python3 $SRC/examples/fuse_logits.py"
else
  echo "      transformers       not installed — logit fusion optional (pip install transformers torch)"
fi
echo "      API models         Claude / GPT / Gemini function-calling: two tool"
echo "                         schemas in life/README.md §5d route to this same file."

# ---- 5. prove the round-trip in the target ---------------------------------
echo ""
echo "[5/5] live round-trip in $TARGET:"
cd "$TARGET"
"$PY" life.py put "fuse.check" "alive-and-exact" >/dev/null
V=$("$PY" life.py get "fuse.check")
M=$("$PY" life.py get "fuse.never.stored" || true)
"$PY" life.py forget "fuse.check" >/dev/null
echo "      put fuse.check -> get: '$V'   (verbatim: $( [ "$V" = "alive-and-exact" ] && echo OK || echo FAIL ))"
echo "      get unstored key     -> '$M'   (structural abstention: $( [ "$M" = "ABSTAIN" ] && echo OK || echo FAIL ))"
[ "$V" = "alive-and-exact" ] && [ "$M" = "ABSTAIN" ] || { echo "      ROUND-TRIP FAILED"; exit 1; }

echo ""
echo "==================================================================="
echo " FUSED. The agent in this directory now remembers exactly,"
echo " forever, and never bluffs about what it wasn't told."
echo " Memory file: $TARGET/life.json — yours, local, inspectable."
echo "==================================================================="
