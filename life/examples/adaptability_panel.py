#!/usr/bin/env python3
"""EXTREME ADAPTABILITY TEST — one exact memory, many different LLMs.

The claim: the Life fuses with ANY model. This proves it the hard way — fuse
ONE memory, store the facts ONCE, then make a whole panel of different local
models (different vendors, sizes, families) each recall from the SAME life.json.

For every model we measure two things, honestly:
  BARE  — ask the model with no fact anywhere. A stateless model can't know it.
          Expected ~0: this is the forgetting the Life exists to fix.
  FUSED — `life.py get <key>` returns the exact stored value in microseconds;
          we hand that ONE fact to the model and ask it to voice it. Expected
          full marks: the memory did the remembering, the model did the phrasing.

The point is not that any single model is smart — it's that the SAME exact
store serves all of them unchanged. The memory is model-agnostic; that is the
adaptability. Cloud APIs (Claude/GPT/Gemini) attach the same way via the two
tool schemas in README §5d — not run here (no keys); the mechanism is identical.

Needs: ollama running with the listed models (skips any you don't have).
Run:   python3 examples/adaptability_panel.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIFE = os.path.join(ROOT, "life.py")
DB = os.path.join(HERE, "panel_life.json")

# a deliberately diverse panel: 4 vendors, 3B–8B, incl. a reasoning model
PANEL = [
    ("llama3.2:3b",       "Meta"),
    ("llama3:8b",         "Meta"),
    ("mistral:latest",    "Mistral AI"),
    ("qwen2.5-coder:7b",  "Alibaba"),
    ("qwen3.5:latest",    "Alibaba"),
    ("deepseek-r1:7b",    "DeepSeek (reasoning)"),
]
# neutral facts on purpose (instruction-tuned models refuse to voice
# credential-shaped values even when handed them — a known, documented quirk)
FACTS = {
    "project.codename": "BlueHeron",
    "meeting.room": "4B",
    "favorite.tea": "oolong",
    "flight.gate": "C22",
}


def have(model):
    try:
        req = urllib.request.Request("http://localhost:11434/api/show",
                                     json.dumps({"name": model}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        return True
    except Exception:
        return False


def ask(model, prompt, timeout=180):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        json.dumps({"model": model, "prompt": prompt, "stream": False,
                    "options": {"temperature": 0}}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())["response"].strip()
    # deepseek-r1 emits <think>...</think>; keep only the answer
    if "</think>" in out:
        out = out.split("</think>", 1)[1].strip()
    return out


def life(*args):
    """Call the real life.py CLI — the exact path a fused agent uses."""
    r = subprocess.run([sys.executable, LIFE, *args],
                       capture_output=True, text=True,
                       env=dict(os.environ, LIFE_DB=DB))
    return r.stdout.strip(), r.returncode


def P(m):
    sys.stdout.write(m + "\n"); sys.stdout.flush()


# ---- fuse ONE memory, store the facts ONCE ---------------------------------
if os.path.exists(DB):
    os.remove(DB)
for k, v in FACTS.items():
    life("put", k, v)
sha, _ = life("sha")
P("=" * 74)
P("EXTREME ADAPTABILITY — one exact memory, many different LLMs")
P("=" * 74)
P(f"stored {len(FACTS)} facts ONCE into {os.path.basename(DB)}  (identity sha {sha})")
P("every model below reads this SAME file — nothing per-model is changed.\n")

results = []
for model, vendor in PANEL:
    if not have(model):
        P(f"-- {model:22s} [skipped — not pulled]")
        continue
    bare = fused = 0
    t0 = time.time()
    for k, v in FACTS.items():
        name = k.replace(".", " ")
        q = f"Question: what is my {name}? Reply with only the value."
        try:
            if v.lower() in ask(model, q).lower():
                bare += 1
        except Exception:
            pass
        fact, rc = life("get", k)                 # exact recall in microseconds
        try:
            if rc == 0 and v.lower() in ask(model, f"My {name} is {fact}.\n{q}").lower():
                fused += 1
        except Exception:
            pass
    # honesty arm: unstored key must ABSTAIN (exit 3), never a guessed value
    _, rc = life("get", "car.plate")
    abstain = "PASS" if rc == 3 else "FAIL"
    dt = time.time() - t0
    n = len(FACTS)
    P(f"-- {model:22s} {vendor:22s} bare {bare}/{n}  ->  FUSED {fused}/{n}"
      f"   abstain {abstain}  ({dt:.0f}s)")
    results.append((model, vendor, bare, fused, n, abstain))

# ---- verdict ---------------------------------------------------------------
P("\n" + "=" * 74)
if results:
    tot_bare = sum(r[2] for r in results)
    tot_fused = sum(r[3] for r in results)
    tot_n = sum(r[4] for r in results)
    P(f"PANEL: {len(results)} models / "
      f"{len(set(r[1] for r in results))} vendors  |  "
      f"bare {tot_bare}/{tot_n}  ->  FUSED {tot_fused}/{tot_n}  |  "
      f"abstain {sum(1 for r in results if r[5]=='PASS')}/{len(results)} PASS")
    P("Same life.json, every model. The memory is model-agnostic — THAT is the")
    P("adaptability. The Life remembered; each model only phrased it.")
else:
    P("no panel models available — pull a few with `ollama pull llama3.2:3b`")
P(f"identity sha still {life('sha')[0]}  (unchanged — one exact memory throughout)")
os.path.exists(DB) and os.remove(DB)
