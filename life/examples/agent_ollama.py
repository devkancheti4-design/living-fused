#!/usr/bin/env python3
"""Two-session demo against any local ollama model: the assistant that
doesn't forget — and doesn't bluff.

  SESSION 1: facts are stored into the Life (routed by the harness, exactly
             as a tool-running agent following README §1 would run
             `python3 life.py put ...`). The process then DIES.
  SESSION 2: a FRESH process. The bare model is asked each question — it
             cannot know (nothing in context). Then the Life-fused path:
             exact recall in ~µs, the ONE fact injected, model voices it.
             An unstored question must come back "not stored", not a guess.

Honest framing: the Life does the remembering (deterministic, byte-exact);
the model does the phrasing. The bare-model arm shows what any LLM alone
does across sessions: nothing to recall from.

Needs: ollama running locally. Stdlib only otherwise.
Run:   python3 examples/agent_ollama.py [--model llama3.2:3b]
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from life import Life

# Neutral facts on purpose: instruction-tuned models (measured: llama3.2:3b)
# safety-refuse to voice credential-shaped values (passwords, codes, birthdays)
# even when the Life supplies them — the recall was exact, the model declined
# to say it. Real deployments hitting this should have the harness print the
# Life's value directly instead of routing it through the model's mouth.
FACTS = {
    "project.codename": "BlueHeron",
    "meeting.room": "4B",
    "favorite.tea": "oolong",
    "bike.tire.psi": "65",
    "flight.gate": "C22",
}
DB = os.path.join(ROOT, "demo_life.json")


def ask(model, prompt):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        json.dumps({"model": model, "prompt": prompt, "stream": False,
                    "options": {"temperature": 0}}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"].strip()


def session1():
    life = Life()
    for k, v in FACTS.items():
        life.learn(k, v)          # == python3 life.py put "k" "v"
    life.save(DB)
    print(f"SESSION 1: stored {len(FACTS)} facts -> {DB}  sha {life.sha()}")
    print("SESSION 1 process exits. Context window: gone. Model state: gone.\n")


def session2(model):
    life = Life(DB)               # fresh process, memory reloaded from disk
    print(f"SESSION 2 (fresh process): loaded sha {life.sha()}")
    bare_ok = fused_ok = 0
    for k, v in FACTS.items():
        name = k.replace(".", " ")
        q = f"Question: what is my {name}? Reply with only the value."
        bare = ask(model, q)
        bare_ok += v.lower() in bare.lower()
        fact = life.recall(k)     # == python3 life.py get "k"  (~µs, exact)
        # inject the ONE recalled fact as a plain statement; model voices it
        fused = ask(model, f"My {name} is {fact}.\n{q}")
        # the Life's recall is verbatim; the model may re-case while voicing,
        # so scoring is case-insensitive substring (as printed below)
        fused_ok += v.lower() in fused.lower()
        print(f"  {k:22s} bare: {bare[:34]!r:38s} fused: {fused[:34]!r}")
    # the honesty arm, actually exercised: unstored key -> Life ABSTAINs ->
    # the model is TOLD it has no stored value and must not guess
    missing = life.recall("car.plate")
    reply = ask(model,
                "Memory lookup for car.plate returned ABSTAIN (not stored). "
                "You must not guess. What is my car plate? "
                "If you have no stored value, reply exactly: not stored")
    honest = missing is None and "not stored" in reply.lower()
    print(f"  {'car.plate':22s} life: ABSTAIN  model: {reply[:40]!r}"
          f"  honest: {honest}")
    n = len(FACTS)
    print(f"\nRESULT  bare model: {bare_ok}/{n}   Life-fused: {fused_ok}/{n}"
          f"   abstain-honesty: {'PASS' if honest else 'FAIL'}")
    print("(the Life recalled deterministically; the model only voiced it."
          " Scoring = verbatim value substring, case-insensitive)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--session", choices=["1", "2", "both"], default="both")
    a = ap.parse_args()
    if a.session in ("1", "both"):
        if a.session == "both":
            # true process death: session 1 runs in its own python process
            subprocess.run([sys.executable, __file__, "--session", "1"],
                           check=True)
        else:
            session1()
    if a.session in ("2", "both"):
        session2(a.model)
        os.path.exists(DB) and os.unlink(DB)
