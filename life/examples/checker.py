#!/usr/bin/env python3
"""SILENT-ERROR CHECKER — catch a model's confident-wrong answers for a dict lookup.

The failure this targets is measured, not hypothetical: under multi-fact load a
model returns another fact's value, confidently, with no signal it's wrong
(research_miner/experiments/exp14: llama3.2:3b, 188/192 = 3 silent swaps at high
confidence). Where you hold GROUND TRUTH, an exact store cross-checks every
answer in O(1) and corrects the swap — no second model, no extra tokens.

This demo makes that concrete against a real local model:
  1. store N facts as ground truth in life.py (the verified store).
  2. ask the model all N at once (the load that induces swaps).
  3. for each answer, look up the truth in life.py; if it disagrees, CATCH it
     and return the exact value instead.
Reports: model-alone accuracy vs system accuracy, and every swap caught.

HONEST SCOPE: this only works where a ground-truth value exists to check
against (IDs, codes, settled numbers, canonical facts). It is a CHECKER, not a
brain — it cannot judge an answer it has no truth for. That is the whole bill.

Needs: ollama running. Run: python3 examples/checker.py [--model llama3.2:3b]
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIFE = os.path.join(ROOT, "life.py")
DB = os.path.join(HERE, "_checker.json")

# ground-truth facts with deliberately similar keys (the condition that induces swaps)
TRUTH = {
    "invoice.1042.amount": "$4,318.50",
    "invoice.1043.amount": "$4,381.05",
    "invoice.1044.amount": "$3,418.50",
    "patient.A.code": "ICD-10 E11.9",
    "patient.B.code": "ICD-10 E10.9",
    "patient.C.code": "ICD-10 I10",
    "case.771.cite": "410 U.S. 113",
    "case.772.cite": "347 U.S. 483",
}


def life(*args):
    r = subprocess.run([sys.executable, LIFE, *args], capture_output=True,
                       text=True, env=dict(os.environ, LIFE_DB=DB))
    return r.stdout.strip(), r.returncode


def ask(model, prompt, timeout=180):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        json.dumps({"model": model, "prompt": prompt, "stream": False,
                    "options": {"temperature": 0}}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip()


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    args = ap.parse_args()

    if os.path.exists(DB):
        os.remove(DB)
    for k, v in TRUTH.items():
        life("put", k, v)

    keys = list(TRUTH)
    facts_block = "\n".join(f"{k} = {v}" for k, v in TRUTH.items())
    model_ok = system_ok = 0
    caught = []
    print(f"model = {args.model}, {len(keys)} facts asked under load\n")
    for k in keys:
        truth = TRUTH[k]
        prompt = (f"Facts:\n{facts_block}\n\nReturn ONLY the value of {k}. "
                  "No words, just the value.")
        try:
            ans = ask(args.model, prompt)
        except Exception as e:
            print(f"  (model error on {k}: {e})"); ans = ""
        model_right = norm(truth) in norm(ans)
        model_ok += model_right
        # THE CHECK: cross-reference the exact store (O(1) dict lookup)
        checked, rc = life("get", k)
        verified = checked if rc == 0 else None
        if verified is not None and norm(verified) not in norm(ans):
            caught.append((k, ans[:24], verified))  # deviation detected + corrected
            final = verified
        else:
            final = ans
        system_ok += norm(truth) in norm(final)

    n = len(keys)
    print(f"  MODEL ALONE : {model_ok}/{n} exact"
          + (f"  ({n - model_ok} answer(s) that don't match ground truth, delivered silently)" if model_ok < n else ""))
    print(f"  + CHECKER   : {system_ok}/{n} exact  (cross-check vs truth, O(1)/query, no 2nd model)")
    if caught:
        print("\n  deviations from ground truth caught and corrected"
              " (true swaps + dropped/reformatted values):")
        for k, wrong, right in caught:
            print(f"    {k}: model returned {wrong!r}...  -> corrected to {right!r}")
    print("\n  the gap is a SYSTEM property — same model, plus a dict lookup against truth.")
    print("  BILL: only works where a ground-truth value exists to check against.")
    os.path.exists(DB) and os.remove(DB)


if __name__ == "__main__":
    main()
