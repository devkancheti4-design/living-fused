#!/usr/bin/env python3
"""BENCHMARK — life.py vs a plain dict vs semantic RAG, on the same facts.

This exists to answer a skeptic's three fair questions with measured numbers,
not adjectives:

  1. How is a fact like "user's age is 42" stored, and can you recall the year
     of birth from it? (Answer: stored as key->value; NO, the store can't do
     the arithmetic — it abstains; the model does that.)
  2. How is this different from a dictionary? (Measured below: on pure exact
     store+recall it is NOT different — a dict ties it. The real differences
     are the abstention contract, cross-process determinism, and logit fusion.)
  3. Prove "doesn't forget / doesn't hallucinate" against existing systems.
     (Measured below on 4 axes vs a dict and vs real MiniLM RAG.)

Deterministic core (dict/life) needs only stdlib. The RAG arm needs
sentence-transformers/torch; if absent it is reported as SKIPPED, not faked.

Run:  python3 benchmark.py
"""
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from life import Life

# ---- the dataset: entity-keyed facts, each with an exact and a paraphrase query
FACTS = [
    # key,                value,               exact query,                       paraphrase query
    ("user.age",          "42",                "user age",                        "how old is the user"),
    ("user.name",         "Devieswar",         "user name",                       "what should I call them"),
    ("wifi.password",     "SunFlower42",        "wifi password",                   "the code to get online"),
    ("server.ip",         "192.168.7.203",      "server ip",                       "address of the box"),
    ("mom.birthday",      "March 11",           "mom birthday",                    "when is his mother's big day"),
    ("dentist.appt",      "Friday 3pm",         "dentist appt",                    "when do I see the tooth doctor"),
    ("car.make",          "Toyota",             "car make",                        "brand of the vehicle"),
    ("flight.gate",       "C22",                "flight gate",                     "where do I board"),
    ("project.codename",  "BlueHeron",          "project codename",                "secret name of the project"),
    ("locker.number",     "48",                 "locker number",                   "which locker is mine"),
]
# keys that were NEVER stored — the hallucination probe
UNSTORED = ["user.ssn", "bank.pin", "passport.number", "home.alarm.code", "blood.type"]


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def P(m):
    sys.stdout.write(m + "\n"); sys.stdout.flush()


# ============================================================================
# System 1: a plain Python dict (the skeptic's baseline)
# ============================================================================
class DictStore:
    name = "plain dict"
    def __init__(s): s.d = {}
    def put(s, k, v): s.d[k] = v          # assignment overwrites; no history
    def get(s, k):                         # exact key only; miss -> None
        return s.d.get(_key_for(k))
    def sha(s):
        return hashlib.sha256(repr(sorted(s.d.items())).encode()).hexdigest()[:16]


# ============================================================================
# System 2: life.py (via the real CLI, exactly as a fused agent calls it)
# ============================================================================
class LifeStore:
    name = "life.py"
    def __init__(s):
        s.db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench.json")
        if os.path.exists(s.db): os.remove(s.db)
        s.cli = os.path.join(os.path.dirname(os.path.abspath(__file__)), "life.py")
    def _run(s, *a):
        r = subprocess.run([sys.executable, s.cli, *a], capture_output=True,
                           text=True, env=dict(os.environ, LIFE_DB=s.db))
        return r.stdout.strip(), r.returncode
    def put(s, k, v): s._run("put", k, v)
    def get(s, k):
        out, rc = s._run("get", _key_for(k))
        return None if rc == 3 else out    # ABSTAIN(exit 3) -> None (honest miss)
    def sha(s): return s._run("sha")[0]
    def cleanup(s): os.path.exists(s.db) and os.remove(s.db)


# ============================================================================
# System 3: real semantic RAG (MiniLM embeddings + cosine, with an abstain gate)
# ============================================================================
class RAGStore:
    name = "RAG (MiniLM)"
    THRESH = 0.45     # below this cosine, abstain instead of returning a wrong fact
    def __init__(s):
        import torch, numpy as np
        from transformers import AutoTokenizer, AutoModel
        s.torch, s.np = torch, np
        m = "sentence-transformers/all-MiniLM-L6-v2"
        s.tok = AutoTokenizer.from_pretrained(m)
        s.model = AutoModel.from_pretrained(m).eval()
        s.keys, s.vals, s.vecs = [], [], None
    def _embed(s, texts):
        with s.torch.no_grad():
            t = s.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=48)
            h = s.model(**t).last_hidden_state
            mask = t.attention_mask.unsqueeze(-1)
            p = (h * mask).sum(1) / mask.sum(1)
            return (p / p.norm(dim=-1, keepdim=True)).numpy()
    def put(s, k, v):
        s.keys.append(k.replace(".", " ")); s.vals.append(v)
        s.vecs = s._embed(s.keys)
    def get(s, query):                     # semantic top-1, gated by threshold
        q = s._embed([query])[0]
        sims = s.vecs @ q
        i = int(sims.argmax())
        return s.vals[i] if sims[i] >= s.THRESH else None


def _key_for(q):
    """Map a natural query to the stored key for the exact systems (dict/life).
    This is what a fused AGENT does: it picks the key. We give dict/life the
    fair benefit of a correct key on EXACT queries; on paraphrases the agent
    would still pick the key, but to isolate the STORE we test raw query text."""
    return _QUERYKEY.get(norm(q), norm(q).replace(" ", "."))


_QUERYKEY = {}

# ---------------------------------------------------------------------------
def run(store, load, queries):
    for k, v in load:
        store.put(k, v)
    hits = 0
    for q, want in queries:
        got = store.get(q)
        if got is not None and norm(got) == norm(want):
            hits += 1
    return hits


def main():
    # build the exact-query->key map so dict/life get the right key on EXACT probes
    for k, v, exact, para in FACTS:
        _QUERYKEY[norm(exact)] = k
    load = [(k, v) for k, v, _, _ in FACTS]
    exact_q = [(exact, v) for k, v, exact, para in FACTS]
    para_q = [(para, v) for k, v, exact, para in FACTS]
    n = len(FACTS)

    systems = [DictStore(), LifeStore()]
    rag = None
    try:
        rag = RAGStore(); systems.append(rag)
    except Exception as e:
        P(f"(RAG arm SKIPPED — sentence-transformers/torch not available: {type(e).__name__})")

    P("=" * 74)
    P(f"BENCHMARK — {n} facts, three systems, four honest axes")
    P("=" * 74)
    P(f"{'system':<15}{'exact':>10}{'paraphrase':>13}{'no-halluc.':>13}{'determ.':>10}")
    P("-" * 74)

    for s in systems:
        # fresh load per system
        if isinstance(s, LifeStore):
            s.__init__()
        elif isinstance(s, RAGStore):
            s.keys, s.vals, s.vecs = [], [], None
        else:
            s.d = {}
        for k, v in load:
            s.put(k, v)

        exact_hits = sum(1 for q, w in exact_q
                         if (g := s.get(q)) is not None and norm(g) == norm(w))
        para_hits = sum(1 for q, w in para_q
                        if (g := s.get(q)) is not None and norm(g) == norm(w))
        # hallucination probe: query never-stored keys. GOOD = returns nothing.
        abstains = 0
        for uk in UNSTORED:
            _QUERYKEY[norm(uk)] = uk
            probe = uk if not isinstance(s, RAGStore) else uk.replace(".", " ")
            if s.get(probe) is None:
                abstains += 1
        # determinism: rebuild from scratch, compare identity
        determ = "n/a"
        if hasattr(s, "sha"):
            sha1 = s.sha()
            if isinstance(s, LifeStore):
                s.__init__()
            else:
                s.d = {}
            for k, v in load:
                s.put(k, v)
            determ = "byte-exact" if s.sha() == sha1 else "DRIFT"

        P(f"{s.name:<15}{exact_hits}/{n:<8}{para_hits}/{n:<11}"
          f"{abstains}/{len(UNSTORED):<11}{determ:>10}")

    P("-" * 74)
    P("exact      = entity-named recall (the fact you asked for, verbatim)")
    P("paraphrase = same fact asked in different words (semantic)")
    P("no-halluc. = of 5 NEVER-stored keys, how many it correctly returned NOTHING for")
    P("determ.    = rebuild from the same facts -> identical store? (byte-exact SHA)")

    # ---- the specific age -> birth-year question, honestly -----------------
    P("\n" + "=" * 74)
    P("THE 'age 42 -> year of birth' QUESTION")
    P("=" * 74)
    life = LifeStore(); life.__init__()
    life.put("user.age", "42")
    P(f"  life.py get 'user.age'         -> {life.get('user age')}   (exact recall)")
    P(f"  life.py get 'user.birth.year'  -> {life.get('user birth year')}   "
      "(ABSTAIN: the STORE does no arithmetic, and refuses to guess)")
    P("  the FUSED answer = store gives '42' + the MODEL computes 2026-42 = ~1984.")
    P("  Memory remembers; model reasons. The store abstaining is CORRECT, not broken.")
    life.cleanup()

    if rag is None:
        P("\nNOTE: RAG arm was skipped, so the paraphrase/hallucination contrast")
        P("vs a semantic system is not shown here. Install transformers+torch to see it.")


if __name__ == "__main__":
    main()
