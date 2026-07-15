#!/usr/bin/env python3
"""personal_brain.py — a private, on-device, conversational memory.

Talk to it in plain language. It figures out whether you're telling it a fact (it remembers) or
asking a question (it answers from what it knows), and it holds the last few turns so follow-ups
like "and where is that?" work. Everything is local and persistent (~/.personal_brain.json).

How the "talking" works (honest):
  - MEMORY: an exact, flat, persistent store of your facts. It never grows with how long you use it,
    only with how many facts you save.
  - RECALL: a semantic embedder if `transformers` is installed (handles rewording), else keyword.
    Strong, but a float matcher with holes (negation, rare synonyms, thin margins with many facts).
  - TALKING: your LOCAL model (Apple MLX by default) phrases replies, grounded in the recalled facts
    and the conversation so far. The memory feeds it facts; the model does the language. With no
    local model, it simply returns the matching facts. Quality of the talk == quality of your model.

Usage:
  python3 personal_brain.py                       # interactive chat
  python3 personal_brain.py "when is my project due?"   # one-shot
  AUTO_REMEMBER=0 python3 personal_brain.py       # only remember on explicit 'remember ...'
"""
import sys, os, json, re
STORE = os.path.expanduser("~/.personal_brain.json")
AUTO_REMEMBER = os.environ.get("AUTO_REMEMBER", "1") != "0"
GREET = {"hi", "hello", "hey", "yo", "sup", "thanks", "thank you", "thx", "ok", "okay", "k",
         "cool", "nice", "great", "yes", "no", "yeah", "nope", "bye", "goodbye", "lol", "haha"}
# words that follow "I'm"/"I am" but are NOT a name (so "I'm allergic" != name "allergic")
_NOT_A_NAME = {"going", "allergic", "here", "not", "sorry", "sure", "fine", "good", "ok",
               "okay", "ready", "done", "tired", "happy", "working", "looking", "trying",
               "a", "an", "the", "so", "really", "very", "just", "still", "at", "in", "on"}

def _hf_cached(repo):
    """Is this model already in the local HF cache? (no network)"""
    import glob
    pat = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*")
    return bool(glob.glob(pat))

def _online(host="huggingface.co", timeout=3):
    """Quick reachability check so a missing network fails FAST, never hangs."""
    import socket
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


class Brain:
    MAX_TURNS = 6  # conversation turns the model sees
    PERSONA = ("You are a warm, concise personal assistant with the user's private saved notes. "
               "Answer using those notes and the conversation so far. If the notes don't contain the "
               "answer, say so briefly instead of inventing one. Keep replies short and natural.")

    def __init__(self):
        self.facts = []
        self.pins = {}              # EXACT key->value (name, "my X is Y"): the Life layer.
        self.history = []           # [{role, content}, ...]  (per session, not persisted)
        self._load()
        self.mode = "keyword"
        self._try_embedder()

    # ---------- persistence ----------
    def _load(self):
        if os.path.exists(STORE):
            try:
                d = json.load(open(STORE))
                self.facts = d.get("facts", [])
                self.pins = d.get("pins", {})   # back-compat: old files have no pins
            except Exception:
                self.facts = []; self.pins = {}

    def _save(self):
        json.dump({"facts": self.facts, "pins": self.pins}, open(STORE, "w"))

    # ---------- EXACT pinned facts (the Life layer — never fuzzy, never out-ranked) ----------
    def _extract_pins(self, message):
        """Capture identity/attribute facts as EXACT key->value. These are ALWAYS
        injected into context, so — unlike top-k semantic recall — they can never
        be out-ranked or forgotten. Name introductions are caught even when they
        are too short/casual for the semantic _is_fact filter (the reported bug:
        'I'm Devieswar' was silently dropped)."""
        m = message.strip()
        # name: "my name is X" | "I'm X" | "I am X" | "call me X" | "name's X" | "this is X"
        nm = re.search(r"\b(?:my name is|i am|i'm|call me|name's|name is|this is)\s+"
                       r"([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,2})", m, re.I)
        if nm:
            cand = nm.group(1).strip()
            # reject obvious non-names ("I'm going", "I'm allergic", ...)
            if cand.lower().split()[0] not in _NOT_A_NAME:
                self.pins["name"] = cand
        # attributes: "my <attr> is <value>"  (server ip, wifi password, phone number, ...)
        # value may contain dots (IPs, versions, decimals); stop at comma/semicolon/
        # newline or a sentence-ending period, and trim trailing sentence punctuation.
        for a, v in re.findall(r"\bmy\s+([\w ]{2,30}?)\s+(?:is|are)\s+([^,;\n]{1,60})", m, re.I):
            a = a.strip().lower()
            v = re.sub(r"[.!?\s]+$", "", v.strip())  # drop trailing '.', keep 192.168.1.9
            if a and a != "name" and v:
                self.pins[a] = v
        return bool(nm) or ("my " in m.lower() and " is " in m.lower())

    def _pinned_context(self):
        return "\n".join(f"- {k}: {v}" for k, v in self.pins.items())

    def _answer_from_pins(self, query):
        """Direct exact answer for pin queries — works even with no model loaded."""
        q = query.lower()
        if "name" in q and "name" in self.pins:
            return f"Your name is {self.pins['name']}."
        for k, v in self.pins.items():
            if k != "name" and k in q:
                return f"Your {k} is {v}."
        return None

    # ---------- retrieval ----------
    def _try_embedder(self):
        try:
            import torch, numpy as np
            from transformers import AutoTokenizer, AutoModel
            emb = "sentence-transformers/all-MiniLM-L6-v2"
            if not _hf_cached(emb):
                if not _online():
                    print("(offline & embedding model not cached -> semantic recall off, keyword recall on)", flush=True)
                    self.mode = "keyword"; return
                print("(first run: downloading a ~90 MB embedding model -- one time, needs internet...)", flush=True)
            self._torch, self._np = torch, np
            self._t = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self._m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
            self.mode = "semantic"
            self._vecs = self._embed(self.facts) if self.facts else None
        except Exception:
            self.mode = "keyword"

    def _embed(self, texts):
        with self._torch.no_grad():
            t = self._t(texts, return_tensors="pt", padding=True, truncation=True, max_length=48)
            h = self._m(**t).last_hidden_state
            m = t.attention_mask.unsqueeze(-1)
            p = (h * m).sum(1) / m.sum(1)
            return (p / p.norm(dim=-1, keepdim=True)).numpy()

    def remember(self, text):
        text = text.strip()
        if not text: return
        self.facts.append(text)
        if self.mode == "semantic":
            v = self._embed([text])
            self._vecs = v if self._vecs is None else self._np.vstack([self._vecs, v])
        self._save()

    def recall(self, query, k=3):
        if not self.facts: return []
        if self.mode == "semantic" and self._vecs is not None:
            q = self._embed([query])[0]
            sims = self._vecs @ q
            idx = sims.argsort()[::-1][:k]
            return [(self.facts[i], float(sims[i])) for i in idx]
        qw = set(re.findall(r"\w+", query.lower()))
        scored = sorted(((len(qw & set(re.findall(r"\w+", f.lower()))), f) for f in self.facts), reverse=True)
        return [(f, float(sc)) for sc, f in scored[:k] if sc > 0]

    # ---------- intent ----------
    def _is_question(self, m):
        ml = m.strip().lower()
        return ml.endswith("?") or re.match(
            r"^(what|when|where|who|whom|whose|why|how|which|is|are|am|was|were|do|does|did|can|could|"
            r"would|should|will|shall|have|has|had|tell|show|list|find|remind|give|explain)\b", ml) is not None

    def _is_fact(self, m):
        ml = m.strip().lower().rstrip(".!")
        if ml in GREET or len(m.split()) < 3 or self._is_question(m): return False
        return re.search(r"\b(my|i'm|i am|i've|i have|i live|i work|i like|i prefer|is|are|was|were|will|"
                         r"allergic|deadline|password|number|address|birthday|appointment|meeting|due)\b", ml) is not None

    # ---------- the main entry: talk to it ----------
    def chat(self, message):
        message = (message or "").strip()
        if not message: return ""
        low = message.lower()
        if low.startswith("remember "):
            self.remember(message[9:]); return f"Got it — I'll remember that. ({len(self.facts)} saved)"
        if low.startswith("forget all"):
            self.facts = []; self._vecs = None; self.pins = {}; self._save(); self.history = []; return "Cleared everything I had."
        if low.startswith("forget "):
            target = message[7:].strip().lower(); before = len(self.facts) + len(self.pins)
            self.facts = [f for f in self.facts if target not in f.lower()]
            self.pins = {k: v for k, v in self.pins.items()
                         if target not in k.lower() and target not in v.lower()}
            if self.mode == "semantic": self._vecs = self._embed(self.facts) if self.facts else None
            self._save(); return f"Forgot {before - len(self.facts) - len(self.pins)} matching note(s)."
        # EXACT pins first: capture identity/attributes on EVERY message (even short
        # ones the semantic filter drops), and answer pin questions exactly.
        saved_pin = self._extract_pins(message) if AUTO_REMEMBER else False
        if saved_pin: self._save()
        if self._is_question(message):
            exact = self._answer_from_pins(message)
            if exact:
                self._push(message, exact); return exact
        if AUTO_REMEMBER and self._is_fact(message):
            self.remember(message)
            reply = self._respond(message, self.recall(message), just_saved=True)
            self._push(message, reply); return reply
        reply = self._respond(message, self.recall(message), just_saved=saved_pin)
        self._push(message, reply); return reply

    def ask(self, q):  # backward compatible
        return self.chat(q)

    def _push(self, user, assistant):
        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        self.history = self.history[-self.MAX_TURNS * 2:]

    def _respond(self, message, hits, just_saved=False):
        ctx = "\n".join(f"- {f}" for f, _ in hits) if hits else "(no matching notes)"
        sysmsg = self.PERSONA
        if just_saved: sysmsg += "\nThe user just told you a new fact — acknowledge it warmly in one short line."
        # EXACT pinned facts are ALWAYS in context — never out-ranked by top-k recall.
        if self.pins:
            sysmsg += "\n\nAlways-true facts about the user (exact, never guess these):\n" + self._pinned_context()
        sysmsg += "\n\nSaved notes relevant to this turn:\n" + ctx
        msgs = [{"role": "system", "content": sysmsg}] + self.history[-self.MAX_TURNS * 2:] + \
               [{"role": "user", "content": message}]
        out = self._model_chat(msgs)
        if out: return out
        if just_saved: return "Got it — saved."
        return ("From your notes:\n" + ctx) if hits else "I don't have anything on that yet — tell me and I'll remember it."

    def _model_chat(self, msgs):
        # your LOCAL model phrases the reply. Apple MLX by default; plug in any backend here.
        try:
            from mlx_lm import load, generate
            if not hasattr(self, "_mlx"):
                repo = "mlx-community/Qwen2.5-7B-Instruct-4bit"
                if not _hf_cached(repo):
                    if not _online():
                        return None  # offline & model not cached: facts-only replies, no hang
                    print(f"(first run: downloading the local phrasing model {repo} -- several GB, ONE time."
                          " Ctrl+C to skip; facts-only replies still work.)", flush=True)
                self._mlx = load(repo)
            model, tok = self._mlx
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            return generate(model, tok, prompt=text, max_tokens=140, verbose=False).strip()
        except Exception:
            return None  # no local model -> _respond returns the facts directly

def main():
    b = Brain()
    args = sys.argv[1:]
    if args:
        print(b.chat(" ".join(args))); return
    ar = "on" if AUTO_REMEMBER else "off"
    print(f"Personal Brain — {len(b.facts)} facts · recall: {b.mode} · auto-remember: {ar}")
    print("Just talk to me. ('forget <text>' / 'forget all' to manage · 'quit' to leave)")
    while True:
        try: line = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if line.lower() in ("quit", "exit"): break
        if line: print("brain >", b.chat(line))

if __name__ == "__main__":
    main()
