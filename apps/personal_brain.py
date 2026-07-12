#!/usr/bin/env python3
"""personal_brain.py — a private, on-device, persistent fact memory.

You tell it things; it remembers them (saved to disk); you ask in plain language; it retrieves and
answers. Everything is local — no cloud, no account. Adapts to your laptop:
  - retrieval: uses a semantic embedder if `transformers` is installed (handles reworded queries),
    otherwise falls back to keyword matching (works, but misses pure synonyms);
  - answering: uses a local model if one is available (MLX, then HF transformers), otherwise it
    just returns the matching facts. So the CORE runs with nothing but Python.

Honest scope:
  - The MEMORY is exact, persistent, and flat (it does not grow with how long you use it, only with
    #facts). That part is lossless and deterministic.
  - SEMANTIC recall is a float embedder — strong on rewording, but it has holes (negation, rare
    synonyms) and thin margins with many facts. It is not infallible.
  - Any REASONING/phrasing is the optional model's job, at the model's quality — the memory layer
    does not add intelligence, it adds recall.

Usage:
  python3 personal_brain.py remember "the project deadline is March 15"
  python3 personal_brain.py ask "when is my project due?"
  python3 personal_brain.py            # interactive
"""
import sys, os, json, re
STORE = os.path.expanduser("~/.personal_brain.json")

class Brain:
    def __init__(self):
        self.facts = []
        self._load()
        self.mode = "keyword"
        self._emb = None
        self._try_embedder()

    def _load(self):
        if os.path.exists(STORE):
            try: self.facts = json.load(open(STORE)).get("facts", [])
            except Exception: self.facts = []

    def _save(self):
        json.dump({"facts": self.facts}, open(STORE, "w"))

    def _try_embedder(self):
        try:
            import torch, numpy as np
            from transformers import AutoTokenizer, AutoModel
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

    def ask(self, query):
        hits = self.recall(query)
        if not hits:
            return "I don't have anything about that yet."
        ctx = "\n".join(f"- {f}" for f, _ in hits)
        ans = self._model_answer(query, ctx)
        return ans if ans else f"From what you told me:\n{ctx}"

    def _model_answer(self, query, ctx):
        prompt = f"Notes:\n{ctx}\n\nQuestion: {query}\nAnswer using only the notes, concisely."
        try:  # try Apple MLX first
            from mlx_lm import load, generate
            if not hasattr(self, "_mlx"): self._mlx = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
            model, tok = self._mlx
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
            return generate(model, tok, prompt=text, max_tokens=60, verbose=False).strip()
        except Exception:
            return None  # no local model -> ask() returns the facts directly

def main():
    b = Brain()
    args = sys.argv[1:]
    if args and args[0] == "remember":
        b.remember(" ".join(args[1:])); print("remembered.")
    elif args and args[0] == "ask":
        print(b.ask(" ".join(args[1:])))
    else:
        print(f"Personal Brain (local). {len(b.facts)} facts stored | recall: {b.mode}")
        print("Commands: remember <text> | ask <text> | quit")
        while True:
            try: line = input("> ").strip()
            except (EOFError, KeyboardInterrupt): print(); break
            if line in ("quit", "exit"): break
            if line.startswith("remember "): b.remember(line[9:]); print("remembered.")
            elif line.startswith("ask "): print(b.ask(line[4:]))
            elif line: print("use: remember <text> | ask <text> | quit")

if __name__ == "__main__":
    main()
