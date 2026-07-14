#!/usr/bin/env python3
"""
EXPERIMENT 1 — mined by the engine in round 1 (candidate D-2):
"Embedding min-distance novelty scoring saturates as the archive fills: what does the
saturation curve look like, and does compression-based surprise saturate more slowly?"

Design (deterministic, stdlib only):
  1. Stream 40 real questions (20 baseline + 20 windowed, interleaved) into an archive.
     At each insertion record the item's novelty under both scorers.
  2. After the archive is full, score three probe classes:
       DUP   = 5 exact duplicates of archived questions      (true novelty: zero)
       PARA  = 5 paraphrases of archived questions           (true novelty: near zero)
       NEW   = 5 questions from unrelated domains            (true novelty: high)
  3. A good scorer, late in the stream, still separates NEW > PARA > DUP.
     Metric: ordering violations (pairs where a NEW probe scores <= a DUP/PARA probe).
  4. Saturation: mean insertion-novelty of first 10 vs last 10 stream items.
"""
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine import tokenize, build_idf, vector, distance  # noqa: E402

BASE = Path(__file__).parent.parent


def emb_novelty(text, archive_texts):
    """TF-IDF min-distance novelty vs archive (idf over archive + probe)."""
    if not archive_texts:
        return 1.0
    toks = [tokenize(t) for t in archive_texts + [text]]
    idf = build_idf(toks)
    vecs = [vector(t, idf) for t in toks]
    probe = vecs[-1]
    return min(distance(probe, v) for v in vecs[:-1])


def comp_surprise(text, archive_blob):
    """Compression surprise: extra bits per character to append text to the archive."""
    base = len(zlib.compress(archive_blob, 9))
    ext = len(zlib.compress(archive_blob + b"\n" + text.encode(), 9))
    return (ext - base) * 8.0 / max(1, len(text))


def load_stream():
    a = [x["question"] for x in json.loads((BASE / "rounds/mve_baseline.json").read_text())]
    b = [x["question"] for x in json.loads((BASE / "rounds/round1_candidates.json").read_text())]
    stream = []
    for i in range(max(len(a), len(b))):          # deterministic interleave
        if i < len(a):
            stream.append(a[i])
        if i < len(b):
            stream.append(b[i])
    return stream


PARAPHRASES = [
    "For what reason do large language models make things up, and what methods shrink that behaviour?",
    "In transformers, what is the mechanism by which attention operates?",
    "What stops neural nets from remembering old tasks when they learn new ones?",
    "Is a gradient update over a batch an average that throws away the disagreement between examples?",
    "Why is the key-value cache rebuilt every session instead of being kept?",
]

NEW_DOMAIN = [
    "Why do coral reefs bleach when ocean temperatures rise by only two degrees?",
    "How does a court decide whether a contract clause is unconscionable?",
    "What makes sourdough rise without commercial yeast?",
    "Why did the Bretton Woods system of fixed exchange rates collapse in 1971?",
    "How does a violin's wooden body amplify string vibrations into audible sound?",
]


def main():
    stream = load_stream()

    # --- 1. insertion novelty over the stream (saturation curves) ---
    emb_curve, comp_curve = [], []
    archive_texts, blob = [], b""
    for q in stream:
        emb_curve.append(emb_novelty(q, archive_texts))
        comp_curve.append(comp_surprise(q, blob))
        archive_texts.append(q)
        blob = blob + b"\n" + q.encode() if blob else q.encode()

    def sat(curve):
        first, last = curve[:10], curve[-10:]
        f, l = sum(first) / 10, sum(last) / 10
        return f, l, (l - f) / f * 100

    ef, el, edrop = sat(emb_curve)
    cf, cl, cdrop = sat(comp_curve)

    # --- 2. probe discrimination on the FULL archive ---
    dups = [stream[3], stream[11], stream[19], stream[27], stream[35]]

    def score_all(probes):
        return [(emb_novelty(p, archive_texts), comp_surprise(p, blob)) for p in probes]

    s_dup, s_para, s_new = score_all(dups), score_all(PARAPHRASES), score_all(NEW_DOMAIN)

    def violations(new_scores, other_scores, idx):
        return sum(1 for n in new_scores for o in other_scores if n[idx] <= o[idx])

    def mean(scores, idx):
        return sum(s[idx] for s in scores) / len(scores)

    print("=== SATURATION (insertion novelty, first 10 vs last 10 of 40) ===")
    print(f"embedding min-dist : {ef:.4f} -> {el:.4f}   drift {edrop:+.1f}%")
    print(f"compression bits/ch: {cf:.4f} -> {cl:.4f}   drift {cdrop:+.1f}%")

    print("\n=== PROBE SCORES on full archive (mean per class) ===")
    print(f"{'class':6} {'emb-novelty':>12} {'comp-bits/char':>15}")
    for name, s in [("DUP", s_dup), ("PARA", s_para), ("NEW", s_new)]:
        print(f"{name:6} {mean(s, 0):12.4f} {mean(s, 1):15.4f}")

    print("\n=== ORDERING VIOLATIONS (out of 25 pairs each; 0 = perfect) ===")
    print(f"embedding  : NEW<=DUP {violations(s_new, s_dup, 0):2d}   NEW<=PARA {violations(s_new, s_para, 0):2d}")
    print(f"compression: NEW<=DUP {violations(s_new, s_dup, 1):2d}   NEW<=PARA {violations(s_new, s_para, 1):2d}")

    emb_margin = mean(s_new, 0) - mean(s_para, 0)
    comp_margin = (mean(s_new, 1) - mean(s_para, 1)) / mean(s_new, 1)
    emb_margin_rel = emb_margin / mean(s_new, 0)
    print("\n=== RELATIVE NEW-vs-PARA MARGIN (bigger = harder to fool with paraphrase) ===")
    print(f"embedding  : {emb_margin_rel * 100:5.1f}% of its NEW score")
    print(f"compression: {comp_margin * 100:5.1f}% of its NEW score")


if __name__ == "__main__":
    main()
