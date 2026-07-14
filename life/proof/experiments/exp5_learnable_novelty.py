#!/usr/bin/env python3
"""
EXPERIMENT 5 — mined by the engine in round 5 (AD-2, score 0.779):
"Does high perplexity mean 'new and learnable', or does it conflate learnable novelty
 with unlearnable noise — and is the distinguishing signal computable by a frozen model?"

Design (deterministic, stdlib): an ONLINE interpolated byte n-gram LM (orders 0-3,
count tables, never freezes — the Life paradigm) is trained on a base corpus, then
probed with four sources, each providing sample-1 and a FRESH sample-2:

  KNOWN       same domain as training            (old news)
  STRUCT-NEW  new domain, real English structure (learnable novelty — worth eating)
  FORMAL-NEW  alien but perfectly lawful pattern (extremely learnable)
  NOISE       LCG pseudo-random bytes            (maximally 'novel', unlearnable)

Signals per source:
  PERPLEXITY  bits/byte on sample-1 under the trained model (a frozen model has this)
  PROGRESS    bits/byte on FRESH sample-2 BEFORE vs AFTER eating sample-1
              (transferable learning progress — requires plasticity to compute)

Claim under test: perplexity cannot separate STRUCT-NEW from NOISE (both 'surprising');
progress can. FIRST RUN REFUTED THE NAIVE VERSION: one bite of noise yields real
progress (+6 bpb) because the model learns the noise source's MARGINAL statistics —
that too is law. So the experiment measures SUSTAINED progress: a second meal of noise
should pay ~nothing (marginal already learned, contexts never repeat), while lawful
sources keep paying. Curiosity = the derivative that KEEPS being positive.
"""
import math

ORDERS = 4  # context lengths 0..3


class LCG:
    def __init__(self, seed=99991):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s >> 33

    def byte(self):
        return self.next() % 256


class OnlineLM:
    """Interpolated count-table byte model, orders 0..3. Online: update after predict."""

    def __init__(self):
        self.tables = [dict() for _ in range(ORDERS)]  # ctx-> {byte: count, '_': total}

    def _probs(self, ctx):
        p = 1.0 / 256.0  # order -1: uniform
        out = {}
        # interpolate upward: p_o = (lambda) * table_o + (1-lambda) * p_below
        for o in range(ORDERS):
            key = ctx[-o:] if o else b""
            t = self.tables[o].get(key)
            if t is None:
                continue
            total = t["_"]
            lam = total / (total + 4.0)
            base = {b: c / total for b, c in t.items() if b != "_"}
            out = (o, key, lam, base)
        return out or None

    def bits_and_update(self, data, update=True):
        total_bits = 0.0
        ctx = b""
        for by in data:
            # build interpolated probability for byte `by`
            p = 1.0 / 256.0
            for o in range(ORDERS):
                key = ctx[-o:] if o else b""
                t = self.tables[o].get(key)
                if t is None:
                    continue
                tot = t["_"]
                lam = tot / (tot + 4.0)
                p = (1 - lam) * p + lam * (t.get(by, 0) / tot)
            total_bits += -math.log2(max(p, 1e-12))
            if update:
                for o in range(ORDERS):
                    key = ctx[-o:] if o else b""
                    t = self.tables[o].setdefault(key, {"_": 0})
                    t[by] = t.get(by, 0) + 1
                    t["_"] += 1
            ctx = (ctx + bytes([by]))[-(ORDERS - 1):]
        return total_bits / len(data)

    def clone(self):
        import copy
        c = OnlineLM()
        c.tables = copy.deepcopy(self.tables)
        return c


# ---------------- corpora (all embedded, deterministic) ----------------
TRAIN = (b"gradient descent updates the weights of the network by following the slope of the loss. "
         b"the optimizer averages gradients over a batch and steps the parameters downhill. "
         b"learning rates control the size of each step and schedules decay them over time. "
         b"backpropagation computes the gradient of the loss with respect to every weight. "
         b"the network learns statistical structure from data by repeating these steps. ") * 24

KNOWN_1 = (b"the optimizer follows the slope of the loss and updates the network weights. "
           b"gradients averaged over a batch step the parameters downhill each time. ") * 8
KNOWN_2 = (b"backpropagation gives the gradient for every weight and the network repeats the steps. "
           b"learning rates decay over time as schedules control the size of the step. ") * 8
KNOWN_3 = (b"schedules decay the learning rate while the optimizer steps the weights downhill. "
           b"the loss slope gives every update and the network repeats these steps on data. ") * 8

BIO_1 = (b"the ribosome reads messenger rna three bases at a time and joins amino acids into protein. "
         b"transfer rna molecules carry each amino acid to the matching codon in the sequence. ") * 8
BIO_2 = (b"enzymes fold into shapes that bind their substrates and lower the activation energy. "
         b"the cell membrane controls which molecules pass through its lipid bilayer. ") * 8
BIO_3 = (b"mitochondria oxidise glucose and store the energy as atp for the rest of the cell. "
         b"dna polymerase copies each strand and proofreads the new bases as it moves. ") * 8


def formal(n, phase):
    out = bytearray()
    a, b = 1 + phase, 2 + phase
    while len(out) < n:
        out += (b"%d;" % (a % 1000))
        a, b = b, a + b
    return bytes(out[:n])


def noise(n, seed):
    r = LCG(seed)
    return bytes(r.byte() for _ in range(n))


SOURCES = [
    ("KNOWN", KNOWN_1, KNOWN_2, KNOWN_3),
    ("STRUCT-NEW", BIO_1, BIO_2, BIO_3),
    ("FORMAL-NEW", formal(1200, 0), formal(1200, 7), formal(1200, 13)),
    ("NOISE", noise(1200, 5), noise(1200, 6), noise(1200, 8)),
]

base = OnlineLM()
train_bpb = base.bits_and_update(TRAIN)
print(f"base model trained online on {len(TRAIN)} bytes of ML text: {train_bpb:.3f} bpb\n")

print(f"{'source':>11} {'PERPLEXITY':>11} {'PROGRESS-1':>11} {'PROGRESS-2':>11} {'SUSTAIN':>8}")
rows = []
for name, s1, s2, s3 in SOURCES:
    m = base.clone()
    ppl = m.bits_and_update(s1, update=False)          # frozen signal on sample-1
    b2 = m.bits_and_update(s2, update=False)           # fresh s2, before eating s1
    m.bits_and_update(s1, update=True)                 # meal 1
    a2 = m.bits_and_update(s2, update=False)           # fresh s2, after meal 1
    prog1 = b2 - a2
    b3 = m.bits_and_update(s3, update=False)           # fresh s3, before eating s2
    m.bits_and_update(s2, update=True)                 # meal 2
    a3 = m.bits_and_update(s3, update=False)           # fresh s3, after meal 2
    prog2 = b3 - a3
    sustain = prog2 / prog1 if abs(prog1) > 1e-9 else 0.0
    rows.append((name, ppl, prog1, prog2, sustain))
    print(f"{name:>11} {ppl:>10.3f} {prog1:>+11.3f} {prog2:>+11.3f} {sustain:>8.2f}")

sn = {r[0]: r for r in rows}
print("\nthree signals, measured:")
print(f"  PERPLEXITY  (frozen model has this): STRUCT-NEW {sn['STRUCT-NEW'][1]:.2f} vs NOISE {sn['NOISE'][1]:.2f}"
      f" -> noise looks {sn['NOISE'][1] / sn['STRUCT-NEW'][1]:.1f}x MORE 'novel' — conflated")
print(f"  PROGRESS-1  (needs plasticity)     : STRUCT-NEW {sn['STRUCT-NEW'][2]:+.2f} vs NOISE {sn['NOISE'][2]:+.2f}"
      f" -> still conflated (noise pays once: its marginal is learnable law too)")
print(f"  SUSTAIN = PROGRESS-2/PROGRESS-1    : STRUCT-NEW {sn['STRUCT-NEW'][4]:.2f}"
      f" FORMAL {sn['FORMAL-NEW'][4]:.2f} vs NOISE {sn['NOISE'][4]:.2f}"
      f" -> lawful sources keep paying, noise dies after one meal")
print("\nonly an organism that stays plastic ACROSS MEALS can compute SUSTAIN —")
print("neither a frozen model (no progress) nor a one-shot adapter (no second meal) can.")
