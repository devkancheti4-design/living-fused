#!/usr/bin/env python3
"""
EXPERIMENT 9 — mined in Session 2 round 1 (IG-2), queued, now run (IG-5, score 0.699):
Two quantitative laws implied by 'scalar reward = rank-1 shadow of vector value':

LAW 1 (cosine law): under a resource budget ||y|| <= R, optimizing proxy direction w
  achieves true value V = V* . cos(angle(v, w)) — the value you get is EXACTLY the
  cosine between what you want and what you measure. (Analytic; verified numerically
  by projected gradient ascent, including that ascent CONVERGES to the proxy optimum.)

LAW 2 (peak-then-fall): if true value is CONCAVE in the real dimensions (diminishing
  returns) and the proxy MISREADS one dimension (rewards a hack the truth penalizes),
  then along the ascent trajectory true value rises, peaks, and falls while the proxy
  rises monotonically — the measured RLHF overoptimization shape (Gao et al. 2023),
  reproduced from rank deficiency + curvature alone, no neural net required.

Honest label: a constructed DEMONSTRATION that quantifies the mechanism's shape —
not a discovery about any trained system. That is what Rule 20 requires me to say.
"""
import math

# ---------------- LAW 1: cosine law ----------------
print("LAW 1 — cosine law: achieved true value / optimal = cos(proxy, truth)")
print(f"{'angle(deg)':>10} {'predicted cos':>14} {'measured (ascent)':>18}")
D = 6
v = [1.0] + [0.0] * (D - 1)                      # true value direction (unit)
for deg in (0, 15, 30, 45, 60, 75, 90):
    rad = math.radians(deg)
    w = [math.cos(rad), math.sin(rad)] + [0.0] * (D - 2)   # proxy at angle
    # projected gradient ascent of w.y on the unit ball, from origin-ish
    y = [1e-3] * D
    for _ in range(500):
        y = [yi + 0.05 * wi for yi, wi in zip(y, w)]
        n = math.sqrt(sum(t * t for t in y))
        if n > 1.0:
            y = [t / n for t in y]
    achieved = sum(a * b for a, b in zip(v, y))   # V* = 1 on the unit ball
    print(f"{deg:>10} {math.cos(rad):>14.4f} {achieved:>18.4f}")

# ---------------- LAW 2: peak-then-fall ----------------
print("\nLAW 2 — overoptimization curve from rank deficiency + diminishing returns")
print("true V(y) = sqrt(y1) + sqrt(y2) - 2*y3   (y3 = hack: truth penalizes it)")
print("proxy r(y) = y1 + 0.0*y2 + 0.8*y3        (blind to y2, MISREADS the hack)")
w = [1.0, 0.0, 0.8]
nw = math.sqrt(sum(t * t for t in w))
print(f"\n{'ascent t':>9} {'proxy r':>9} {'true V':>9}")
peak = (-1e9, 0.0)
final = None
for step in range(0, 21):
    t = step / 20.0                                # radial progress toward proxy optimum
    y = [t * wi / nw for wi in w]                  # budget spent along proxy direction
    r = sum(a * b for a, b in zip(w, y))
    V = math.sqrt(max(y[0], 0)) + math.sqrt(max(y[1], 0)) - 2.0 * y[2]
    if V > peak[0]:
        peak = (V, t)
    final = (V, r, t)
    if step % 2 == 0:
        print(f"{t:>9.2f} {r:>9.3f} {V:>9.3f}")
print(f"\nproxy rose monotonically to {final[1]:.3f}; true value peaked at t={peak[1]:.2f}"
      f" (V={peak[0]:.3f}) then FELL to {final[0]:.3f} at the proxy optimum")
print(f"overoptimization cost: {(peak[0] - final[0]):.3f} = {(peak[0] - final[0]) / abs(peak[0]) * 100:.0f}% of peak true value destroyed by finishing the climb")
