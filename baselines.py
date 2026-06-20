"""
Reference baselines for RedlineBench v2 -- run with no API, no judge.

Puts the frontier-model eval numbers in context by scoring three reference
policies through the ACTUAL vendor loop (vendor_decision in redline_v2.py) and
the ACTUAL reward (scoring.py):

  - naive split  : propose [0.5]*8 every round (the dumb anchor).
  - blind constant: propose [c]*8 every round, ignoring every counter. A policy
                    with NO negotiation skill -- it never reads the vendor.
  - logroll oracle: full-info upper bound -- the frontier deal that concedes the
                    vendor's high-weight terms and holds the buyer's. The most a
                    perfect logrolling agent could capture.

Why this matters: it establishes what "skill" is worth in this environment. If
the frontier models (~0.50 buyer reward) sit at the blind-constant level (~0.55)
and far below the oracle (1.00), the headline is honest and sharp -- current
models negotiate no better than a policy that ignores the other side, and leave
~half the achievable value on the table.

This vendor logic is kept identical to redline_v2.vendor_decision; this file
just avoids importing the verifiers stack so it runs standalone.

Run:  python baselines.py
"""

from __future__ import annotations
import random, statistics as st
import scoring as sc


def vendor_decision(s, buyer_x, rn, total):
    """Identical to redline_v2.vendor_decision (copied to keep this dep-free)."""
    uv = sc.vendor_utility(s, buyer_x)
    progress = rn / max(1, total)
    threshold = s.vendor.batna + (1.0 - s.vendor.batna) * 0.45 * (1.0 - progress)
    if uv >= threshold:
        return True, False, buyer_x
    if rn >= total:
        return (uv >= s.vendor.batna), (uv < s.vendor.batna), buyer_x
    order = sorted(range(len(s.terms)), key=lambda i: s.wv[i], reverse=True)
    hold = set(order[: len(order) // 2])
    counter = [min(buyer_x[i], 0.25) if i in hold else max(buyer_x[i], 0.6)
               for i in range(len(s.terms))]
    return False, False, counter


def run_episode(s, propose, total=4):
    buyer_x = [0.5] * len(s.terms)
    for rn in range(1, total + 1):
        buyer_x = propose(rn)
        accepted, walked, counter = vendor_decision(s, buyer_x, rn, total)
        if accepted:
            return sc.score(s, buyer_x, True)
        if walked or rn >= total:
            return sc.score(s, None, False)
    return sc.score(s, None, False)


def logroll_oracle(s):
    """Full-info upper bound: frontier deal pushing the vendor to its BATNA."""
    best_x, best_ub = [0.5] * len(s.terms), -1.0
    for k in range(401):
        lam = k / 400
        x = [1.0 if lam * wb >= (1 - lam) * wv else 0.0
             for wb, wv in zip(s.wb, s.wv)]
        if sc.vendor_utility(s, x) >= s.vendor.batna - 1e-9 and sc.buyer_utility(s, x) > best_ub:
            best_ub, best_x = sc.buyer_utility(s, x), x
    return lambda rn: best_x


def sweep(n=3000, seed=7):
    rng = random.Random(seed)
    scenarios = [sc.sample_scenario(rng) for _ in range(n)]
    policies = [
        ("naive split [0.5]x8",         lambda s: (lambda rn: [0.5] * 8)),
        ("blind constant [0.55]x8",     lambda s: (lambda rn: [0.55] * 8)),
        ("blind constant [0.60]x8",     lambda s: (lambda rn: [0.60] * 8)),
        ("logroll oracle (full info)",  logroll_oracle),
    ]
    print(f"\nRedlineBench v2 reference baselines"
          f"\n{n} random scenarios, 4 rounds, real reward (scoring.py), no API.\n")
    print(f"{'policy':<30}{'reward':>9}{'deal_rate':>11}{'logroll':>9}")
    print("-" * 59)
    for name, make in policies:
        rs, deals, lrs = [], [], []
        for s in scenarios:
            o = run_episode(s, make(s))
            rs.append(o.buyer_score); deals.append(1.0 if o.deal_reached else 0.0)
            lrs.append(o.logroll_index)
        print(f"{name:<30}{st.mean(rs):>9.3f}{st.mean(deals):>11.3f}{st.mean(lrs):>9.3f}")
    print("\nReading: frontier models score ~0.50 (see baseline chart) -- i.e. at"
          "\nthe blind-constant level and far below the logroll oracle (1.00)."
          "\nThe environment is solvable; current models just don't trade.\n")


if __name__ == "__main__":
    sweep()
