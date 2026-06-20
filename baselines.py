"""
Reference baselines for RedlineBench v2 -- run with no API, no judge.

Puts the frontier-model eval numbers in context by scoring four reference
policies through the ACTUAL vendor loop (vendor_decision in redline_v2.py) and
the ACTUAL reward (scoring.py):

  - naive split    : propose [0.5]*8 every round (the dumb anchor).
  - blind constant : propose [c]*8 every round, ignoring every counter. No skill.
  - counter-reader : a ~20-line rule-based logroller. Reads the vendor's counter
                     to infer its priorities, concedes the terms the vendor holds
                     AND the buyer values least, pushes the terms the buyer values
                     most. Knows its own weights, never sees the vendor's.
  - logroll oracle : full-info upper bound -- the frontier deal that pushes the
                     vendor to its BATNA. The most a perfect logroller could get.

The headline: the counter-reading heuristic (~0.62) beats every frontier model
in the baseline sweep (all ~0.50, see redline_v2_baseline.png) and the blind
constant (~0.55). Current frontier models do not even execute the obvious
logrolling heuristic; they negotiate no better than a constant that ignores the
other side, and leave most of the value on the table.

The vendor logic here is identical to redline_v2.vendor_decision; this file just
avoids importing the verifiers stack so it runs standalone.

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
    hold = set(order[: len(s.terms) // 2])
    counter = [min(buyer_x[i], 0.25) if i in hold else max(buyer_x[i], 0.6)
               for i in range(len(s.terms))]
    return False, False, counter


def run_episode(s, policy, total=4):
    """policy(round_num, last_counter, scenario) -> buyer_x."""
    buyer_x = [0.5] * len(s.terms)
    last_counter = None
    for rn in range(1, total + 1):
        buyer_x = policy(rn, last_counter, s)
        accepted, walked, counter = vendor_decision(s, buyer_x, rn, total)
        if accepted:
            return sc.score(s, buyer_x, True)
        if walked or rn >= total:
            return sc.score(s, None, False)
        last_counter = counter
    return sc.score(s, None, False)


# --- policies ---

def naive(rn, last, s):
    return [0.5] * len(s.terms)

def blind_constant(c):
    return lambda rn, last, s: [c] * len(s.terms)

def counter_reader(rn, last, s):
    """Rule-based logroller. Round 1 anchors high to draw a counter. After that,
    it reads which terms the vendor held firm on (counter pushed toward 0) and
    concedes the ones it also values least, while holding the terms it values
    most. No LLM, no vendor weights -- just reading the counter."""
    n = len(s.terms)
    if last is None:
        return [0.7] * n
    median_w = sorted(s.wb)[n // 2]
    x = [0.5] * n
    for i in range(n):
        vendor_holds = last[i] <= 0.3
        if vendor_holds and s.wb[i] <= median_w:   # vendor wants it, buyer doesn't -> give it
            x[i] = 0.1
        elif s.wb[i] > median_w:                    # buyer values it -> hold/push
            x[i] = 0.9
        else:
            x[i] = 0.5
    return x

def logroll_oracle(rn, last, s):
    best_x, best_ub = [0.5] * len(s.terms), -1.0
    for k in range(401):
        lam = k / 400
        x = [1.0 if lam * wb >= (1 - lam) * wv else 0.0
             for wb, wv in zip(s.wb, s.wv)]
        if sc.vendor_utility(s, x) >= s.vendor.batna - 1e-9 and sc.buyer_utility(s, x) > best_ub:
            best_ub, best_x = sc.buyer_utility(s, x), x
    return best_x


def sweep(n=3000, seed=7):
    rng = random.Random(seed)
    scenarios = [sc.sample_scenario(rng) for _ in range(n)]
    policies = [
        ("naive split [0.5]x8",         naive),
        ("blind constant [0.6]x8",      blind_constant(0.6)),
        ("counter-reading heuristic",   counter_reader),
        ("logroll oracle (full info)",  logroll_oracle),
    ]
    print(f"\nRedlineBench v2 reference baselines"
          f"\n{n} random scenarios, 4 rounds, real reward (scoring.py), no API.\n")
    print(f"{'policy':<30}{'reward':>9}{'deal_rate':>11}{'logroll':>9}")
    print("-" * 59)
    for name, pol in policies:
        rs, deals, lrs = [], [], []
        for s in scenarios:
            o = run_episode(s, pol)
            rs.append(o.buyer_score); deals.append(1.0 if o.deal_reached else 0.0)
            lrs.append(o.logroll_index)
        print(f"{name:<30}{st.mean(rs):>9.3f}{st.mean(deals):>11.3f}{st.mean(lrs):>9.3f}")
    print("\nReading: the six frontier models score ~0.50 (see redline_v2_baseline.png),"
          "\ni.e. BELOW the rule-based counter-reading heuristic and at the blind-constant"
          "\nlevel. The models don't execute the obvious logroll; the skill ceiling is 1.00.\n")


if __name__ == "__main__":
    sweep()
