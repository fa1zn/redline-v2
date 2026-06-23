"""
RedlineBench v2: multi-term negotiation scoring (step 1).

The whole point of v2: turn the single-scalar liability-cap reward (which
collapsed to a midpoint-split exploit) into a multi-issue reward whose signal
comes from the *economic structure of the negotiation*, not from a judge.

Core idea (Raiffa, multi-issue bargaining):
- A deal is a vector x of N terms, each normalized so x_i in [0,1],
  1.0 = fully BUYER-favorable, 0.0 = fully VENDOR-favorable.
- Each side privately weights the terms (w_buyer, w_vendor). They care about
  DIFFERENT terms. That asymmetry is what makes trading ("logrolling") pay.
- Each side has a BATNA (reservation utility): the value of walking away.

A naive agent splits every term 50/50. That deal sits strictly INSIDE the
Pareto frontier, so both sides leave joint surplus on the table. A smart agent
trades: concede the terms it weights low, hold the terms it weights high, and
the joint outcome moves OUT to the frontier. Because that frontier deal can
give the buyer more utility for the *same* vendor utility, pure self-interest
plus the ability to package a trade is enough to produce logrolling.

This module is dependency-free so it runs on its own:  `python scoring.py`
It plugs into vf.MultiTurnEnv as the reward func (see SPEC.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Deal / term model
# ---------------------------------------------------------------------------

@dataclass
class Term:
    """One negotiable term, normalized to [0,1] where 1.0 favors the buyer."""
    name: str
    # Redline: conceding the buyer's position BELOW this value is a must-escalate
    # item. Used by the v2 escalation reward (step 6); ignored by the Pareto core.
    escalate_below: float | None = None


@dataclass
class Profile:
    """A party's private valuation of the deal."""
    weights: list[float]          # importance per term, will be normalized to sum 1
    batna: float                  # reservation utility in [0,1]; deal must beat this

    def normalized(self) -> list[float]:
        s = sum(self.weights)
        return [w / s for w in self.weights] if s else self.weights


@dataclass
class Scenario:
    terms: list[Term]
    buyer: Profile
    vendor: Profile
    label: str = ""
    # cached after construction
    wb: list[float] = field(default_factory=list)
    wv: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.wb = self.buyer.normalized()
        self.wv = self.vendor.normalized()

    def to_info(self) -> dict:
        """Serialize to a JSON-safe dict for a verifiers dataset `info` field."""
        return {
            "label": self.label,
            "terms": [{"name": t.name, "escalate_below": t.escalate_below}
                      for t in self.terms],
            "buyer": {"weights": self.buyer.weights, "batna": self.buyer.batna},
            "vendor": {"weights": self.vendor.weights, "batna": self.vendor.batna},
        }


def scenario_from_info(d: dict) -> Scenario:
    """Rebuild a Scenario from a `to_info()` dict."""
    return Scenario(
        terms=[Term(t["name"], t.get("escalate_below")) for t in d["terms"]],
        buyer=Profile(d["buyer"]["weights"], d["buyer"]["batna"]),
        vendor=Profile(d["vendor"]["weights"], d["vendor"]["batna"]),
        label=d.get("label", ""),
    )


# Fixed realistic MSA term set. x=1.0 is fully buyer-favorable.
DEFAULT_TERMS = [
    Term("liability_cap", escalate_below=0.2),
    Term("sla_credits"),
    Term("governing_law", escalate_below=0.1),
    Term("ip_indemnity", escalate_below=0.15),
    Term("payment_net_days"),
    Term("termination_notice"),
    Term("auto_renewal"),
    Term("warranty_period"),
]


def sample_scenario(rng: random.Random, terms: list[Term] | None = None) -> Scenario:
    """Sample a scenario with random private weights so the agent must infer them.
    Buyer and vendor priorities are drawn independently, so on most scenarios the
    two sides weight different terms and logrolling pays off."""
    terms = terms or DEFAULT_TERMS
    n = len(terms)
    wb = [rng.random() for _ in range(n)]
    wv = [rng.random() for _ in range(n)]
    return Scenario(
        terms=terms,
        buyer=Profile(weights=wb, batna=round(rng.uniform(0.2, 0.4), 3)),
        # Vendor walkaway is set above what a flat, untraded offer yields (a
        # constant 0.6 gives the vendor only 0.4), so clearing it requires
        # conceding the vendor's priorities, i.e. actually trading.
        vendor=Profile(weights=wv, batna=round(rng.uniform(0.45, 0.55), 3)),
        label=f"sampled-{n}term",
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def buyer_utility(s: Scenario, x: list[float]) -> float:
    """Buyer gains as terms move toward 1.0."""
    return sum(w * xi for w, xi in zip(s.wb, x))


def vendor_utility(s: Scenario, x: list[float]) -> float:
    """Vendor gains as terms move toward 0.0."""
    return sum(w * (1.0 - xi) for w, xi in zip(s.wv, x))


# ---------------------------------------------------------------------------
# Pareto frontier  (analytic, via the lambda-sweep / greedy-assignment trick)
# ---------------------------------------------------------------------------
# For a weight lambda on the buyer, the deal that maximizes
#   lambda * U_buyer + (1-lambda) * U_vendor
# assigns each term independently: x_i = 1 if lambda*wb_i >= (1-lambda)*wv_i
# else x_i = 0. Sweeping lambda from 0..1 traces the full (piecewise-linear)
# Pareto frontier of achievable (U_buyer, U_vendor) outcomes.

def frontier(s: Scenario, steps: int = 400) -> list[tuple[float, float]]:
    """Return Pareto-frontier points as sorted (U_vendor, U_buyer) pairs."""
    pts = set()
    for k in range(steps + 1):
        lam = k / steps
        x = [1.0 if lam * wb >= (1 - lam) * wv else 0.0
             for wb, wv in zip(s.wb, s.wv)]
        pts.add((round(vendor_utility(s, x), 9), round(buyer_utility(s, x), 9)))
    # upper envelope: for each U_vendor keep the max U_buyer
    by_uv: dict[float, float] = {}
    for uv, ub in pts:
        by_uv[uv] = max(by_uv.get(uv, -1.0), ub)
    return sorted(by_uv.items())


def max_buyer_at(frontier_pts: list[tuple[float, float]], uv_target: float) -> float:
    """Max buyer utility achievable on the frontier at vendor utility >= uv_target.
    This is the buyer utility a fully efficient deal would give while leaving the
    vendor exactly as satisfied as the achieved deal did."""
    best = 0.0
    for uv, ub in frontier_pts:
        if uv >= uv_target - 1e-9:
            best = max(best, ub)
    return best


# ---------------------------------------------------------------------------
# The reward
# ---------------------------------------------------------------------------

@dataclass
class Outcome:
    deal_reached: bool
    u_buyer: float
    u_vendor: float
    pareto_gap: float        # buyer surplus left on the table vs. an efficient deal
    efficiency: float        # captured joint surplus / max joint surplus  (0..1)
    buyer_score: float       # the RL reward for the buyer agent  (0..1)
    nash_product: float      # (Ub-batna_b)*(Uv-batna_v); 0 if either walks
    logroll_index: float     # 0 = split-everything-50/50, 1 = fully traded


def score(s: Scenario, x: list[float] | None, deal_reached: bool = True) -> Outcome:
    """Score a finished negotiation. `x` is the agreed term vector (None if no deal)."""
    bb, bv = s.buyer.batna, s.vendor.batna

    if not deal_reached or x is None:
        # Both sides fall back to BATNA. Buyer score is its walkaway value,
        # so walking is better than a deal worse than BATNA, but worse than a real win.
        return Outcome(False, bb, bv, 0.0, 0.0, 0.0, 0.0, 0.0)

    ub, uv = buyer_utility(s, x), vendor_utility(s, x)

    # Individual rationality: a deal below either BATNA wouldn't be signed.
    if ub < bb - 1e-9 or uv < bv - 1e-9:
        return Outcome(False, bb, bv, 0.0, 0.0, 0.0, 0.0, 0.0)

    fr = frontier(s)

    # Efficiency: did the deal reach the frontier, vs. the best joint surplus.
    max_joint = max(ub_ + uv_ for uv_, ub_ in fr)
    efficiency = (ub + uv) / max_joint if max_joint else 0.0

    # Pareto gap: at the vendor satisfaction this deal delivered, how much MORE
    # buyer utility a fully efficient deal would have captured. >0 => money left
    # on the table; this is exactly what a logrolling agent recovers.
    ub_frontier = max_buyer_at(fr, uv)
    pareto_gap = max(0.0, ub_frontier - ub)

    # Buyer RL reward: surplus captured vs. the most the buyer could get while
    # keeping the vendor individually rational. Normalized to [0,1].
    ub_ceiling = max_buyer_at(fr, bv)            # best feasible buyer outcome
    denom = ub_ceiling - bb
    buyer_score = (ub - bb) / denom if denom > 1e-9 else 0.0
    buyer_score = min(1.0, max(0.0, buyer_score))

    nash_product = max(0.0, ub - bb) * max(0.0, uv - bv)

    # Logroll index: how far the deal is from splitting every term at 0.5.
    # 0 => exactly the naive midpoint; ~1 => terms pushed to their extremes (traded).
    spread = sum(abs(xi - 0.5) for xi in x) / (0.5 * len(x))

    return Outcome(True, ub, uv, pareto_gap, efficiency,
                   buyer_score, nash_product, round(spread, 4))


def concession_alignment(s: Scenario, x: list[float]) -> float:
    """Faithful trade-quality metric for behavioral analysis.

    The raw logroll index measures distance from the 50/50 midpoint, which is
    just extremeness: an aggressive policy that pushes every term to an extreme
    scores high without doing any real trading. This instead asks the question
    that actually defines logrolling: did the buyer concede on the terms the
    vendor values MORE than the buyer does?

    Pearson correlation across terms between concession (1 - x_i) and
    (vendor_weight_i - buyer_weight_i). ~0.5+ = genuine logrolling, ~0 = pushing
    terms to extremes without reading the other side. Not foolable by extremeness.
    """
    import math
    n = len(s.terms)
    if n == 0:
        return 0.0
    conc = [1.0 - xi for xi in x]
    diff = [s.wv[i] - s.wb[i] for i in range(n)]
    mc, md = sum(conc) / n, sum(diff) / n
    num = sum((conc[i] - mc) * (diff[i] - md) for i in range(n))
    dc = math.sqrt(sum((conc[i] - mc) ** 2 for i in range(n)))
    dd = math.sqrt(sum((diff[i] - md) ** 2 for i in range(n)))
    return num / (dc * dd) if dc > 1e-9 and dd > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Demo: prove the midpoint exploit is Pareto-dominated
# ---------------------------------------------------------------------------

def _demo_scenario() -> Scenario:
    # 5 realistic MSA terms. Buyer and vendor care about DIFFERENT ones.
    terms = [
        Term("liability_cap",     escalate_below=0.2),
        Term("sla_credits"),
        Term("governing_law",     escalate_below=0.1),
        Term("ip_indemnity",      escalate_below=0.15),
        Term("payment_net_days"),
    ]
    # Buyer cares most about the cap; vendor cares most about governing law.
    buyer  = Profile(weights=[0.40, 0.25, 0.05, 0.20, 0.10], batna=0.30)
    vendor = Profile(weights=[0.10, 0.15, 0.45, 0.10, 0.20], batna=0.30)
    return Scenario(terms, buyer, vendor, label="MSA-5term")


def _frontier_deal_at(s: Scenario, uv_floor: float) -> list[float]:
    """Recover the term vector that maxes buyer utility on the frontier while
    keeping vendor utility >= uv_floor."""
    best_x, best_ub = None, -1.0
    for k in range(401):
        lam = k / 400
        x = [1.0 if lam * wb >= (1 - lam) * wv else 0.0
             for wb, wv in zip(s.wb, s.wv)]
        if vendor_utility(s, x) >= uv_floor - 1e-9 and buyer_utility(s, x) > best_ub:
            best_ub, best_x = buyer_utility(s, x), x
    return best_x or [0.5] * len(s.terms)


if __name__ == "__main__":
    s = _demo_scenario()

    midpoint = [0.5] * len(s.terms)
    mid = score(s, midpoint)
    # The deal that Pareto-DOMINATES the midpoint: vendor at least as happy,
    # buyer strictly better.
    dominating = _frontier_deal_at(s, uv_floor=mid.u_vendor)
    # The buyer-maximizing efficient deal: push vendor down to its BATNA.
    buyer_max = _frontier_deal_at(s, uv_floor=s.vendor.batna)

    print(f"Scenario: {s.label}  ({len(s.terms)} terms)\n")
    print(f"{'deal':<24}{'U_buyer':>9}{'U_vendor':>10}{'eff':>7}"
          f"{'gap':>8}{'logroll':>9}{'reward':>9}")
    for name, x in [("naive midpoint (0.5)", midpoint),
                    ("frontier @ same vendor", dominating),
                    ("buyer-max (vendor->BATNA)", buyer_max)]:
        o = score(s, x)
        print(f"{name:<24}{o.u_buyer:>9.3f}{o.u_vendor:>10.3f}"
              f"{o.efficiency:>7.2f}{o.pareto_gap:>8.3f}"
              f"{o.logroll_index:>9.2f}{o.buyer_score:>9.3f}")

    dom = score(s, dominating)
    print(f"\nThe frontier deal leaves the vendor no worse off "
          f"({dom.u_vendor:.3f} >= {mid.u_vendor:.3f}) and hands the buyer "
          f"{dom.u_buyer - mid.u_buyer:+.3f} more utility")
    print(f"(reward {mid.buyer_score:.3f} -> {dom.buyer_score:.3f}). "
          f"The midpoint split is strictly Pareto-dominated.")
    print("No judge, no rubric: the reward falls out of the game's own surplus. "
          "That is the v2 thesis.")
