# RedlineBench v2: Step 1 Spec

Negotiation as a **verifiable-reward, self-play RL game**, demonstrated on contracts.
The reward comes from the economic structure of the deal, not a judge or rubric.

## Why this exists

v1 reduced a negotiation to a single liability cap and rewarded the midpoint
settlement. That collapsed into a degenerate "split the difference" exploit
(baseline 0.46 → 0.681 after patching it). The fix is not a better patch. The
fix is to make the reward multi-dimensional so the midpoint is *provably* a bad
move, and the only way to score well is to negotiate skillfully.

## The model (this directory, `scoring.py`)

- A **deal** is a vector `x` of N terms, each normalized to `[0,1]`, where
  `1.0` = fully buyer-favorable and `0.0` = fully vendor-favorable.
- Each side has a **private weight vector** over the terms and a **BATNA**
  (reservation utility). The two sides weight *different* terms, and that
  asymmetry is what makes trading across issues ("logrolling") profitable.
- `U_buyer(x) = Σ wᵇ·xᵢ`, `U_vendor(x) = Σ wᵛ·(1−xᵢ)`.
- The **Pareto frontier** is computed analytically by sweeping a scalarization
  weight λ and greedily assigning each term to whoever values it more at that λ.
- A deal below either side's BATNA is "not signed" → no-deal payoff.

### The verifiable signals (all judge-free, all reproducible)

| signal | meaning |
|---|---|
| `buyer_score` | **the RL reward.** Buyer surplus captured, normalized to the best deal that still keeps the vendor individually rational. |
| `efficiency` | captured joint surplus ÷ max joint surplus. 1.0 = on the frontier. |
| `pareto_gap` | buyer utility left on the table vs. an efficient deal at the same vendor satisfaction. |
| `nash_product` | `(Ub−batnaᵇ)(Uv−batnaᵛ)`, the classic bargaining-quality measure. |
| `logroll_index` | 0 = split every term 50/50, ~1 = terms traded to extremes. |

Run `python scoring.py` to see the midpoint deal scored as **strictly
Pareto-dominated** by a traded deal: the demo prints reward 0.31 (midpoint) vs
0.85 (frontier deal at the same vendor satisfaction). That single fact is the
thesis.

## Term schema (starting set, realistic MSA terms)

| term | x=1.0 (buyer) | x=0.0 (vendor) | typically valued by |
|---|---|---|---|
| liability_cap | high cap | low cap | buyer |
| sla_credits | large credits | small credits | buyer |
| governing_law | buyer's state | vendor's state | vendor |
| ip_indemnity | uncapped to buyer | capped/excluded | mixed |
| payment_net_days | net-90 | net-30 | vendor |

Add cap-vs-indemnity ordering, termination notice, early-term fee, auto-renewal,
warranty period to scale N up. `escalate_below` on a `Term` marks a redline for
the escalation reward (step 6).

## How it plugs into verifiers (next, not in step 1)

Reuse v1's `vf.MultiTurnEnv` shape. Changes:

1. **State** holds the running `x` vector and each side's private `Profile`
   (sampled per rollout from a distribution → generalization, partial
   observability since the agent never sees the vendor's weights).
2. **`env_response`** drives the opposing-counsel policy. For step 1–4 keep it
   a fixed LLM or rule-based concession policy; for step 5 make it a *learned*
   policy (self-play).
3. **Action parsing**: the agent proposes a full term package each turn
   (JSON block of `{term: value}`), not a single number. Vendor accepts /
   counters / walks.
4. **Reward**: call `score(scenario, final_x, deal_reached).buyer_score`.
   Log the other Outcome fields as metrics. That table is the failure autopsy.

## Milestone ladder

1. **(this)** multi-term model + Pareto scorer, standalone + tested.
2. Frontier models negotiate → show they leave surplus on the table (postable).
3. GEPA on the system prompt for a cheap lift.
4. RL vs. a fixed opponent → first training curve.
5. **Self-play** → headline curve + emergent logrolling/anchoring.
6. Escalation reward: reward calibrated `accept/counter/reject/escalate`,
   punish both over- and under-escalation.

## Anti-goals

- No LLM judge anywhere in the reward path. The whole pitch is "we didn't need one."
- No breadth-without-depth (don't add 20 contract archetypes before self-play works).
- No fancy UI.
