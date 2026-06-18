# redline-v2

### Overview
- **Environment ID**: `redline-v2`
- **Short description**: Multi-term contract negotiation where the model trades across interdependent terms against an opposing-counsel vendor, scored by a verifiable Pareto-frontier reward with no judge in the loop.
- **Tags**: multi-turn, negotiation, legal, self-play, train, eval

This is the v2 extension of RedlineBench. v1 negotiated a single liability cap on a one-dimensional, zero-sum line, which collapsed into a midpoint-split exploit. v2 makes the deal a vector of terms that each side weights differently, so the game becomes positive-sum: the only way to score well is to trade across issues (logrolling), which a naive 50/50 split cannot do.

### Datasets
- **Primary dataset(s)**: Procedurally sampled negotiation scenarios. Each scenario draws private buyer and vendor weight vectors over a fixed set of MSA terms, plus a BATNA per side.
- **Source links**: Generated in-environment from `scoring.sample_scenario`.
- **Split sizes**: Train and eval are independently sampled with different seeds (defaults 64 train, 32 eval).

### Task
- **Type**: multi-turn
- **Output format expectations**: A JSON object mapping each term name to a value in [0,1], where 1.0 is fully buyer-favorable and 0.0 fully vendor-favorable.
- **Rubric overview**:
  - `negotiation_reward`: main verifiable reward, buyer surplus captured normalized to the best feasible deal (0.0 to 1.0)
  - `efficiency_metric`: captured joint surplus over max joint surplus
  - `pareto_gap_metric`: buyer utility left on the table vs. an efficient deal at the same vendor satisfaction
  - `logroll_metric`: 0 for a 50/50 split, near 1 when terms are traded to the extremes
  - `deal_rate_metric`: fraction of rollouts that reach a signed deal

### Reward Function (Verifiable)

The reward is computed from the deal's position on the Pareto frontier of the two sides' private utilities. No LLM or rubric grades the text.

| Situation | Reward |
|-----------|--------|
| No deal, or a deal below either side's BATNA | 0.0 |
| Signed deal | buyer surplus over (best feasible buyer utility minus buyer BATNA), clipped to [0,1] |

A naive deal that splits every term at 0.5 sits strictly inside the frontier and scores poorly. A deal that concedes the terms the buyer values little and holds the terms it values highly reaches the frontier and scores high. Run `python scoring.py` to see the midpoint deal scored as Pareto-dominated.

### Quickstart

```bash
prime env install redline-v2
prime eval run redline-v2
```

### Files
- `scoring.py`: pure-Python reward core (deal model, Pareto frontier, scorer). Runs standalone.
- `redline_v2.py`: the `vf.MultiTurnEnv` wrapper, opposing-counsel policy, dataset, and `load_environment`.
- `SPEC.md`: design notes and the milestone ladder toward self-play.
