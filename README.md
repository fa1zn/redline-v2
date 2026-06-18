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

### Results

Baseline across six models, 24 scenarios with 3 rollouts each, uniform 4000 max tokens. Reward is client surplus captured, 0 to 1. For reference, a model that splits every term 50/50 scores about 0.43, and a perfect-trade oracle scores 1.0.

| model | reward (avg ± std) | efficiency | logroll | deal rate |
|---|---|---|---|---|
| claude-sonnet-4.5 | 0.521 ± 0.116 | 0.813 | 0.365 | 1.000 |
| openai/gpt-5 | 0.515 ± 0.260 | 0.670 | 0.364 | 0.819 |
| deepseek-v4-flash | 0.506 ± 0.167 | 0.793 | 0.394 | 0.958 |
| openai/gpt-4.1-mini | 0.494 ± 0.205 | 0.749 | 0.397 | 0.917 |
| claude-haiku-4.5 | 0.455 ± 0.143 | 0.768 | 0.274 | 0.958 |
| openai/gpt-4.1-nano | 0.375 ± 0.163 | 0.759 | 0.303 | 0.986 |

Two things stand out. The top three overlap inside their error bars, so the frontier is a statistical tie. And every model lands just above the naive 50/50 line and far below the oracle, with logrolling never passing 0.40. The models grow the joint pie but capture little of it for their own client, and they barely trade across terms.

Optimizing the system prompt with GEPA (gpt-4.1-mini, prompt only, no weight training) moved the number and exposed a tradeoff:

| gpt-4.1-mini | reward | efficiency | logroll | deal rate |
|---|---|---|---|---|
| baseline prompt | 0.494 | 0.749 | 0.397 | 0.917 |
| GEPA-optimized | 0.555 | 0.655 | 0.468 | 0.764 |

Client value rose 12% and the model traded across terms more, but it also walked away from more deals. GEPA turned a mild negotiator into an aggressive one: higher value per closed deal, fewer deals closed.

### Quickstart

```bash
prime env install fa1zvn/redline-v2
prime eval run redline-v2
```

### Files
- `scoring.py`: pure-Python reward core (deal model, Pareto frontier, scorer). Runs standalone.
- `redline_v2.py`: the `vf.MultiTurnEnv` wrapper, opposing-counsel policy, dataset, and `load_environment`.
- `SPEC.md`: design notes and the milestone ladder toward self-play.

### Limitations
Be aware of what this version does and does not model:
- The opposing counsel is a deterministic rule-based policy, not a learned or LLM agent. It concedes on the terms it weights low and holds firm on the ones it weights high. This is enough to make logrolling pay off and to leak priority signal, but a trained model can learn to exploit a fixed opponent. Self-play is the planned fix (see SPEC.md).
- Scenarios are synthetic: term weights and BATNAs are sampled, not drawn from real contracts. The verifiable reward is real, but the term valuations are stand-ins for a real playbook. Grounding the weights in an actual MSA playbook is the planned step toward realism.
- Term values are continuous in [0,1] and treated as independent. Real contract terms interact, and the language-to-value mapping is itself a judgment task this version abstracts away.

This is a research environment for studying verifiable-reward negotiation, not a claim that contract negotiation is solved.
