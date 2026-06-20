# RedlineBench v2

Multi-term contract negotiation, scored by a verifiable outcome instead of an AI judge.

## The idea

How do you tell whether an AI did a good job at negotiating? With math or code you can check the answer. A negotiation has no obvious right answer, so normally a person or another AI reads it and gives an opinion. That is slow, and a little subjective.

The first version scored a negotiation as a single number. That worked, but it was the easy case. This version handles a whole contract with eight terms and no single right answer. There is still nobody judging it. The score comes from the deal itself, by measuring how close the two sides got to the best deal they both could have accepted. The one human step is deciding up front how much each term is worth. After that, the negotiation scores itself.

## What's here so far

- A scorekeeper that grades a finished contract from the client's side, with no judge
- Eight terms the two sides weight differently, so the skill being tested is trading across them (logrolling), not splitting them
- An opposing-counsel vendor that counters offers and can walk away
- A baseline across six frontier models, with reference baselines and error bars
- A GEPA prompt-optimization result, reported separately from the baselines

![Six frontier models cluster near 0.50, statistically tied and no better than a blind constant, far below the logrolling ceiling of 1.0](redline_v2_baseline.png)

## Baselines: what skill is worth here

To make the model numbers legible, three reference policies are scored through the
real vendor loop and reward, with no API and no judge (`python baselines.py`):

| policy | buyer reward | what it is |
| --- | --- | --- |
| naive split `[0.5]×8` | 0.36 | the dumb anchor: split every term down the middle |
| **blind constant `[0.6]×8`** | **0.55** | ignores every counter, never trades; no negotiation skill |
| logroll oracle (full info) | 1.00 | concedes the vendor's priorities, holds its own; the ceiling |

## Status

Environment, baseline, and a GEPA result complete. **The six frontier models all
score ~0.50 — statistically tied with each other (n=72, error bars overlap), and
at the level of a blind constant that ignores the vendor entirely.** They reach
efficient deals but capture little value for their own client, and sit far below
the logrolling ceiling of 1.00. In other words, none of them actually trade across
terms; they negotiate no better than a policy with no negotiation skill. GEPA
prompt-optimization lifted gpt-4.1-mini from its 0.47 baseline to 0.55 (a separate,
annotated point on the chart, not part of the baseline sweep), but made it more
aggressive: more value per deal, fewer deals closed.

Caveats: the opposing counsel is a fixed rule-based policy and the scenarios are
synthetic, so this is a research setup, not solved contract negotiation. A known
limitation the baselines expose: the blind constant reaches 0.55, so the reward
has a flat-ish basin a future RL run could settle into before learning to trade —
hardening the vendor against that (without breaking the logroll ceiling) is the
main open design problem before training.

Live on the Prime Intellect Hub: `prime env install fa1zvn/redline-v2`. Next: self-play training, and grounding the term values in a real contract playbook.
