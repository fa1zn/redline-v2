# RedlineBench v2

Multi-term contract negotiation, scored by a verifiable outcome instead of an AI judge.

## The idea

How do you tell whether an AI did a good job at negotiating? With math or code you can check the answer. A negotiation has no obvious right answer, so normally a person or another AI reads it and gives an opinion. That is slow, and a little subjective.

The first version scored a negotiation as a single number. That worked, but it was the easy case. This version handles a whole contract with eight terms and no single right answer. There is still nobody judging it. The score comes from the deal itself, by measuring how close the two sides got to the best deal they both could have accepted. The one human step is deciding up front how much each term is worth. After that, the negotiation scores itself.

## What's here so far

- A scorekeeper that grades a finished contract from the client's side, with no judge
- Eight terms the two sides weight differently, so the skill being tested is trading across them (logrolling), not splitting them
- An opposing-counsel vendor that counters offers and walks away from flat, untraded offers
- A verified skill gradient (no API, no judge) showing exactly what the reward rewards

![A flat constant the vendor walks from scores 0, naive splitting 0.24, a rule-based logroller that reads the counters 0.46, and optimal logrolling 1.0](redline_v2_baseline.png)

## What the reward actually requires

The vendor's walkaway is set above what a flat, untraded offer yields, so the only
way to close a good deal is to infer the vendor's priorities from its counters and
trade. Four reference policies, scored through the real vendor loop and reward with
no API and no judge (`python baselines.py`, n=3000):

| policy | buyer reward | closes | what it is |
| --- | --- | --- | --- |
| blind constant `[0.6]×8` | **0.00** | 0% | ignores every counter; the vendor just walks |
| naive split `[0.5]×8` | 0.24 | 51% | split every term down the middle |
| **rule-based logroller** | **0.46** | 62% | ~20 lines: reads the counters, concedes what the vendor wants and it values least, holds the rest |
| logroll oracle (full info) | 1.00 | 100% | optimal trade; the ceiling |

The point: not-trading is near-worthless (the vendor walks), a simple counter-reading
heuristic captures real value, and optimal logrolling is the 1.0 ceiling. The reward
has a genuine, climbable skill gradient with no judge anywhere.

## Status

Environment and the verified skill gradient above are complete. The headline open
question: **do frontier models clear the bar a 20-line heuristic sets?** A six-model
sweep was run against an earlier, softer vendor; those numbers are being re-run under
this hardened vendor and are not reported here until they are. (An earlier GEPA
prompt-optimization run lifted gpt-4.1-mini under the old vendor; also pending re-run.)

Caveats: the opposing counsel is a fixed rule-based policy and the scenarios are
synthetic, so this is a research setup, not solved contract negotiation.

Live on the Prime Intellect Hub: `prime env install fa1zvn/redline-v2`. Next: the
frontier-model sweep under the hardened vendor, then self-play training.
