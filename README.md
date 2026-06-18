# RedlineBench v2

Multi-term contract negotiation, scored by a verifiable outcome instead of an AI judge.

## The idea

How do you tell whether an AI did a good job at something like negotiating? With math or code you can just check the answer. A negotiation has no obvious right answer, so normally you would have a person, or another AI, read it over and give an opinion. That is slow, and it is a little subjective.

The first version of this tried the simplest case. The AI negotiated a single dollar amount, and you could score how it did just by looking at the final number, with nobody judging it.

Someone pointed out, fairly, that one number is the easy case. Most real negotiations have a lot of moving pieces at once, and no single outcome you could call correct.

This version handles that harder case. The AI negotiates a whole contract with eight different terms. There is still nobody judging it. The score comes from the deal itself, by measuring how close the two sides got to the best deal they both could have accepted.

That is the whole point. You can put a real score on something fuzzy without a judge, even when there is no clear right answer, and the score is good enough to train an AI to get better at it.

One honest note. A person still has to decide up front how much each term is worth. Once that is set, the negotiation scores itself.

A bit more technically: the first version was a single number where one side's gain was the other side's loss, and it could be gamed by just meeting in the middle. This version has many terms, and the two sides care about different ones, so they can trade. Giving up something you do not mind in order to keep something you do is what makes a good deal here, and splitting everything down the middle does not get you there.

## What's here so far

- A scorekeeper that grades a finished contract from the client's side, with no judge in the loop
- Eight terms the two sides weight differently, so the skill being tested is trading across them, not splitting them
- An opposing-counsel vendor that counters offers and can walk away
- A baseline across six frontier models
- A result showing the score climbs when you optimize the prompt with GEPA, no weight training

## Results

Baseline across six models, 24 scenarios with 3 rollouts each. Reward is client value captured, 0 to 1. For reference, a model that splits every term 50/50 scores about 0.43, and a perfect-trade oracle scores 1.0.

| model | reward (avg ± std) | efficiency | logroll | deal rate |
|---|---|---|---|---|
| claude-sonnet-4.5 | 0.521 ± 0.116 | 0.813 | 0.365 | 1.000 |
| openai/gpt-5 | 0.515 ± 0.260 | 0.670 | 0.364 | 0.819 |
| deepseek-v4-flash | 0.506 ± 0.167 | 0.793 | 0.394 | 0.958 |
| openai/gpt-4.1-mini | 0.494 ± 0.205 | 0.749 | 0.397 | 0.917 |
| claude-haiku-4.5 | 0.455 ± 0.143 | 0.768 | 0.274 | 0.958 |
| openai/gpt-4.1-nano | 0.375 ± 0.163 | 0.759 | 0.303 | 0.986 |

The top three overlap inside their error bars, so they are a tie. Every model lands just above the naive 50/50 line and far below the oracle, and none of them trade across terms much (logroll never passes 0.40). They grow the joint pie but capture little of it for their own client.

Optimizing the prompt with GEPA (gpt-4.1-mini, prompt only, no weight training) moved the score and surfaced a tradeoff:

| gpt-4.1-mini | reward | efficiency | logroll | deal rate |
|---|---|---|---|---|
| baseline prompt | 0.494 | 0.749 | 0.397 | 0.917 |
| GEPA-optimized | 0.555 | 0.655 | 0.468 | 0.764 |

Client value rose 12% and the model traded across terms more, but it also walked away from more deals. The optimized prompt turned a mild negotiator into an aggressive one: more value per closed deal, fewer deals closed.

## Status

Environment, baseline, and a prompt-optimization result complete. Live on the Prime Intellect Hub at `fa1zvn/redline-v2`. Next: training against it with self-play, and grounding the term values in a real contract playbook instead of sampled numbers.

## Limitations

- The opposing counsel is a fixed rule-based policy, not a learned or LLM agent. It concedes on the terms it weights low and holds firm on the ones it weights high. That is enough to make trading pay off, but a trained model can learn to exploit a fixed opponent. Self-play is the planned fix.
- Scenarios are synthetic. The term weights and walkaway points are sampled, not pulled from real contracts. The reward is real, but the values are stand-ins for a real playbook.
- Terms are treated as independent numbers in [0,1]. Real contract terms interact, and turning contract language into a number is itself a judgment step this version skips.

This is a research environment for studying verifiable-reward negotiation, not a claim that contract negotiation is solved.

## Try it

```bash
prime env install fa1zvn/redline-v2
prime eval run redline-v2
```

## Files

- `scoring.py`: the reward core (deal model, Pareto frontier, scorer). Runs on its own with `python scoring.py`.
- `redline_v2.py`: the environment wrapper, the vendor policy, the dataset, and `load_environment`.
- `SPEC.md`: design notes and the plan toward self-play.
