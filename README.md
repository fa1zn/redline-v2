# RedlineBench v2

Multi-term contract negotiation, scored by a verifiable outcome instead of an AI judge.

## The idea

How do you tell whether an AI did a good job at negotiating? With math or code you can check the answer. A negotiation has no obvious right answer, so normally a person or another AI reads it and gives an opinion. That is slow, and a little subjective.

The first version scored a negotiation as a single number. That worked, but it was the easy case. This version handles a whole contract with eight terms and no single right answer. There is still nobody judging it. The score comes from the deal itself, by measuring how close the two sides got to the best deal they both could have accepted. The one human step is deciding up front how much each term is worth. After that, the negotiation scores itself.

## What's here so far

- A scorekeeper that grades a finished contract from the client's side, with no judge
- Eight terms the two sides weight differently, so the skill being tested is trading across them, not splitting them
- An opposing-counsel vendor that counters offers and can walk away
- A baseline across six frontier models, and a prompt-optimization result that moves the score

## Status

Environment, baseline, and a GEPA result complete. The six models tested all negotiate about as well as splitting every term 50/50, and none of them trade across terms much. They reach efficient deals but capture little of the value for their own client. Optimizing the prompt with GEPA lifted one model from 0.49 to 0.55, but made it more aggressive: more value per deal, fewer deals closed.

Caveats: the opposing counsel is a fixed rule-based policy and the scenarios are synthetic, so this is a research setup, not solved contract negotiation.

Live on the Prime Intellect Hub: `prime env install fa1zvn/redline-v2`. Next: self-play training, and grounding the term values in a real contract playbook.
