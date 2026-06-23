# RedlineBench v2 — LinkedIn post (draft)

Visual: redline_v2_baseline.png (attach it)

---

Harvey named the open problem in contract AI. It's an agent that can negotiate against a live opponent and learn from it.

I built it. Tested six of the top models, and they all lost to a 20-line script. GPT-5 was the worst, one closed deal out of 32.

It's a controlled setup on purpose. Eight contract terms, a counterparty that counters and walks, and a score that comes from the math of the final deal instead of a human or another model grading it. A verifiable reward for work everyone assumes needs a judge.

The failure is specific. Winning means trading, conceding the terms you don't care about to win the ones you do. The models don't. They split everything down the middle, or like GPT-5 they come down a little but stay too greedy to close, clinging to one term while the deal dies. They sound like they're negotiating. They aren't trading.

This is the bottleneck for deploying negotiation agents. You can't hand a real contract to something that blows up the deal or quietly gives away the store. What's new is you can now measure that without a judge, and train against it.

Next: a model trained to actually trade, then taught when to escalate to a human instead of conceding.

Writeup and environment in comments.

---

## Prep: questions a sharp reader will ask, and plain answers

**"How is this different from / competing with Harvey's LAB?"**
It isn't competing. LAB is the realistic, expert-graded benchmark, real contracts, playbooks, escalation memos. Mine is a minimal open sandbox for one mechanic they flagged as missing: an interactive opponent with a verifiable reward you can train against. Different altitude, same direction. I would never claim it's in LAB's league on realism.

**"Isn't this just a synthetic game with a fixed opponent?"**
Yes, and I say so in the post. Fixed rule-based vendor, synthetic scenarios. It's a controlled probe, not solved negotiation. The point is the scoring method (math, no judge) and that even in this clean setting strong models fail in a legible way.

**"gpt-5 scoring 0.02 sounds like a harness bug."**
It started as one. gpt-5 is a reasoning model; at a low token budget it spent everything on hidden reasoning and returned no answer, which crashed the rollout. I fixed the env to tolerate that and gave it an 8000-token budget. With that fixed it closes 1 of 32 because it anchors at 0.9 to 0.95 and won't concede. Other models, same prompt, close 30 to 50 percent. The zero is real behavior.

**"Is the vendor just too harsh, so nobody can win?"**
No. The 20-line bot wins at 0.46, optimal is 1.0, five of six models close deals. Winnable. gpt-5 throws it away.

**"How is the reward verifiable with no judge?"**
Each side has private weights on the eight terms. Reward is the buyer's captured value, normalized against the best deal on the Pareto frontier the vendor would still accept. Arithmetic on the final contract. No model grades anything.

**"What's next?"**
Train a small model against it and see if it learns to read the counters and trade. The reward has a real gradient (bot 0.46, optimal 1.0) and a small base model sits in a trainable range, so the run should catch signal. Longer term, wire up escalation (knowing when a term is past your authority and you have to escalate instead of concede), which is the part LAB centers.
