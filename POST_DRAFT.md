# RedlineBench v2 — post

**Image:** `redline_v1_v2.png` (your two runs: v1 one-clause prototype barely moving,
v2 full-contract 4B climbing past the frontier-model and 20-line-script reference
lines). Your figure, your data. The alignment proof lives in the copy. The Applied
Compute / Harvey link is made in the words, not by borrowing their chart.

**Headline number:** 0.11 -> ~0.55 (the sustained back-half level). Do NOT cite 0.64;
that was a single-step spike and citing it reads as cherry-picking.

---

## Post (LinkedIn / X long form) — researcher voice, lay can still follow

To train a model on knowledge work you need a grader, and with no answer key that's usually another model. I work on the cases where you don't need one: outcomes you can score with math and train against directly.

Applied Compute and Harvey just showed how strong the judge-based version is, training an open model past GPT-5.5 on Harvey's legal benchmark. Negotiation doesn't need that judge. The value of a settlement is computable, so the reward is arithmetic on the final deal.

My first version was a single clause, and mostly showed how easily a model games the reward instead of earning it. The second is a full contract, eight weighted terms against a counterparty that counters and walks. I trained a 4B on it.

The frontier models never learned to trade: GPT-5, Claude, and the rest score below a 20-line script that just concedes whatever the other side doesn't care about. The 4B did learn. Buyer score rose from 0.11 to about 0.55, past the script, by giving ground on the terms the counterparty valued most, which is the part you can't fake by being aggressive.

Synthetic, one slice, and real legal work still needs a judge. But wherever the outcome is computable, the reward costs nothing and can't be hacked, and it still teaches a small model the skill.

Writeup and environment in the comments.

---

## Framing guardrails (what NOT to say)

- Do not claim your result matches or rivals Harvey/AC. Different task, different
  metric, different scale. The connection is the *idea* (score interpretive work),
  not the numbers.
- Do not put your synthetic toy and their real-LAB result on one shared axis as if
  comparable. The link goes in the prose.
- Keep the scope caveat in the post (synthetic, one slice, real work still needs a
  judge). It is what makes the rest credible.

## First comment (paste under the post)

Environment, code, and the full writeup, including the training result and the honest caveats: https://app.primeintellect.ai/dashboard/environments/fa1zvn/redline-v2

Run it yourself: prime env install fa1zvn/redline-v2

v1, the single-clause version and the reward-hacking postmortem that taught me to validate the reward before training: https://github.com/fa1zn/redlinebench
