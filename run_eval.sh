#!/usr/bin/env bash
# Re-eval frontier models on the HARDENED RedlineBench v2 (local code, not the
# Hub wheel). Uses Prime Inference, billed to your Prime account. Cheap models
# run first so you can Ctrl-C after them if you only want the cheap signal.
#
# This is the decisive, low-cost spend: it is both the headline finding AND the
# training-readiness check. After it runs:  python summarize_eval.py
#
# Override scenario count:  N=72 ./run_eval.sh     (default 50, ~a few dollars total)
set -uo pipefail
cd "$(dirname "$0")"

N="${N:-50}"
OUT="outputs/hardened_eval"          # separate dir: never mixed with old soft-vendor runs

# Cheap -> expensive. Stop early (Ctrl-C) after the cheap ones if you want.
MODELS=(
  "openai/gpt-4.1-nano"
  "openai/gpt-4.1-mini"
  "anthropic/claude-haiku-4.5"
  "deepseek/deepseek-v4-flash"
  "openai/gpt-5"
  "anthropic/claude-sonnet-4.5"
)

# Ensure the eval runs against the LOCAL hardened code (vendor BATNA ~0.5),
# not the older wheel published to the Hub.
prime env install . >/dev/null 2>&1 || true

for m in "${MODELS[@]}"; do
  echo ">>> $m  on hardened redline-v2  (n=$N)"
  prime eval run redline-v2 \
    --env-dir-path .. \
    --provider prime --model "$m" \
    --num-examples "$N" --rollouts-per-example 1 \
    --max-tokens 8000 --temperature 0.7 \
    `# 8000 tokens: reasoning models (gpt-5, o-series) need room to think AND` \
    `# emit the answer. The env tolerates content=None either way, but a low` \
    `# budget makes reasoning models fall back to a naive offer (understating them).` \
    --output-dir "$OUT" --save-results \
    || echo "!! $m failed (continuing)"
done

echo
echo "All done. Now run:  python summarize_eval.py"
