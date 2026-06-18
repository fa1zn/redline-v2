"""
RedlineBench v2 negotiation environment.

The buyer's counsel (the model) negotiates a full package of contract terms
against an opposing-counsel vendor. Each side privately weights the terms, so
the only way to score well is to trade across issues (logrolling), not to split
every term down the middle. The reward is verifiable and comes from the deal's
position on the Pareto frontier, with no judge in the loop. See SPEC.md.

This wraps the pure-Python reward core in scoring.py as a vf.MultiTurnEnv,
mirroring the v1 redline_negotiate environment so it installs and runs the same
way: load_environment(...) -> vf.Environment, prime env install, prime eval run.
"""

import re
import json
import random
from typing import Optional

import verifiers as vf
from datasets import Dataset

import scoring as sc


# ============================================================================
# Action parsing
# ============================================================================

def parse_package(text: str, term_names: list[str], fallback: list[float]) -> list[float]:
    """Pull a JSON term package out of the model's message.

    Accepts a fenced or bare JSON object mapping term name -> value in [0,1].
    Missing or malformed terms keep the fallback (previous) value, so a sloppy
    turn never crashes the rollout.
    """
    x = list(fallback)
    if not text:
        return x

    block = None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        block = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            block = brace.group(0)

    if block:
        try:
            obj = json.loads(block)
            for i, name in enumerate(term_names):
                if name in obj:
                    x[i] = min(1.0, max(0.0, float(obj[name])))
        except (ValueError, TypeError):
            pass
    return x


# ============================================================================
# Opposing-counsel policy (rule-based, deterministic)
# ============================================================================

def vendor_decision(
    s: sc.Scenario, buyer_x: list[float], round_num: int, total_rounds: int
) -> tuple[bool, bool, list[float]]:
    """Return (accepted, walked, counter_x).

    The vendor accepts once the buyer's package clears a threshold that softens
    each round. Its counter concedes on the terms it weights low and holds firm
    on the terms it weights high, which leaks its priorities to a buyer paying
    attention. That signal is what makes inference (and logrolling) learnable.
    """
    uv = sc.vendor_utility(s, buyer_x)

    # Threshold starts above BATNA and decays toward it as the deadline nears.
    progress = round_num / max(1, total_rounds)
    threshold = s.vendor.batna + (1.0 - s.vendor.batna) * 0.45 * (1.0 - progress)
    if uv >= threshold:
        return True, False, buyer_x

    if round_num >= total_rounds:
        # Out of rounds: sign only if individually rational, else walk.
        return (uv >= s.vendor.batna), (uv < s.vendor.batna), buyer_x

    # Counter: hold the high-weight half toward vendor-favorable, concede the rest.
    order = sorted(range(len(s.terms)), key=lambda i: s.wv[i], reverse=True)
    hold = set(order[: len(order) // 2])
    counter = [
        min(buyer_x[i], 0.25) if i in hold else max(buyer_x[i], 0.6)
        for i in range(len(s.terms))
    ]
    return False, False, counter


# ============================================================================
# Environment
# ============================================================================

class RedlineV2Env(vf.MultiTurnEnv):
    """Multi-term contract negotiation with a verifiable, frontier-based reward."""

    def __init__(self, dataset, rubric, system_prompt, total_rounds=4,
                 eval_dataset=None, **kwargs):
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            system_prompt=system_prompt,
            eval_dataset=eval_dataset,
            max_turns=total_rounds,
            **kwargs,
        )
        self.total_rounds = total_rounds

    async def setup_state(self, state: vf.State, **kwargs) -> None:
        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
        s = sc.scenario_from_info(info)
        state["scenario_info"] = s.to_info()
        state["term_names"] = [t.name for t in s.terms]
        # Deal on the table starts at the naive midpoint; the agent must improve it.
        state["buyer_x"] = [0.5] * len(s.terms)
        state["round_num"] = 0
        state["deal_reached"] = False
        state["final_x"] = None
        await super().setup_state(state, **kwargs)

    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        s = sc.scenario_from_info(state["scenario_info"])
        names = state["term_names"]
        state["round_num"] += 1
        rn, total = state["round_num"], self.total_rounds

        last = messages[-1].get("content", "") if messages else ""
        buyer_x = parse_package(last, names, state["buyer_x"])
        state["buyer_x"] = buyer_x

        accepted, walked, counter = vendor_decision(s, buyer_x, rn, total)

        if accepted:
            state["deal_reached"] = True
            state["final_x"] = buyer_x
            msg = (f"Round {rn}: opposing counsel ACCEPTS your package. "
                   f"Deal closed.")
            return [{"role": "user", "content": msg}]

        if walked or rn >= total:
            state["deal_reached"] = False
            state["final_x"] = None
            msg = (f"Round {rn}: no agreement reached. Negotiations have ended "
                   f"with no deal.")
            return [{"role": "user", "content": msg}]

        counter_str = json.dumps({names[i]: round(counter[i], 2)
                                  for i in range(len(names))})
        msg = (f"Round {rn} of {total}. Opposing counsel rejects and counters "
               f"with this package (1.0 favors you, 0.0 favors them):\n"
               f"{counter_str}\n"
               f"Send your revised package as a JSON object of term -> value.")
        return [{"role": "user", "content": msg}]


# ============================================================================
# Reward + metrics
# ============================================================================

def _outcome(state: vf.State) -> sc.Outcome:
    s = sc.scenario_from_info(state["scenario_info"])
    return sc.score(s, state.get("final_x"), state.get("deal_reached", False))


async def negotiation_reward(state: vf.State) -> float:
    """Verifiable reward: buyer surplus captured vs. the best feasible deal."""
    return _outcome(state).buyer_score


async def efficiency_metric(state: vf.State) -> float:
    return _outcome(state).efficiency


async def pareto_gap_metric(state: vf.State) -> float:
    return _outcome(state).pareto_gap


async def logroll_metric(state: vf.State) -> float:
    return _outcome(state).logroll_index


async def deal_rate_metric(state: vf.State) -> float:
    return 1.0 if state.get("deal_reached") else 0.0


# ============================================================================
# Dataset
# ============================================================================

def build_dataset(n_scenarios: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_scenarios):
        s = sc.sample_scenario(rng)
        names = [t.name for t in s.terms]
        rows.append({
            "question": (
                "You are negotiating a contract for your client. Propose a full "
                "package covering every term as a JSON object mapping term name "
                "to a value in [0,1], where 1.0 is fully favorable to your client "
                "and 0.0 fully favors the vendor.\n"
                f"Terms: {', '.join(names)}.\n"
                "You do not know which terms the vendor cares about. Read its "
                "counters to infer its priorities, then trade: concede where you "
                "care little, hold where you care a lot. Maximize your client's "
                "value without pushing so hard the vendor walks."
            ),
            "info": json.dumps(s.to_info()),
        })
    return Dataset.from_list(rows)


SYSTEM_PROMPT = (
    "You are skilled contract counsel negotiating a multi-term agreement on "
    "behalf of your client. Each turn, reply with a JSON object mapping every "
    "term to a value in [0,1] (1.0 favors your client, 0.0 favors the vendor). "
    "The vendor weights the terms differently from you, so the best deals come "
    "from trading across issues rather than splitting each term in half. Infer "
    "the vendor's priorities from its counters, give ground on terms you value "
    "little, and hold firm on the ones you value most, without killing the deal."
)


# ============================================================================
# Loader
# ============================================================================

def load_environment(
    total_rounds: int = 4,
    num_train: int = 64,
    num_eval: int = 32,
    seed: int = 0,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> vf.Environment:
    dataset = build_dataset(num_train, seed=seed)
    eval_dataset = build_dataset(num_eval, seed=seed + 1000)

    rubric = vf.Rubric(funcs=[negotiation_reward])
    rubric.add_metric(efficiency_metric)
    rubric.add_metric(pareto_gap_metric)
    rubric.add_metric(logroll_metric)
    rubric.add_metric(deal_rate_metric)

    return RedlineV2Env(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        system_prompt=system_prompt or SYSTEM_PROMPT,
        total_rounds=total_rounds,
        **kwargs,
    )
