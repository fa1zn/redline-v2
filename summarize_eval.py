"""
Summarize the hardened-vendor model eval next to the free reference policies,
and give a training-readiness verdict.

Run AFTER run_eval.sh. The reference policies (constant / naive / counter-reading
bot / oracle) use NO API, so this is free to run as many times as you like; it
just folds in whatever model results exist under outputs/hardened_eval/.

Two questions it answers in one table:
  1. Finding:   where do the frontier models land vs a 20-line rule-based bot?
  2. Readiness: are the models' deal rates in the trainable band (so GRPO will
                see within-group reward variance), or in the dead zone (~0)?
"""
import json, glob, os, random, statistics as st
import scoring as sc
import baselines as bl

MODEL_DIR = "outputs/hardened_eval"

REFERENCE = [
    ("blind constant [0.6]x8", bl.blind_constant(0.6)),
    ("naive split [0.5]x8",    bl.naive),
    ("counter-reading bot",    bl.counter_reader),
    ("logroll oracle",         bl.logroll_oracle),
]


def reference_rows(n=3000, seed=7):
    rng = random.Random(seed)
    scen = [sc.sample_scenario(rng) for _ in range(n)]
    rows = []
    for name, pol in REFERENCE:
        rs, dr = [], []
        for s in scen:
            o = bl.run_episode(s, pol)
            rs.append(o.buyer_score); dr.append(1.0 if o.deal_reached else 0.0)
        rows.append((name, st.mean(rs), st.mean(dr), n, "ref"))
    return rows


def model_rows(d=MODEL_DIR):
    best = {}
    for f in glob.glob(os.path.join(d, "**", "results.jsonl"), recursive=True):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if not recs:
            continue
        model = next((p for p in f.split(os.sep) if "--" in p), os.path.dirname(f))
        rw = [r.get("reward", 0.0) for r in recs]
        dr = [((r.get("metrics") or {}).get("deal_rate_metric",
               r.get("deal_rate_metric", 0.0))) for r in recs]
        if model not in best or len(recs) > best[model][3]:
            best[model] = (model.replace("redline-v2--", "").replace("--", "/"),
                           st.mean(rw), st.mean(dr), len(recs), "model")
    return list(best.values())


def main():
    refs = reference_rows()
    mods = model_rows()
    if not mods:
        print(f"No model results in {MODEL_DIR}/ yet. Run ./run_eval.sh first.\n"
              "Showing reference policies only.\n")

    allrows = refs + mods
    allrows.sort(key=lambda r: r[1])  # by reward

    print("\nRedlineBench v2 (hardened vendor) -- models vs reference policies\n")
    print(f"{'policy / model':<28}{'reward':>8}{'deal%':>8}{'n':>6}   kind")
    print("-" * 62)
    bot = next((r for r in refs if r[0] == "counter-reading bot"), None)
    for name, rw, dr, n, kind in allrows:
        star = "  <- 20-line bot" if kind == "ref" and name == "counter-reading bot" else ""
        print(f"{name:<28}{rw:>8.3f}{dr*100:>7.0f}%{n:>6}   {kind}{star}")

    # ---- training-readiness verdict ----
    print("\n" + "=" * 62)
    if mods:
        in_band = [m for m in mods if 0.12 <= m[2] <= 0.92]
        dead    = [m for m in mods if m[2] < 0.05]
        print(f"TRAINING READINESS: {len(in_band)}/{len(mods)} models in the trainable "
              f"deal-rate band (0.12-0.92).")
        if dead:
            print(f"  DEAD ZONE: {len(dead)} model(s) close <5% of deals -> GRPO would see")
            print(f"  near-zero within-group variance. Soften the vendor BATNA or add a")
            print(f"  curriculum BEFORE spending on a training run.")
        if len(in_band) >= max(1, len(mods) // 2):
            print("  -> Green light: enough signal for a training run to catch.")
        else:
            print("  -> Hold: not enough models in the trainable band. Fix the env first.")
        if bot:
            beaten = [m for m in mods if m[1] < bot[1]]
            print(f"\nFINDING: the 20-line counter-reading bot scores {bot[1]:.3f}; "
                  f"{len(beaten)}/{len(mods)} frontier models score BELOW it.")
    else:
        print("Run ./run_eval.sh to get the model rows and the readiness verdict.")
    print()


if __name__ == "__main__":
    main()
