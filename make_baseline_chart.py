"""
Skill-gradient chart for RedlineBench v2 (hardened vendor).

Shows what the reward actually requires, measured with no API and no judge
(see baselines.py): a flat offer the vendor walks away from scores ~0, naive
splitting scores low, a ~20-line rule-based logroller that reads the vendor's
counters scores in the middle, and optimal logrolling tops out at 1.0. The gap
between the rule-based logroller and the ceiling is the headroom; the gap below
it is the cost of not trading at all.

Frontier-model results under this hardened vendor are pending (see README).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
})

# Verified reference policies under the hardened vendor (vendor BATNA ~0.5).
# (label, buyer_reward, deal_rate, color)  -- from `python baselines.py`, n=3000.
GRAY, BLUE, GREEN = "#9aa3af", "#2f6fb0", "#2e8b57"
rows = [
    ("blind constant\n(vendor walks)",        0.00, 0.00, GRAY),
    ("naive 50/50 split",                      0.24, 0.51, GRAY),
    ("rule-based logroller\n(reads counters)", 0.46, 0.62, BLUE),
    ("logroll oracle\n(optimal play)",         1.00, 1.00, GREEN),
]

fig, ax = plt.subplots(figsize=(10.4, 5.6), dpi=130)
fig.subplots_adjust(top=0.80, left=0.30, right=0.965, bottom=0.12)

ys = list(range(len(rows)))
labels = [r[0] for r in rows]
vals   = [r[1] for r in rows]
colors = [r[3] for r in rows]

bars = ax.barh(ys, vals, color=colors, height=0.62, edgecolor="white", linewidth=0.8, zorder=3)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=11.5)

# value + deal-rate labels at the end of each bar
for y, (lab, rw, dr, c) in zip(ys, rows):
    txt = f"{rw:.2f}" + (f"   ({dr*100:.0f}% close)" if dr > 0 else "   (no deal)")
    ax.text(rw + 0.015, y, txt, va="center", ha="left", fontsize=11, color="#222222")

# ceiling line
ax.axvline(1.0, ls="--", lw=1.2, color=GREEN, alpha=0.8, zorder=1)

ax.set_xlim(0, 1.5)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("buyer reward   (client value captured, 0 to 1)   ·   verifiable, no judge", fontsize=11)
ax.grid(True, axis="x", color="#ececec", lw=0.6, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(left=False)

fig.text(0.30, 0.93, "RedlineBench v2: what does the reward actually require?",
         fontsize=15, fontweight="bold", ha="left")
fig.text(0.30, 0.865,
         "8-term contract · vendor walks from flat offers · a 20-line bot that reads the\n"
         "counters beats not-trading; optimal logrolling is the 1.0 ceiling. No API, no judge.",
         fontsize=10.5, color="#555555", ha="left")

out = "redline_v2_baseline.png"
fig.savefig(out, facecolor="white")
import os
print("saved:", os.path.abspath(out))
