"""
Headline chart for RedlineBench v2 (hardened vendor): six frontier models vs a
20-line rule-based bot, all measured with the same verifiable reward (no judge).

Model numbers: prime eval, n=32/model, hardened vendor (vendor BATNA ~0.5),
reasoning models given an 8000-token budget. Reference policies (bot, oracle,
naive) from baselines.py, n=3000, no API.
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

BLUE, GRAY, RED, GREEN = "#2f6fb0", "#9aa3af", "#d1453b", "#2e8b57"

# (label, reward, deal_rate, color) -- frontier models + naive, sorted ascending.
rows = [
    ("gpt-5",              0.017, 0.03, BLUE),
    ("gpt-4.1-nano",       0.156, 0.44, BLUE),
    ("deepseek-v4-flash",  0.162, 0.28, BLUE),
    ("claude-sonnet-4.5",  0.208, 0.38, BLUE),
    ("gpt-4.1-mini",       0.231, 0.44, BLUE),
    ("claude-haiku-4.5",   0.231, 0.50, BLUE),
    ("naive 50/50 split",  0.239, 0.51, GRAY),
]
BOT = 0.464      # 20-line counter-reading bot
ORACLE = 1.00    # optimal logrolling

fig, ax = plt.subplots(figsize=(10.8, 6.0), dpi=130)
fig.subplots_adjust(top=0.80, left=0.215, right=0.965, bottom=0.12)

ys = list(range(len(rows)))
ax.barh(ys, [r[1] for r in rows], color=[r[3] for r in rows], height=0.62,
        edgecolor="white", linewidth=0.8, zorder=3)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=11.5)
for y, (lab, rw, dr, c) in zip(ys, rows):
    ax.text(rw + 0.012, y, f"{rw:.2f}   ({dr*100:.0f}% close)",
            va="center", ha="left", fontsize=10.5, color="#222222")

# the bot and the ceiling as vertical reference lines the models fall short of
ax.axvline(BOT, ls="--", lw=1.6, color=RED, zorder=4)
ax.text(BOT + 0.008, len(rows) - 0.35, f"20-line counter-reading bot = {BOT:.2f}",
        color=RED, fontsize=10.5, va="center", ha="left", fontweight="bold")
ax.axvline(ORACLE, ls="--", lw=1.3, color=GREEN, alpha=0.85, zorder=1)
ax.text(ORACLE - 0.01, 0.15, f"optimal logrolling = {ORACLE:.2f}",
        color=GREEN, fontsize=10, va="center", ha="right")

ax.set_xlim(0, 1.12)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("buyer reward   (client value captured, 0 to 1)   ·   verifiable, no judge", fontsize=11)
ax.grid(True, axis="x", color="#ececec", lw=0.6, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(left=False)

fig.text(0.215, 0.93, "Six frontier models negotiate worse than a 20-line script",
         fontsize=15.5, fontweight="bold", ha="left")
fig.text(0.215, 0.865,
         "RedlineBench v2 · 8-term contract · verifiable reward, no judge · n=32/model.\n"
         "Every model lands below the bot; gpt-5 closes almost no deals by refusing to concede.",
         fontsize=10.5, color="#555555", ha="left")

out = "redline_v2_baseline.png"
fig.savefig(out, facecolor="white")
import os
print("saved:", os.path.abspath(out))
