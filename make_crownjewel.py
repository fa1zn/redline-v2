"""3Blue1Brown-style crown-jewel diagram for RedlineBench v2.
The geometry of why splitting every term 50/50 is provably a worse deal."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import scoring as S

# ---- real geometry from the demo scenario ----
s = S._demo_scenario()
fr = S.frontier(s)                                  # (U_vendor, U_buyer) frontier points
fx = [p[0] for p in fr]; fy = [p[1] for p in fr]
mid = (0.5, 0.5)            # 50/50 split  -> reward 0.31
trade = (0.65, 0.85)        # traded deal on the frontier -> reward 0.85
BATNA = 0.30

# ---- palette (manim) ----
BG     = "#0e1117"
BLUE   = "#58c4dd"
RED    = "#fc6255"
GOLD   = "#ffd43b"
GREEN  = "#83c167"
WHITE  = "#ececec"
GRAY   = "#8b94a3"
AXIS   = "#3a4150"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "light",
    "text.color": WHITE,
})

fig, ax = plt.subplots(figsize=(10.2, 7.3), dpi=130)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
fig.subplots_adjust(left=0.085, right=0.97, top=0.80, bottom=0.10)

# ---- arrowed axes ----
ax.annotate("", xy=(1.07, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.4))
ax.annotate("", xy=(0, 1.10), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.4))

# ---- walk-away (BATNA) lines: the deals both sides will actually sign ----
ax.plot([BATNA, BATNA], [0, 1.10], color=GRAY, lw=0.8, ls=(0, (4, 4)), alpha=0.30)
ax.plot([0, 1.07], [BATNA, BATNA], color=GRAY, lw=0.8, ls=(0, (4, 4)), alpha=0.30)
ax.text(BATNA + 0.012, 0.045, "below here a side walks away", color=GRAY,
        fontsize=9.5, alpha=0.8)

# ---- feasible region (deals that exist) ----
ax.fill_between(fx, 0, fy, color=BLUE, alpha=0.055, zorder=1)

# ---- the frontier, with a soft glow ----
ax.plot(fx, fy, color=BLUE, lw=9, alpha=0.12, solid_capstyle="round", zorder=2)
ax.plot(fx, fy, color=BLUE, lw=2.8, solid_capstyle="round", zorder=3)
ax.text(0.105, 1.045, "the efficient frontier", color=BLUE, fontsize=13)

# ---- the arrow from the split up to the frontier ----
arr = FancyArrowPatch(mid, trade, arrowstyle="-|>", mutation_scale=20,
                      lw=2.4, color=GOLD, connectionstyle="arc3,rad=-0.18", zorder=5)
ax.add_patch(arr)
ax.text(0.40, 0.70, "this deal is better\nfor BOTH sides", color=GOLD,
        fontsize=12.5, ha="right", va="center", linespacing=1.25)

# ---- the two deals ----
for (x, y), c in [(mid, RED), (trade, GOLD)]:
    ax.scatter([x], [y], s=420, color=c, alpha=0.16, zorder=5)   # glow
    ax.scatter([x], [y], s=130, color=c, edgecolor=BG, linewidth=1.6, zorder=6)

# split label
ax.text(0.515, 0.435, "split every term 50/50", color=WHITE, fontsize=12.5, va="top")
ax.text(0.515, 0.385, "reward  0.31", color=RED, fontsize=12.5, va="top", weight="bold")
# trade label
ax.text(0.675, 0.865, "trade low-value terms\nfor the ones you care about",
        color=WHITE, fontsize=12.5, va="bottom", linespacing=1.2)
ax.text(0.675, 0.815, "reward  0.85", color=GOLD, fontsize=12.5, va="top", weight="bold")

# ---- axis labels ----
ax.text(1.04, -0.055, "value to the other side", color=GRAY, fontsize=11.5, ha="right")
ax.text(-0.055, 1.10, "value to\nyour client", color=GRAY, fontsize=11.5, ha="center",
        va="top", linespacing=1.15)

ax.set_xlim(-0.07, 1.12)
ax.set_ylim(-0.08, 1.16)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

# ---- title block ----
fig.text(0.085, 0.945, "Why splitting the difference is a losing move",
         fontsize=21, fontweight="bold", color=WHITE)
fig.text(0.085, 0.885,
         "Every contract term is value the two sides weigh differently. The 50/50 split (red) sits inside the",
         fontsize=12.5, color=GRAY)
fig.text(0.085, 0.852,
         "frontier, so a deal exists that both sides prefer (gold). That gap is the reward in RedlineBench v2.",
         fontsize=12.5, color=GRAY)

out = "redline_v2_crownjewel.png"
fig.savefig(out, facecolor=BG)
import shutil, os
dst = os.path.expanduser("~/Desktop/redline_v2_crownjewel.png")
shutil.copy(out, dst)
print("saved:", os.path.abspath(out), "->", dst)
