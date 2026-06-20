import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
})

# Real v2 BASELINE eval data, n=72 negotiations/model, no prompt optimization.
# (model, logroll_index, buyer_reward, sem)  -- means + standard error over 72 scenarios.
models = [
    ("claude-sonnet-4.5",  0.365, 0.521, 0.014),
    ("gpt-5",              0.364, 0.515, 0.031),
    ("deepseek-v4-flash",  0.394, 0.506, 0.020),
    ("gpt-4.1-mini",       0.372, 0.469, 0.025),
    ("claude-haiku-4.5",   0.286, 0.461, 0.016),
    ("gpt-4.1-nano",       0.303, 0.375, 0.019),
]
# GEPA prompt-optimized gpt-4.1-mini -- shown separately, NOT a baseline point.
GEPA = ("gpt-4.1-mini + GEPA", 0.468, 0.555, 0.038)
SPLIT = 0.361   # mean buyer_score of the naive 50/50 split, 4000 scenarios
CEIL  = 1.00    # efficient-frontier ceiling

RED, GREEN, BLUE = "#d1453b", "#2e8b57", "#2f6fb0"

fig, ax = plt.subplots(figsize=(11.0, 6.6), dpi=130)
fig.subplots_adjust(top=0.84, left=0.085, right=0.985, bottom=0.12)

# reference lines
ax.axhline(CEIL, ls="--", lw=1.3, color=GREEN, alpha=0.9, zorder=1)
ax.axhline(SPLIT, ls=":", lw=1.4, color="#888888", zorder=1)

# shaded "value left on the table" band between best baseline model and ceiling
best = max(m[2] for m in models)
ax.axhspan(best, CEIL, color=RED, alpha=0.05, zorder=0)

# naive split marker at logroll 0
ax.scatter([0.0], [SPLIT], marker="*", s=260, color="#888888", zorder=5,
           edgecolor="white", linewidth=0.6)

# baseline model points with standard-error bars
xs = [m[1] for m in models]; ys = [m[2] for m in models]; es = [m[3] for m in models]
ax.errorbar(xs, ys, yerr=es, fmt="o", ms=9, color=BLUE, zorder=5,
            ecolor="#9bb8d6", elinewidth=1.4, capsize=3,
            markeredgecolor="white", markeredgewidth=0.8)

# GEPA point, drawn distinctly, with an arrow from the gpt-4.1-mini baseline
mini = next(m for m in models if m[0] == "gpt-4.1-mini")
ax.errorbar([GEPA[1]], [GEPA[2]], yerr=[GEPA[3]], fmt="D", ms=9, color=GREEN, zorder=6,
            ecolor="#9cc4ad", elinewidth=1.4, capsize=3,
            markeredgecolor="white", markeredgewidth=0.8)
ax.annotate("", xy=(GEPA[1], GEPA[2]), xytext=(mini[1], mini[2]),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3, alpha=0.8))

# label each baseline model in a clean right-hand stack, ordered to avoid crossings
label_y = {
    "claude-sonnet-4.5": 0.655,
    "gpt-5":             0.590,
    "deepseek-v4-flash": 0.525,
    "gpt-4.1-mini":      0.455,
    "claude-haiku-4.5":  0.390,
    "gpt-4.1-nano":      0.320,
}
LX = 0.560
for name, lr, rw, sem in models:
    ax.annotate(f"{name}   {rw:.3f}", xy=(lr, rw), xytext=(LX, label_y[name]),
                fontsize=11, va="center", color="#222222",
                arrowprops=dict(arrowstyle="-", color="#c2c2c2", lw=0.8))

# reference-line labels
ax.text(0.012, CEIL - 0.02, "efficient ceiling = 1.00   (most an agent could win for its client)",
        color=GREEN, fontsize=10.5, va="top")
ax.text(0.012, SPLIT - 0.022, "naive 50/50 split = 0.36", color="#6f6f6f", fontsize=10.5,
        ha="left", va="top")

# story annotations
ax.annotate("every frontier model\nbeats the dumb split...",
            xy=(0.303, 0.375), xytext=(0.085, 0.245),
            fontsize=11, color=BLUE, ha="left",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
ax.annotate("...but the top models cluster near\nhalf the value and are statistically tied\n(error bars overlap), and none fully\ntrade across terms (logroll < 0.5)",
            xy=(0.372, 0.515), xytext=(0.085, 0.80),
            fontsize=11, color=RED, ha="left",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
ax.annotate("GEPA prompt-optimization:\n+0.09, but more aggressive\n(fewer deals closed)",
            xy=(GEPA[1], GEPA[2]), xytext=(0.585, 0.86),
            fontsize=10.5, color=GREEN, ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))

ax.set_xlim(-0.02, 0.82)
ax.set_ylim(0.16, 1.06)
ax.set_xlabel("logroll index   (0 = split every term 50/50,   higher = trades low-value terms for high-value ones)",
              fontsize=11)
ax.set_ylabel("buyer reward   (client value captured, 0 to 1)", fontsize=11.5)
ax.grid(True, color="#e9e9e9", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# title + config subtitle, manually placed so they never collide
fig.text(0.085, 0.945, "RedlineBench v2: do frontier models negotiate, or just split the difference?",
         fontsize=15, fontweight="bold", ha="left")
fig.text(0.085, 0.895, "8-term MSA contract   ·   verifiable reward, no judge   ·   72 negotiations/model, error bars = ±1 s.e.",
         fontsize=11, color="#555555", ha="left")

out = "redline_v2_baseline.png"
fig.savefig(out, facecolor="white")
import shutil, os
dst = os.path.expanduser("~/Desktop/redline_v2_baseline.png")
shutil.copy(out, dst)
print("saved:", os.path.abspath(out), "->", dst)
