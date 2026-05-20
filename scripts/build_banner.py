"""Generate the GitHub social-preview banner for the repo.

Output: docs/banner.png  (1280 x 640, GitHub social-preview spec)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "banner.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

BG = "#0E1117"
PANEL = "#161B22"
FG = "#E6EDF3"
ACCENT = "#58A6FF"
ACCENT2 = "#3FB950"
ACCENT3 = "#D29922"
ACCENT4 = "#F85149"
MUTED = "#7D8590"

DPI = 96
W, H = 1280 / DPI, 640 / DPI

fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Accent bar on the left
bar = patches.Rectangle((0, 0), 1.6, 100, facecolor=ACCENT, edgecolor="none")
ax.add_patch(bar)

# Main title
ax.text(5, 78, "FlagOS Track 1", ha="left", va="center",
        fontsize=58, color=FG, weight="bold")
ax.text(5, 66, "20 Triton GPU Operators", ha="left", va="center",
        fontsize=34, color=ACCENT, weight="bold")

# Subtitle
ax.text(5, 56, "FlagOS Open Computing Global Challenge - Season 1",
        ha="left", va="center", fontsize=16, color=FG)
ax.text(5, 51, "Operator Development & Optimization",
        ha="left", va="center", fontsize=14, color=MUTED)

# Tier badges
tiers = [
    ("EASY",   "8 ops",   ACCENT2, 5),
    ("MEDIUM", "8 ops",   ACCENT3, 25),
    ("HARD",   "4 ops",   ACCENT4, 45),
]
for label, count, color, x in tiers:
    rect = patches.FancyBboxPatch(
        (x, 30), 17, 11, boxstyle="round,pad=0.4",
        linewidth=2, edgecolor=color, facecolor=PANEL,
    )
    ax.add_patch(rect)
    ax.text(x + 8.5, 37.5, label, ha="center", va="center",
            fontsize=15, color=color, weight="bold")
    ax.text(x + 8.5, 32.5, count, ha="center", va="center",
            fontsize=12, color=FG)

# Stat panel on the right
rect = patches.FancyBboxPatch(
    (66, 18), 30, 64, boxstyle="round,pad=0.6",
    linewidth=2, edgecolor=ACCENT, facecolor=PANEL,
)
ax.add_patch(rect)
stats = [
    ("128",   "tests pass"),
    ("4",     "dtypes covered"),
    ("Apache-2.0", "licensed"),
    ("PyTorch", "fallback"),
]
for i, (k, v) in enumerate(stats):
    y = 72 - i * 14
    ax.text(81, y, k, ha="center", va="center", fontsize=22,
            color=ACCENT, weight="bold", family="monospace")
    ax.text(81, y - 5.5, v, ha="center", va="center", fontsize=11,
            color=MUTED)

# Footer line: repo + author
ax.text(5, 12, "github.com/ladyFaye1998/flagos-track1",
        ha="left", va="center", fontsize=18, color=ACCENT,
        family="monospace", weight="bold")
ax.text(5, 6, "by Danielle Lesin",
        ha="left", va="center", fontsize=12, color=MUTED)

plt.savefig(OUT, dpi=DPI, facecolor=BG, bbox_inches="tight", pad_inches=0)
plt.close(fig)

size_kb = OUT.stat().st_size / 1024
print(f"Wrote {OUT}  ({size_kb:.1f} KB, 1280x640)")
