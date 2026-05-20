"""Generate a 480x480 BUIDL logo for the DoraHacks submission."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = Path(__file__).resolve().parent.parent / "docs" / "logo.png"

BG = "#0E1117"
PANEL = "#161B22"
FG = "#E6EDF3"
ACCENT = "#58A6FF"
ACCENT2 = "#3FB950"
ACCENT3 = "#D29922"
ACCENT4 = "#F85149"
MUTED = "#7D8590"

DPI = 96
W = H = 480 / DPI

fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_aspect("equal"); ax.axis("off")

frame = patches.FancyBboxPatch(
    (3, 3), 94, 94, boxstyle="round,pad=0",
    linewidth=2.5, edgecolor=ACCENT, facecolor=BG,
)
ax.add_patch(frame)

ax.text(50, 78, "FlagOS",
        ha="center", va="center", fontsize=38, color=FG, weight="bold")
ax.text(50, 65, "Track 1",
        ha="center", va="center", fontsize=28, color=ACCENT, weight="bold")

ax.text(50, 50, "20",
        ha="center", va="center", fontsize=72, color=FG,
        weight="bold", family="monospace")
ax.text(50, 38, "Triton GPU operators",
        ha="center", va="center", fontsize=12, color=MUTED)

tiers = [("EASY", ACCENT2), ("MED", ACCENT3), ("HARD", ACCENT4)]
BW, BH, gap = 18, 9, 2
total_w = 3 * BW + 2 * gap
start = (100 - total_w) / 2
for i, (label, color) in enumerate(tiers):
    x0 = start + i * (BW + gap)
    rect = patches.FancyBboxPatch(
        (x0, 16), BW, BH, boxstyle="round,pad=0.3",
        linewidth=1.4, edgecolor=color, facecolor=PANEL,
    )
    ax.add_patch(rect)
    ax.text(x0 + BW / 2, 16 + BH / 2, label,
            ha="center", va="center", fontsize=11, color=color, weight="bold")

plt.savefig(OUT, dpi=DPI, facecolor=BG)
plt.close(fig)
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB, 480x480)")
