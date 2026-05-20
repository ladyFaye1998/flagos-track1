"""Generate the GitHub social-preview banner.

Layout (1280 x 640, all coords in 0..100 canvas units):

  +----------------------------------------------+
  |  | FlagOS Track 1                            |
  |  | 20 Triton GPU Operators       [178]  [25] |
  |  | FlagOS Open Computing -- S1   tests kernel|
  |  |                              ------ ------|
  |  |                              [APACHE] [PT]|
  |  | [EASY]  [MEDIUM]  [HARD]      2.0   fall  |
  |  | github.com/.../flagos-track1              |
  |  | by Danielle Lesin                         |
  +----------------------------------------------+

Outer padding is enforced (xlim 0-100, content lives in x 4-96, y 8-92).
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
PANEL2 = "#1B222C"
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
ax.set_aspect("auto")
ax.axis("off")

# Subtle inset frame so the banner reads as a "card" at any preview size
frame = patches.FancyBboxPatch(
    (1.2, 1.2), 97.6, 97.6, boxstyle="round,pad=0",
    linewidth=1.5, edgecolor="#22303C", facecolor=BG,
)
ax.add_patch(frame)

# Left accent bar (kept inside the inset frame)
bar = patches.Rectangle((3, 8), 0.8, 84, facecolor=ACCENT, edgecolor="none")
ax.add_patch(bar)

# ----- Left hero column -----
LX = 7  # left margin for hero text

ax.text(LX, 82, "FlagOS Track 1", ha="left", va="center",
        fontsize=54, color=FG, weight="bold")
ax.text(LX, 71, "20 Triton GPU Operators", ha="left", va="center",
        fontsize=30, color=ACCENT, weight="bold")
ax.text(LX, 62, "FlagOS Open Computing Global Challenge - Season 1",
        ha="left", va="center", fontsize=14, color=FG)
ax.text(LX, 56, "Operator Development & Optimization",
        ha="left", va="center", fontsize=12, color=MUTED)

# Tier badges, evenly spaced row
tiers = [
    ("EASY",   "8 ops",  ACCENT2),
    ("MEDIUM", "8 ops",  ACCENT3),
    ("HARD",   "4 ops",  ACCENT4),
]
BADGE_W, BADGE_H = 13, 11
BADGE_Y = 32
for i, (label, count, color) in enumerate(tiers):
    x0 = LX + i * (BADGE_W + 3)
    rect = patches.FancyBboxPatch(
        (x0, BADGE_Y), BADGE_W, BADGE_H, boxstyle="round,pad=0.35",
        linewidth=2, edgecolor=color, facecolor=PANEL,
    )
    ax.add_patch(rect)
    ax.text(x0 + BADGE_W / 2, BADGE_Y + BADGE_H * 0.66, label,
            ha="center", va="center", fontsize=14, color=color, weight="bold")
    ax.text(x0 + BADGE_W / 2, BADGE_Y + BADGE_H * 0.30, count,
            ha="center", va="center", fontsize=11, color=FG)

# Footer: repo URL + author + upstream contribution
ax.text(LX, 22, "github.com/ladyFaye1998/flagos-track1",
        ha="left", va="center", fontsize=17, color=ACCENT,
        family="monospace", weight="bold")
ax.text(LX, 16, "upstream: 5 FlagGems perf PRs (#3400-#3404)",
        ha="left", va="center", fontsize=11, color=ACCENT2,
        family="monospace")
ax.text(LX, 11, "by Danielle Lesin  -  Apache-2.0",
        ha="left", va="center", fontsize=12, color=MUTED)

# ----- Right column: 2x2 stat card grid -----
STATS = [
    ("178", "tests pass",    ACCENT2),
    ("25",  "kernels",       ACCENT3),
    ("20",  "Triton ops",    ACCENT),
    ("3",   "tiers",         ACCENT4),
]
GX0 = 60                 # grid left edge
GY0 = 30                 # grid bottom edge
CARD_W, CARD_H = 16, 19  # individual card size
HGAP, VGAP = 3, 3        # gaps between cards
for idx, (big, small, color) in enumerate(STATS):
    col = idx % 2
    row = idx // 2
    x0 = GX0 + col * (CARD_W + HGAP)
    y0 = GY0 + (1 - row) * (CARD_H + VGAP)  # top row first
    rect = patches.FancyBboxPatch(
        (x0, y0), CARD_W, CARD_H, boxstyle="round,pad=0.4",
        linewidth=1.6, edgecolor=color, facecolor=PANEL2,
    )
    ax.add_patch(rect)
    ax.text(x0 + CARD_W / 2, y0 + CARD_H * 0.62, big,
            ha="center", va="center", fontsize=30, color=color,
            weight="bold", family="monospace")
    ax.text(x0 + CARD_W / 2, y0 + CARD_H * 0.25, small,
            ha="center", va="center", fontsize=12, color=FG)

# Right-side stack label above the grid (gives the right column a heading)
ax.text(GX0 + (2 * CARD_W + HGAP) / 2, 79, "by the numbers",
        ha="center", va="center", fontsize=14, color=MUTED, style="italic")

# Bottom-right tagline (PyTorch + Triton stack)
ax.text(GX0 + (2 * CARD_W + HGAP) / 2, 19,
        "PyTorch + Triton  /  CPU fallback",
        ha="center", va="center", fontsize=13, color=MUTED,
        family="monospace")
ax.text(GX0 + (2 * CARD_W + HGAP) / 2, 14,
        "178 / 178 green  /  Apache-2.0",
        ha="center", va="center", fontsize=11, color=ACCENT2,
        family="monospace")

# Save WITHOUT bbox_inches=tight so the output is exactly 1280x640
plt.savefig(OUT, dpi=DPI, facecolor=BG)
plt.close(fig)

size_kb = OUT.stat().st_size / 1024
print(f"Wrote {OUT}  ({size_kb:.1f} KB, 1280x640)")
