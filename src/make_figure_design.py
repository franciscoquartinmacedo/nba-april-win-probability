"""Design schematic: what was fitted on, what was scored, and what was locked. Month widths are equal; game counts are printed.
Colours: grey = games the model is fitted on; blue steps = games that are scored (lighter = rolling validation, mid = final test, dark = blind test).
No model results are shown here."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TRAIN, VAL, FINAL, BLIND = "#d5d4cc", "#8db8ee", "#2a78d6", "#174f95"
months = [("Oct", 80), ("Nov", 219), ("Dec", 197), ("Jan", 233), ("Feb", 166), ("Mar", 239), ("Apr", 96)]
x0 = {m: i for i, (m, _) in enumerate(months)}

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
fig, ax = plt.subplots(figsize=(13.8, 7.0), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE); ax.set_xlim(-2.65, 9.5); ax.set_ylim(-3.1, 8.3); ax.axis("off")

# month header
for i, (m, n) in enumerate(months):
    ax.text(i + 0.5, 6.55, m, ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(i + 0.5, 6.42, f"{n} games", ha="center", va="top", fontsize=8.3, color=MUTED)
    ax.plot([i, i], [-0.35, 6.3], color=GRID, lw=0.8, zorder=0)
ax.plot([7, 7], [-0.35, 6.3], color=GRID, lw=0.8, zorder=0)

rows = [  # label, train end (exclusive month index), test month index, colour, note
    ("Validation fold 1", 2, 2, VAL, "Dec"),
    ("Validation fold 2", 3, 3, VAL, "Jan"),
    ("Validation fold 3", 4, 4, VAL, "Feb"),
    ("Final test", 5, 5, FINAL, "Mar"),
    ("Blind test, version A", 6, 6, BLIND, "Apr: headline"),
    ("Blind test, version B", 6, 6, BLIND, "Apr: sensitivity"),
]
H = 0.62
ys = [5.35, 4.45, 3.55, 2.45, 1.35, 0.45]
for (lab, tr, te, col, note), y in zip(rows, ys):
    ax.text(-2.55, y + H / 2, lab, ha="left", va="center", fontsize=9.8, color=INK, fontweight="bold" if "Blind" in lab or "Final" in lab else "normal")
    ax.add_patch(FancyBboxPatch((0.02, y), tr - 0.04, H, boxstyle="round,pad=0,rounding_size=0.08", fc=TRAIN, ec="none"))
    ax.text(tr / 2, y + H / 2, (f"fitted on all earlier games (Oct to {months[tr-1][0]})" if tr>2 else "fitted on Oct to Nov"), ha="center", va="center", fontsize=8.6, color=INK2)
    ax.add_patch(FancyBboxPatch((te + 0.02, y), 0.96, H, boxstyle="round,pad=0,rounding_size=0.08", fc=col, ec="none"))
    ax.text(te + 0.5, y + H / 2, note if "Apr" not in note else note.split(": ")[0], ha="center", va="center", fontsize=8.6, color="white" if col != VAL else INK, fontweight="bold")
# row captions to the right of the April column are short; extra detail goes in notes below
ax.text(7.1, 1.35 + H / 2, "everything outcome-based\nfrozen at 31 March", ha="left", va="center", fontsize=8.2, color=INK2)
ax.text(7.08, 0.45 + H / 2, "inputs roll forward\nthrough April", ha="left", va="center", fontsize=8.2, color=INK2)
ax.text(4.93, 3.31, "settings frozen before March is scored", ha="right", va="center", fontsize=8.4, color=INK2, style="italic")
ax.plot([5, 5], [2.3, 3.45], color=MUTED, lw=1.2, ls=(0, (4, 3)))

# lock bracket
ax.plot([5.02, 6.98], [-0.18, -0.18], color=INK2, lw=1.4)
ax.plot([5.02, 5.02], [-0.18, -0.06], color=INK2, lw=1.4); ax.plot([6.98, 6.98], [-0.18, -0.06], color=INK2, lw=1.4)
ax.text(7.0, -0.45, "March and April are locked: the data loaders refuse to return them until the final scripts run", ha="right", va="top", fontsize=8.8, color=INK2)

# rules, bottom left
ax.text(-2.55, -1.15, "No look-ahead:", ha="left", va="top", fontsize=9.2, color=INK, fontweight="bold")
ax.text(-2.55, -1.42, "a game's own result is never a feature of that game;\nprobability maps are fitted on training games only;\nperturbation test: changing later outcomes leaves\nversion A inputs unchanged.", ha="left", va="top", fontsize=8.6, color=INK2, linespacing=1.35)

ax.text(-2.55, 8.2, "How the models were tested: chronological, no random splits", ha="left", va="top", fontsize=12.5, color=INK, fontweight="bold")
ax.text(-2.55, 7.78, "Each block is fitted only on earlier games, then scores the next block. Grey = used to fit; blue = scored.", ha="left", va="top", fontsize=8.8, color=INK2)

# legend
lx = 0.15
for i, (c, t) in enumerate([(TRAIN, "fitted on"), (VAL, "validation (Dec-Feb)"), (FINAL, "final test (Mar)"), (BLIND, "blind test (Apr)")]):
    xx = [3.0, 4.0, 5.9, 7.3][i]
    ax.add_patch(FancyBboxPatch((xx, -2.9), 0.22, 0.2, boxstyle="round,pad=0,rounding_size=0.04", fc=c, ec="none"))
    ax.text(xx + 0.3, -2.8, t, ha="left", va="center", fontsize=8.8, color=INK2)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig("../outputs/design_timeline.png", facecolor=SURFACE)
print("saved")
