"""Figure 2: headline results. (A) log loss by model and period (frozen models, April = version A); (B) ridge's paired advantage over each alternative.
Reads ../outputs/pooled_logloss.csv and pooled_paired_gains.csv. Lower log loss is better. Run from src/."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = {"ridge_rating": "#2a78d6", "log5": "#eb6834", "elo_rating": "#1baf7a", "constant": MUTED}
NAME = {"ridge_rating": "ridge", "log5": "log5", "elo_rating": "Elo", "constant": "constant home rate"}
ll = pd.read_csv("../outputs/pooled_logloss.csv").set_index("model"); g = pd.read_csv("../outputs/pooled_paired_gains.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
fig, (a, b) = plt.subplots(1, 2, figsize=(14.2, 5.6), dpi=150, facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})
for ax in (a, b):
    ax.set_facecolor(SURFACE); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE); ax.tick_params(length=0)

# ---- A: log loss by period
periods = ["Dec-Feb", "March", "April", "pooled"]; xs = np.array([0, 1, 2, 3.4]); n = [596, 239, 96, 931]
a.grid(axis="y", color=GRID, lw=0.8)
for m in ("constant", "log5", "elo_rating", "ridge_rating"):
    y = ll.loc[m, periods].to_numpy(float); big = m == "ridge_rating"
    a.plot(xs[:3], y[:3], color=C[m], lw=2.2 if big else 1.6, alpha=1 if big else 0.9, zorder=3)
    px = xs.copy(); px[-1] += {"log5": -0.16, "ridge_rating": 0.0, "elo_rating": 0.16}.get(m, 0)
    a.plot(px, y, ls="none", marker="o", ms=9 if big else 7, mfc=C[m], mec=SURFACE, mew=1.5, zorder=4)
    ly = {"log5": 0.625, "ridge_rating": 0.597, "elo_rating": 0.569}.get(m, y[-1])
    if m in ("log5", "ridge_rating", "elo_rating"): a.plot([px[-1] + 0.05, xs[-1] + 0.45], [y[-1], ly], color=BASE, lw=0.9, zorder=1)
    a.text(xs[-1] + 0.5, ly, f"{NAME[m]} {y[-1]:.3f}", color=INK2, fontsize=9.2, va="center", ha="left")
a.axvline(2.7, color=BASE, lw=1, ls=(0, (3, 3)))
a.set_xticks(xs); a.set_xticklabels([f"{p.replace('Dec-Feb','Dec–Feb').replace('pooled','Pooled')}\nn={k}" for p, k in zip(periods, n)], fontsize=9.2)
a.set_xlim(-0.35, 5.0); a.set_ylim(0.44, 0.72); a.set_ylabel("Log loss (lower is better)", fontsize=9.5)
a.set_title("Team-strength models clearly beat the constant home rate; the gap to log5 is small", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=24)
a.text(0, 1.03, "Out-of-sample log loss; Dec–Feb = three validation folds, March = final test, April = blind test (version A)", transform=a.transAxes, fontsize=8.2, color=INK2)

# ---- B: ridge advantage over each alternative (positive = ridge better)
alts = [("constant", "constant home rate"), ("log5", "log5"), ("log5_recalibrated", "log5, recalibrated"), ("ridge_b2b", "ridge + back-to-back"), ("ridge_all", "ridge + all features"), ("elo_rating", "Elo")]
def adv(alt, per):
    r = g[(g["from"] == alt) & (g["to"] == "ridge_rating") & (g.period == per)]
    if len(r): r = r.iloc[0]; return r.gain, r.naive_lo, r.naive_hi
    r = g[(g["from"] == "ridge_rating") & (g["to"] == alt) & (g.period == per)].iloc[0]; return -r.gain, -r.naive_hi, -r.naive_lo
b.grid(axis="x", color=GRID, lw=0.8); b.axvline(0, color=BASE, lw=1.4)
ys = np.arange(len(alts))[::-1] * 1.0
for (alt, lab), y in zip(alts, ys):
    for per, dy in (("Dec-Feb", .24), ("March", .12), ("April", 0)):
        pass
    pt, lo, hi = adv(alt, "pooled")
    b.hlines(y, lo, hi, color=INK2, lw=2.2); b.plot([pt], [y], marker="o", ms=9.5, mfc=INK2, mec=SURFACE, mew=1.6, ls="none", zorder=4)
    for per, mk in (("Dec-Feb", "s"), ("March", "^"), ("April", "D")):
        v = adv(alt, per)[0]; b.plot([v], [y - 0.3], marker=mk, ms=5.5, mfc="none", mec=MUTED, mew=1.2, ls="none")
    b.text(max(hi, adv(alt, "April")[0], adv(alt, "Dec-Feb")[0]) + 0.004 if alt != "constant" else hi + 0.004, y + 0.02, f"{pt:+.3f}".replace("-", "−"), fontsize=9, color=INK, fontweight="bold", va="center")
b.set_yticks(ys); b.set_yticklabels([l for _, l in alts], fontsize=9.5); b.tick_params(axis="y", pad=6)
b.set_xlim(-0.04, 0.14); b.set_ylim(-0.8, len(alts) - 0.3)
b.set_xlabel("Log loss of alternative minus log loss of ridge  (right of 0 = ridge better)", fontsize=9.2)
b.set_title("Only Elo edges ridge, and only by 0.003", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=24)
b.text(0, 1.03, "Pooled Dec–Apr, 931 games; bar = 95% bootstrap interval", transform=b.transAxes, fontsize=8.2, color=INK2)
hh = [plt.Line2D([0], [0], marker=m, ls="none", mfc="none", mec=MUTED, mew=1.2, ms=6) for m in "s^D"]
b.legend(hh, ["Dec–Feb", "March", "April"], title="Single periods", loc="lower right", frameon=False, fontsize=8.5, title_fontsize=8.5, labelcolor=INK2, ncol=3, bbox_to_anchor=(1.0, -0.27))
fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.2, wspace=0.33)
fig.savefig("../outputs/headline_results.png", facecolor=SURFACE); print("saved")
