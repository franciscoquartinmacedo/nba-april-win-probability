"""Pooled Dec-Apr overview figure (frozen models; April = version A). Reads stored predictions and ../outputs/pooled_paired_gains.csv.
Colours: validated reference palette (blue #2a78d6 = ridge rating, orange #eb6834 = log5), text in ink tokens, recessive grid.
Panels: (A) calibration slope by month, (B) log-loss gain of ridge over log5, (C) log-loss gain of ridge_all over ridge."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics_pooled import load_pooled, slope

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_RIDGE, C_LOG5 = "#2a78d6", "#eb6834"

df = pd.read_csv("../outputs/pooled_paired_gains.csv"); pooled = load_pooled()

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.1), dpi=150, facecolor=SURFACE, gridspec_kw={"width_ratios": [1.35, 1, 1]})
for ax in axes:
    ax.set_facecolor(SURFACE); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE); ax.tick_params(length=0)

# ---------------- Panel A: calibration slope by month ----------------
ax = axes[0]; ax.grid(axis="y", color=GRID, lw=0.8)
months = sorted(pooled.month.unique()); rng = np.random.default_rng(0); S = {}
for name in ("ridge_rating", "log5"):
    pt, lo, hi = [], [], []
    for m in months:
        g = pooled[pooled.month == m]; y, p = g.home_win.to_numpy(), g[name].to_numpy(); pt.append(slope(y, p))
        b = [slope(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(400))]
        lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
    S[name] = (np.array(pt), np.array(lo), np.array(hi))
x = np.arange(len(months)); labels = [f"{pd.Period(m).strftime('%b')}\nn={int((pooled.month == m).sum())}" for m in months]
ax.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (4, 3))); ax.text(3.3, 0.9, "perfectly calibrated (1.0)", color=INK2, fontsize=8.5, va="top", ha="center")
for name, col, off in (("ridge_rating", C_RIDGE, -0.05), ("log5", C_LOG5, 0.05)):
    pt, lo, hi = S[name]
    ax.vlines(x + off, lo, hi, color=col, lw=1.2, alpha=0.55)
    ax.plot(x + off, pt, color=col, lw=2, marker="o", ms=7, mfc=col, mec=SURFACE, mew=1.5)
ax.text(x[-1] + 0.14, S["ridge_rating"][0][-1], f"ridge {S['ridge_rating'][0][-1]:.2f}", color=INK2, fontsize=9, va="top", ha="left")
ax.text(x[-1] + 0.14, S["log5"][0][-1], f"log5 {S['log5'][0][-1]:.2f}", color=INK2, fontsize=9, va="bottom", ha="left")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_xlim(-0.5, len(months) - 0.1); ax.set_ylim(0, 3.7)
ax.set_ylabel("Calibration slope (above 1 = too close to 50%)", fontsize=9)
ax.set_title("Over-confident early, under-confident late", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=24)
ax.text(0, 1.03, "Slope of outcomes on predicted log-odds, by month; bars are 95% bootstrap intervals", transform=ax.transAxes, fontsize=8.2, color=INK2)

# ---------------- Panels B, C: paired log-loss gain by period (forest plots) ----------------
rows = [("Dec-Feb", "Dec–Feb"), ("March", "March"), ("April", "April"), ("pooled", "Pooled")]
ypos = {"Dec-Feb": 3.6, "March": 2.6, "April": 1.6, "pooled": 0.0}


def forest(ax, a, b, title, sub, xlabel):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.axvline(0, color=BASE, lw=1.4)
    for key, lab in rows:
        r = df[(df["from"] == a) & (df["to"] == b) & (df.period == key)].iloc[0]; y = ypos[key]
        big = key == "pooled"
        ax.hlines(y, r.naive_lo, r.naive_hi, color=INK2, lw=2.2 if big else 1.6, alpha=1 if big else 0.55)
        ax.plot([r.gain], [y], marker="o", ms=9.5 if big else 8, mfc=INK2, mec=SURFACE, mew=1.6, ls="none")
        if big:
            ax.hlines(y - 0.42, r.weekblock_lo, r.weekblock_hi, color=MUTED, lw=1.6)
            ax.text(r.weekblock_lo, y - 0.58, "week-block CI", color=MUTED, fontsize=7.8, va="top", ha="left")
            ax.text(r.gain, y + 0.32, f"{r.gain:+.3f}".replace("-", "\u2212"), color=INK, fontsize=9.5, fontweight="bold", va="bottom", ha="center")
    ax.set_yticks([ypos[k] for k, _ in rows]); ax.set_yticklabels([f"{lab}\nn={int(df[(df['from']==a)&(df['to']==b)&(df.period==k)].n.iloc[0])}" for k, lab in rows], fontsize=9)
    ax.tick_params(axis="y", pad=6); ax.set_ylim(-1.35, 4.25); ax.set_xlim(-0.07, 0.065)
    ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06]); ax.set_xticklabels(["−0.06", "−0.04", "−0.02", "0", "+0.02", "+0.04", "+0.06"], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.2, color=INK2)


forest(axes[1], "log5", "ridge_rating", "Ridge beats log5, but unevenly", "Paired log-loss gain by game; bars are 95% intervals", "Gain in log loss (right of 0 = ridge better)")
forest(axes[2], "ridge_rating", "ridge_all", "Extra features hurt, except in April", "Gain of ridge_all (rest, form, box score) over ridge", "Gain in log loss (left of 0 = ridge_all worse)")

h = [plt.Line2D([0], [0], color=C_RIDGE, lw=2, marker="o", ms=7, mec=SURFACE), plt.Line2D([0], [0], color=C_LOG5, lw=2, marker="o", ms=7, mec=SURFACE)]
fig.legend(h, ["Ridge rating (primary model)", "Log5 (win-percentage baseline)"], loc="lower center", ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.2, wspace=0.34)
fig.savefig("../outputs/pooled_overview.png", facecolor=SURFACE)
print("monthly ridge slopes", np.round(S["ridge_rating"][0], 2), "| log5", np.round(S["log5"][0], 2))
