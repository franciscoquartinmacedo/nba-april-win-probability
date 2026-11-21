"""Figure 3: reliability diagram, ridge_rating, March (final test) + April (blind, version A), n=335. 5 equal-count bins (~67 games each).
Bars = 95% Wilson intervals on the observed home-win rate per bin. Run from src/."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics_pooled import load_pooled, slope

SURFACE, INK, INK2, MUTED, GRID, BASE, C_RIDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6"
d = load_pooled(); d = d[d.month.isin(["2026-03", "2026-04"])].reset_index(drop=True)
y, p = d.home_win.to_numpy(float), d.ridge_rating.to_numpy(float); N = len(d)
K = 5; order = np.argsort(p); bins = np.array_split(order, K)
def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z**2 / n; c = (ph + z**2 / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den; return c - h, c + h
rows = []
for ix in bins:
    lo, hi = wilson(y[ix].sum(), len(ix)); rows.append((p[ix].mean(), y[ix].mean(), lo, hi, len(ix), p[ix].min(), p[ix].max()))
R = pd.DataFrame(rows, columns=["pred", "obs", "lo", "hi", "n", "pmin", "pmax"]); print(R.round(3))
R.to_csv("../outputs/reliability_bins_mar_apr.csv", index=False)
sl = slope(y, p); rng = np.random.default_rng(0)
bs = [slope(y[i], p[i]) for i in (rng.integers(0, N, N) for _ in range(2000))]; slo, shi = np.percentile(bs, [2.5, 97.5]); print("slope", sl, slo, shi)

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
fig, (ax, hx) = plt.subplots(2, 1, figsize=(7.4, 7.6), dpi=150, facecolor=SURFACE, gridspec_kw={"height_ratios": [4.2, 1], "hspace": 0.12}, sharex=True)
for a in (ax, hx):
    a.set_facecolor(SURFACE); a.set_axisbelow(True)
    for s in ("top", "right", "left"): a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(BASE); a.tick_params(length=0)
ax.grid(color=GRID, lw=0.8)
ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=(0, (4, 3))); ax.text(0.80, 0.745, "perfectly calibrated", color=INK2, fontsize=8.8, rotation=36, ha="center", va="top")
ax.vlines(R.pred, R.lo, R.hi, color=C_RIDGE, lw=1.6, alpha=0.6)
ax.plot(R.pred, R.obs, color=C_RIDGE, lw=2, marker="o", ms=9, mfc=C_RIDGE, mec=SURFACE, mew=1.5, zorder=4)
for i, r in R.iterrows():
    if i == K - 1: ax.text(r.pred - 0.025, r.obs + 0.02, f"{r.obs:.2f}", fontsize=8.8, color=INK2, ha="right", va="bottom")
    else: ax.text(r.pred + 0.02, r.obs - 0.045, f"{r.obs:.2f}", fontsize=8.8, color=INK2, ha="left", va="top")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_ylabel("Observed home-win rate", fontsize=9.5)
ax.set_title("Too cautious: favourites win more, underdogs less, than predicted", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=24)
ax.text(0, 1.03, f"Ridge, March + April, {N} games; five equal-count groups of ~{N//K}; bars are 95% intervals", transform=ax.transAxes, fontsize=8.2, color=INK2)
ax.text(0.97, 0.06, f"calibration slope {sl:.2f}\n95% CI [{slo:.2f}, {shi:.2f}]\n(1 = calibrated)", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK2)
hx.hist(p, bins=np.linspace(0, 1, 21), color=BASE, edgecolor=SURFACE); hx.set_yticks([]); hx.set_xlabel("Predicted home-win probability", fontsize=9.5); hx.set_xticks(np.linspace(0, 1, 6))
hx.text(0.0, 1.0, "number of games", fontsize=8, color=MUTED, transform=hx.transAxes, va="top")
fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.09)
fig.savefig("../outputs/reliability_mar_apr.png", facecolor=SURFACE); print("saved")
