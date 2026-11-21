"""Calibration in the large: mean predicted home-win probability against the actual home-win rate, with the calibration slope and the
log-loss change from shifting the logit by a constant (slope held at 1; descriptive, fitted on the same games).
Writes ../outputs/calibration_level.csv. Frozen models only; April = version A."""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from metrics_pooled import load_pooled, slope

d = load_pooled(); ll = lambda y, p: float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
sets = {"Oct-Feb (development)": None, "Dec-Feb": ["2025-12", "2026-01", "2026-02"], "March": ["2026-03"], "April": ["2026-04"], "March + April": ["2026-03", "2026-04"]}
rows = []
for name, months in sets.items():
    if months is None: continue
    c = d[d.month.isin(months)]; y = c.home_win.to_numpy(float)
    for m in ("constant", "log5", "ridge_rating", "elo_rating"):
        p = np.clip(c[m].to_numpy(float), 1e-6, 1 - 1e-6)
        s = slope(y, p) if m != "constant" else np.nan
        r = minimize_scalar(lambda b: ll(y, expit(logit(p) + b)), bounds=(-1.5, 1.5), method="bounded")
        rows.append({"period": name, "model": m, "n": len(y), "mean_pred": p.mean(), "actual": y.mean(), "slope": s,
                     "logit_shift": r.x, "log_loss": ll(y, p), "log_loss_after_shift": r.fun, "gain_from_shift": ll(y, p) - r.fun})
res = pd.DataFrame(rows); res.to_csv("../outputs/calibration_level.csv", index=False); print(res.round(3).to_string(index=False))
