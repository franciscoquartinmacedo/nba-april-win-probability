"""How much can the best of several alternatives gain over ridge by luck alone, on the 96 April games?
Alternatives: ridge + back-to-back, ridge + all features, Elo, season-progress variant (post hoc).
Method: per-game log-loss gains of each alternative over ridge are centred to zero mean (no real difference), games are
resampled with replacement, and the best alternative's mean gain is recorded. Compared with the observed gain of
ridge + all features. Writes ../outputs/selection_simulation.csv. Version A is the headline; B is the sensitivity."""
import numpy as np, pandas as pd

ALTS = ["ridge_b2b", "ridge_all", "elo_rating", "ridge_progress"]
B = 20000
rows = []
for ver in ("A", "B"):
    a = pd.read_csv(f"../outputs/april_predictions_{ver}.csv"); y = a.home_win.to_numpy(float)
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    gains = {m: ll(a["ridge_rating"].to_numpy(float)) - ll(a[m].to_numpy(float)) for m in ALTS}   # positive = alternative better
    centred = np.column_stack([gains[m] - gains[m].mean() for m in ALTS])
    rng = np.random.default_rng(0); n = len(y)
    best = np.array([centred[rng.integers(0, n, n)].mean(axis=0).max() for _ in range(B)])
    obs = gains["ridge_all"].mean()
    rows.append({"version": ver, "n_games": n, "luck_best_mean": best.mean(), "luck_best_p95": np.percentile(best, 95),
                 "observed_ridge_all_gain": obs, "p_one_sided": float((best >= obs).mean())})
res = pd.DataFrame(rows); res.to_csv("../outputs/selection_simulation.csv", index=False); print(res.round(4).to_string(index=False))
