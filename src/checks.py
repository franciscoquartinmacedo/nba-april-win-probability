"""Independent sanity checks. Run last; every check must pass (the script stops with an error otherwise).
 1. Lock: the March/April loaders refuse to run without NBA_FINAL_EVAL=1; the development loader never returns March.
 2. Causality: changing one game's result changes no earlier input (ridge inputs not even on the same day).
 3. April file: 96 unique games, valid probabilities, matches the stored version A forecasts.
 4. Permutation: ridge's March + April log loss is far below what shuffled outcomes give (the skill is not an accident of the scoring)."""
import os, subprocess, sys
import numpy as np, pandas as pd

# 1. lock
env = {k: v for k, v in os.environ.items() if k != "NBA_FINAL_EVAL"}
for fn in ("load_final_test_context", "load_final_test_march", "load_all_for_final_evaluation"):
    r = subprocess.run([sys.executable, "-c", f"import data; data.{fn}()"], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "locked" in r.stderr, f"{fn} did not refuse without the flag"
os.environ["NBA_FINAL_EVAL"] = "1"
import data as D, final_eval as FE
assert D.load_development()["game_date"].max() < D.FINAL_TEST_START, "development data reaches March"
print("1. lock: OK (three locked loaders refuse without the flag; development data ends before March)")

# 2. causality: swap the points in one mid-March game, recompute all inputs
ctx = D.load_final_test_context(); k = int(ctx.index[ctx["game_date"] >= pd.Timestamp("2026-03-15")][0]); d0 = ctx.loc[k, "game_date"]
mod = ctx.copy(); mod.loc[k, ["home_points", "away_points"]] = ctx.loc[k, ["away_points", "home_points"]].to_numpy()
mod["margin"] = mod["home_points"] - mod["away_points"]; mod["home_win"] = (mod["margin"] > 0).astype(int)
for label, fz in (("rolling (version B)", None), ("frozen at 1 March (version A path)", D.FINAL_TEST_START)):
    base = FE.compute_inputs(FE.freeze_records(ctx, fz) if fz is not None else ctx, fz)
    pert = FE.compute_inputs(FE.freeze_records(mod, fz) if fz is not None else mod, fz)
    before = (ctx["game_date"] < d0).to_numpy(); same_day = (ctx["game_date"] <= d0).to_numpy()
    assert np.allclose(base[0][same_day], pert[0][same_day]), f"ridge input moved before/at the changed game ({label})"
    assert np.allclose(base[1][before], pert[1][before]), f"Elo input moved before the changed game ({label})"
    assert np.allclose(base[2][before].to_numpy(float), pert[2][before].to_numpy(float)), f"feature input moved ({label})"
    after = (ctx["game_date"] > d0).to_numpy()
    if fz is None: assert not np.allclose(base[0][after], pert[0][after]), "later ridge inputs did not react: the test is blind"
print(f"2. causality: OK (changing game on {d0.date()} moves no earlier ridge, Elo or feature input, nor the ridge inputs of that day; rolling inputs do react afterwards)")

# 3. April file
p = pd.read_csv("../outputs/april_predictions.csv", dtype={"game_id": str}); A = pd.read_csv("../outputs/april_predictions_A.csv", dtype={"game_id": str})
assert len(p) == 96 and p.game_id.is_unique and p.isna().sum().sum() == 0
assert p.p_home.between(0.01, 0.99).all() and set(p.columns) == {"game_id", "game_date", "away", "home", "p_home"}
assert np.allclose(p.set_index("game_id").p_home, A.set_index("game_id").loc[p.game_id, "ridge_rating"].round(3), atol=5e-4)
full = D.load_all_for_final_evaluation(); apr = full[full.game_date >= D.APRIL_START]
assert set(p.game_id) == set(apr.game_id), "April file does not cover exactly the April games"
print(f"3. April file: OK (96 unique games, {p.game_date.min()} to {p.game_date.max()}, probabilities {p.p_home.min():.3f} to {p.p_home.max():.3f})")

# 4. permutation
from metrics_pooled import load_pooled
c = load_pooled(); c = c[c.month.isin(["2026-03", "2026-04"])]; y = c.home_win.to_numpy(float); pr = c.ridge_rating.to_numpy(float)
ll = lambda yy: float(-np.mean(yy * np.log(pr) + (1 - yy) * np.log(1 - pr)))
rng = np.random.default_rng(0); null = np.array([ll(rng.permutation(y)) for _ in range(5000)])
assert ll(y) < np.percentile(null, 0.1)
print(f"4. permutation: OK (actual log loss {ll(y):.3f}; shuffled outcomes give {null.mean():.3f}, best of 5000 shuffles {null.min():.3f})")
print("\nAll checks passed.")
