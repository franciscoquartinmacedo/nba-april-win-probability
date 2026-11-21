"""Pooled out-of-sample comparison, Dec-Apr (Dec-Feb rolling + March final test + April blind, version A).
Frozen models only (ridge_progress is post-hoc and April-only, so it is excluded from pooling).
Reads stored predictions; trains nothing; changes no model. Writes ../outputs/pooled_*.csv."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import data as D

MODELS = ["constant", "log5", "log5_recalibrated", "ridge_rating", "ridge_b2b", "ridge_all", "elo_rating"]
B = 10000


def load_pooled(april_version="A"):
    oos = pd.read_csv("../outputs/oos_predictions_all.csv", dtype={"game_id": str}, parse_dates=["game_date"])
    apr = pd.read_csv(f"../outputs/april_predictions_{april_version}.csv", dtype={"game_id": str}, parse_dates=["game_date"])
    margin = D.load_all_for_final_evaluation().set_index("game_id")["margin"]   # April locked loader; same access path as run_april.py
    apr["margin"] = apr.game_id.map(margin)
    cols = ["game_id", "game_date", "home", "away", "home_win", "margin"] + MODELS
    out = pd.concat([oos[cols + ["period"]], apr[cols].assign(period="April")], ignore_index=True)
    out["month"] = out.game_date.dt.to_period("M").astype(str)
    out["week"] = ((out.game_date - out.game_date.min()).dt.days // 7).astype(int)   # block id for the sensitivity bootstrap
    return out


def ll(y, p): return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_mean(d, rng, blocks=None):
    """95% CI of mean(d). blocks=None: naive game-level bootstrap. blocks=array: resample whole calendar weeks."""
    if blocks is None:
        ix = rng.integers(0, len(d), (B, len(d))); return np.percentile(d[ix].mean(1), [2.5, 97.5])
    ub = np.unique(blocks); sums = np.array([d[blocks == b].sum() for b in ub]); cnt = np.array([(blocks == b).sum() for b in ub])
    ix = rng.integers(0, len(ub), (B, len(ub))); return np.percentile(sums[ix].sum(1) / cnt[ix].sum(1), [2.5, 97.5])


def slope(y, p):
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return float(LogisticRegression(C=np.inf, solver="lbfgs").fit(x, y).coef_[0, 0])


if __name__ == "__main__":
    df = load_pooled(); rng = np.random.default_rng(0); y = df.home_win.to_numpy()
    periods = ["Dec-Feb", "March", "April"]
    print(f"pooled out-of-sample games: {len(df)} | " + ", ".join(f"{p} {int((df.period==p).sum())}" for p in periods) + f" | home-win rate {y.mean():.3f}")

    # [1] log loss by period and pooled
    rows = []
    for m in MODELS:
        r = {"model": m}
        for per in periods:
            s = df[df.period == per]; r[per] = ll(s.home_win.to_numpy(), s[m].to_numpy()).mean()
        r["pooled"] = ll(y, df[m].to_numpy()).mean(); rows.append(r)
    t1 = pd.DataFrame(rows); print("\n[1] LOG LOSS by period and pooled (frozen models)"); print(t1.round(4).to_string(index=False))
    t1.to_csv("../outputs/pooled_logloss.csv", index=False)

    # [2] paired gains with naive and week-block CIs, per period and pooled. positive = second model better
    pairs = [("log5", "ridge_rating"), ("log5_recalibrated", "ridge_rating"), ("ridge_rating", "ridge_b2b"), ("ridge_rating", "ridge_all"), ("ridge_rating", "elo_rating"), ("constant", "ridge_rating")]
    rows = []
    for a, b in pairs:
        for per in periods + ["pooled"]:
            s = df if per == "pooled" else df[df.period == per]; ys = s.home_win.to_numpy()
            d = ll(ys, s[a].to_numpy()) - ll(ys, s[b].to_numpy()); lo, hi = boot_mean(d, rng); blo, bhi = boot_mean(d, rng, s.week.to_numpy())
            rows.append({"from": a, "to": b, "period": per, "n": len(s), "gain": d.mean(), "naive_lo": lo, "naive_hi": hi, "weekblock_lo": blo, "weekblock_hi": bhi})
    t2 = pd.DataFrame(rows); t2.to_csv("../outputs/pooled_paired_gains.csv", index=False)
    print("\n[2] PAIRED LOG-LOSS GAINS (positive = 'to' model better); naive game-level CI and calendar-week block-bootstrap CI")
    for (a, b), g in t2.groupby(["from", "to"], sort=False):
        print(f"  {a} -> {b}")
        for _, r in g.iterrows(): print(f"     {r.period:8s} n={r.n:4d} {r.gain:+.4f}  naive [{r.naive_lo:+.4f},{r.naive_hi:+.4f}]  week-block [{r.weekblock_lo:+.4f},{r.weekblock_hi:+.4f}]")

    # [3] calibration slope by period and pooled, with naive bootstrap CI
    rows = []
    for m in ["log5", "ridge_rating", "elo_rating", "ridge_all"]:
        r = {"model": m}
        for per in periods + ["pooled"]:
            s = df if per == "pooled" else df[df.period == per]; ys = s.home_win.to_numpy(); p = s[m].to_numpy()
            bs = [slope(ys[i], p[i]) for i in (rng.integers(0, len(ys), len(ys)) for _ in range(300))]
            r[per] = f"{slope(ys, p):.2f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]"
        rows.append(r)
    t3 = pd.DataFrame(rows); print("\n[3] CALIBRATION SLOPE (1 = calibrated; >1 under-confident), naive 95% CI"); print(t3.to_string(index=False))
    t3.to_csv("../outputs/pooled_calibration_slopes.csv", index=False)

    # [4] by month
    rows = []
    for mo, s in df.groupby("month"):
        ys = s.home_win.to_numpy(); r = {"month": mo, "n": len(s), "home_rate": ys.mean()}
        for m in ["log5", "ridge_rating", "ridge_all"]: r[f"LL_{m}"] = ll(ys, s[m].to_numpy()).mean()
        r["AUC_ridge"] = roc_auc_score(ys, s.ridge_rating); r["acc_ridge"] = np.mean((s.ridge_rating >= .5) == (ys == 1)); r["acc_log5"] = np.mean((s.log5 >= .5) == (ys == 1))
        r["slope_ridge"] = slope(ys, s.ridge_rating.to_numpy()); rows.append(r)
    t4 = pd.DataFrame(rows); print("\n[4] BY MONTH"); print(t4.round(3).to_string(index=False)); t4.to_csv("../outputs/pooled_by_month.csv", index=False)
