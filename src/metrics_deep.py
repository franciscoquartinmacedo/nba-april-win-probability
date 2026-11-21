"""Deep comparison of scoring rules on all out-of-sample predictions (Dec-Feb pseudo-test + March).
Uses the frozen models. No model is changed here."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import data as D
import final_eval as FE
import metrics as M

MODELS = ["constant", "log5", "log5_recalibrated", "ridge_rating", "ridge_b2b", "ridge_all", "elo_rating"]


def load_predictions():
    dev = D.load_development(); inputs = FE.compute_inputs(dev)
    parts = []
    for s, e in [("2025-12-01", "2026-01-01"), ("2026-01-01", "2026-02-01"), ("2026-02-01", "2026-03-01")]:
        te, P = FE.predict_period(dev, inputs, pd.Timestamp(s), pd.Timestamp(e))
        d = dev.loc[te, ["game_id", "game_date", "home", "away", "home_win", "margin"]].copy()
        for m in MODELS: d[m] = P[m]
        parts.append(d)
    mar = pd.read_csv("../outputs/march_predictions.csv", dtype={"game_id": str}, parse_dates=["game_date"])
    ctx = D.load_final_test_context().set_index("game_id")
    mar["margin"] = mar.game_id.map(ctx["margin"])
    out = pd.concat(parts + [mar[parts[0].columns]], ignore_index=True)
    out["period"] = np.where(out.game_date < pd.Timestamp("2026-03-01"), "Dec-Feb", "March")
    return out


def brier_decomp(y, p, bins=8):
    q = pd.qcut(p, bins, duplicates="drop"); g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    n = g.size().to_numpy(); pk, yk = g.p.mean().to_numpy(), g.y.mean().to_numpy(); N = n.sum(); yb = y.mean()
    rel = (n * (pk - yk) ** 2).sum() / N; res = (n * (yk - yb) ** 2).sum() / N; unc = yb * (1 - yb)
    return rel, res, unc


def calib_slope_intercept(y, p):
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    if np.ptp(x) < 1e-9: return np.nan, np.nan
    lr = LogisticRegression(C=np.inf, solver="lbfgs").fit(x, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


if __name__ == "__main__":
    df = load_predictions(); y = df.home_win.to_numpy(); rng = np.random.default_rng(0)
    print(f"out-of-sample games: {len(df)} (Dec-Feb {int((df.period=='Dec-Feb').sum())}, March {int((df.period=='March').sum())}); home-win rate {y.mean():.3f}")
    df.to_csv("../outputs/oos_predictions_all.csv", index=False)

    print("\n[1] LOG LOSS vs BRIER, with Brier decomposition (8 equal-count bins):  Brier = REL - RES + UNC")
    rows = []
    for m in MODELS:
        p = df[m].to_numpy(); rel, res, unc = brier_decomp(y, p)
        rows.append({"model": m, "log_loss": M.log_loss(y, p), "brier": M.brier(y, p), "REL(miscal.)": rel, "RES(resolution)": res, "UNC": unc})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n[2] CALIBRATION & DISCRIMINATION  (slope 1 / intercept 0 = perfectly calibrated; 95% CI by bootstrap)")
    rows = []
    for m in MODELS:
        p = df[m].to_numpy()
        if m == "constant":
            rows.append({"model": m, "AUC": 0.5, "slope": np.nan, "intercept": np.nan, "mean_pred": p.mean(), "actual": y.mean()}); continue
        s, i = calib_slope_intercept(y, p); bs = []
        for _ in range(500):
            ix = rng.integers(0, len(y), len(y)); bs.append(calib_slope_intercept(y[ix], p[ix]))
        bs = np.array(bs)
        rows.append({"model": m, "AUC": roc_auc_score(y, p), "slope": s, "slope_CI": f"[{np.nanpercentile(bs[:,0],2.5):.2f},{np.nanpercentile(bs[:,0],97.5):.2f}]",
                     "intercept": i, "mean_pred": p.mean(), "actual": y.mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n[3] THE ACCURACY PUZZLE: share of games picked as home wins, accuracy, and accuracy if threshold is moved to the training base rate")
    for per, sub in [("Dec-Feb", df[df.period == "Dec-Feb"]), ("March", df[df.period == "March"])]:
        ys = sub.home_win.to_numpy(); print(f"  -- {per}: always-home accuracy = {ys.mean():.3f}")
        for m in ["log5", "ridge_rating", "elo_rating"]:
            p = sub[m].to_numpy(); thr = 0.5
            print(f"     {m:13s} picks home {np.mean(p>=thr):.3f} of games | accuracy {np.mean((p>=thr)==(ys==1)):.3f} | AUC {roc_auc_score(ys,p):.3f} | mean p {p.mean():.3f}")
    mar = df[df.period == "March"]; ym = mar.home_win.to_numpy()
    a, b = mar.log5.to_numpy(), mar.ridge_rating.to_numpy(); dis = (a >= .5) != (b >= .5)
    print(f"  March: games where log5 and ridge pick DIFFERENT sides: {int(dis.sum())}; log5 right in {int(((a>=.5)==(ym==1))[dis].sum())}, ridge right in {int(((b>=.5)==(ym==1))[dis].sum())}")
    print(f"        mean |p-0.5| in those games: log5 {np.abs(a-.5)[dis].mean():.3f}, ridge {np.abs(b-.5)[dis].mean():.3f}; home won in {int(ym[dis].sum())} of {int(dis.sum())}")
    near = np.abs(b - .5) < 0.05
    print(f"  ridge predictions within 0.45-0.55 in March: {int(near.sum())} games ({near.mean():.0%})")
    ts = np.linspace(0.40, 0.60, 5)
    print("  March accuracy vs decision threshold (ridge / log5):", {round(float(t), 2): (round(float(np.mean((b >= t) == (ym == 1))), 3), round(float(np.mean((a >= t) == (ym == 1))), 3)) for t in ts})

    print("\n[4] SHARPNESS (spread of predictions) alongside calibration")
    for m in ["log5", "ridge_rating", "elo_rating", "ridge_all"]:
        p = df[m].to_numpy(); print(f"  {m:13s} sd {p.std():.3f} | share >0.70 or <0.30: {np.mean((p>.7)|(p<.3)):.3f} | share >0.80 or <0.20: {np.mean((p>.8)|(p<.2)):.3f} | min {p.min():.2f} max {p.max():.2f}")

    print("\n[5] TAILS: how much of total log loss comes from the worst games?  (log loss vs Brier sensitivity)")
    for m in ["log5", "ridge_rating", "ridge_all"]:
        p = df[m].to_numpy(); l = -(y * np.log(p) + (1 - y) * np.log(1 - p)); b_ = (p - y) ** 2
        k = int(0.05 * len(y)); idx = np.argsort(-l)[:k]
        print(f"  {m:13s} worst 5% of games = {l[idx].sum()/l.sum():.1%} of total log loss, {b_[idx].sum()/b_.sum():.1%} of total Brier | max single-game loss {l.max():.2f} (p={p[np.argmax(l)]:.2f}, home {'won' if y[np.argmax(l)] else 'lost'})")
    p = df["ridge_rating"].to_numpy(); l = -(y * np.log(p) + (1 - y) * np.log(1 - p)); w = np.argsort(-l)[:6]
    print("  6 worst games for ridge_rating (confident misses):")
    print(df.loc[w, ["game_date", "home", "away", "margin", "home_win", "ridge_rating", "log5"]].assign(loss=l[w]).round(3).to_string(index=False))

    print("\n[6] PER-PERIOD CONSISTENCY (does the ranking of models depend on the period / metric?)")
    for per in ["Dec-Feb", "March"]:
        sub = df[df.period == per]; ys = sub.home_win.to_numpy()
        print(f"  {per}: " + " | ".join(f"{m}: LL {M.log_loss(ys, sub[m].to_numpy()):.3f} B {M.brier(ys, sub[m].to_numpy()):.3f} Acc {M.accuracy(ys, sub[m].to_numpy()):.3f}" for m in ["log5", "ridge_rating", "elo_rating"]))

    print("\n[7] PRICING IMPLICATION (illustration, no market odds available): March home-win rate vs model mean probability")
    for m in ["log5", "ridge_rating", "elo_rating"]:
        p = mar[m].to_numpy(); print(f"  {m:13s} mean p(home) {p.mean():.3f} -> fair decimal odds {1/p.mean():.2f}; realised home-win rate {ym.mean():.3f} -> fair odds {1/ym.mean():.2f}")
    print("  paired metric differences, ridge vs log5, all 835 games (positive = ridge better): ", end="")
    ll_ = lambda p_: -(y * np.log(p_) + (1 - y) * np.log(1 - p_)); dd = ll_(df.log5.to_numpy()) - ll_(df.ridge_rating.to_numpy())
    bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
    db = (df.log5.to_numpy() - y) ** 2 - (df.ridge_rating.to_numpy() - y) ** 2; bb = [db[rng.integers(0, len(db), len(db))].mean() for _ in range(4000)]
    print(f"log loss {dd.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] | Brier {db.mean():+.4f} [{np.percentile(bb,2.5):+.4f},{np.percentile(bb,97.5):+.4f}]")
