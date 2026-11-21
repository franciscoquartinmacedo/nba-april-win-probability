"""Evaluation of the frozen candidate models on a held-out test window.
Usage:  python final_eval.py dry     -> Dec/Jan/Feb as pseudo-test months (must reproduce validation numbers)
        python final_eval.py march   -> March, scored once
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import baselines as B
import data as D
import elo as E
import features as F
import metrics as M
import ridge as R
from frozen_config import FROZEN

LAM = FROZEN["ridge_rating"]["lambda"]
ALL_COLS = [c for b in ("rest", "form", "split", "box") for c in F.FEATURE_BLOCKS[b]]


def freeze_records(df, freeze_after):
    """Version A support: returns a copy where (i) gp_min = games played by the less-experienced side (a schedule count,
    left unfrozen) is stored, and (ii) the season-to-date W/L record columns of every game on/after freeze_after are
    overwritten with each team's record as of the day before freeze_after."""
    out = df.copy()
    out["gp_min"] = np.minimum(out["home_wins"] + out["home_losses"], out["away_wins"] + out["away_losses"])
    pre = out[out["game_date"] < freeze_after]
    w, l = {}, {}
    for r in pre.itertuples():
        w[r.home] = w.get(r.home, 0) + r.home_win; l[r.home] = l.get(r.home, 0) + (1 - r.home_win)
        w[r.away] = w.get(r.away, 0) + (1 - r.home_win); l[r.away] = l.get(r.away, 0) + r.home_win
    post = out["game_date"] >= freeze_after
    for side in ("home", "away"):
        out.loc[post, f"{side}_wins"] = out.loc[post, side].map(w).fillna(0).to_numpy()
        out.loc[post, f"{side}_losses"] = out.loc[post, side].map(l).fillna(0).to_numpy()
    return out


def compute_inputs(df, freeze_after=None):
    """Pre-game quantities for every row (causal: row i depends only on earlier dates).
    freeze_after=None -> rolling inputs (version B); a date -> everything outcome-based frozen from that date (version A)."""
    fi = None if freeze_after is None else int((df["game_date"] < freeze_after).sum())
    d_ridge = R.run_ridge(df, LAM, "none", None, freeze_idx=fi)
    e = FROZEN["elo_robustness"]
    d_elo, _, _ = E.run_elo(df, e["K_inf"], None, "none", None, kscale=e["kscale"], freeze_idx=fi)
    return d_ridge, d_elo, F.build_features(df, freeze_after=freeze_after)


def predict_period(df, inputs, test_start, test_end):
    d_ridge, d_elo, feat = inputs
    y = df["home_win"].to_numpy()
    tr = df.index[df["game_date"] < test_start].to_numpy()
    te = df.index[(df["game_date"] >= test_start) & (df["game_date"] < test_end)].to_numpy()
    P = {}
    P["constant"] = B.constant_home_rate(df, tr, te)
    P["log5"] = B.log5(df, tr, te, k=FROZEN["log5"]["k_shrink"])
    x_tr, x_te = B._log5_offset(df.loc[tr], 10).reshape(-1, 1), B._log5_offset(df.loc[te], 10).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf, solver="lbfgs").fit(x_tr, y[tr])
    P["log5_recalibrated"] = lr.predict_proba(x_te)[:, 1]
    a, b = E.fit_map(d_ridge[tr], y[tr]); P["ridge_rating"] = E.predict_map(d_ridge[te], a, b)
    a, b = E.fit_map(d_elo[tr], y[tr]); P["elo_rating"] = E.predict_map(d_elo[te], a, b)

    def logit_model(cols, C):
        X = np.column_stack([d_ridge] + [feat[c].to_numpy(float) for c in cols])
        sc = StandardScaler().fit(X[tr])
        return LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    if "gp_min" in df.columns:   # post-March variant: probability slope depends on season progress (fitted on training games only)
        # k=10: first look (pre-dates the Oct-Feb estimate). k=12.7: estimated on Oct-Feb only, adopted AFTER April k=10 results were seen -> sensitivity, not pre-registered
        for name, k in (("ridge_progress", 10.0), ("ridge_progress_k127", 12.7)):
            u = E.early_weight(df["gp_min"].to_numpy(float), k)
            co, b = E.fit_map_gp(d_ridge[tr], u[tr], y[tr]); P[name] = E.predict_map_gp(d_ridge[te], u[te], co, b)
    P["ridge_b2b"] = logit_model(["home_b2b", "away_b2b"], FROZEN["logistic_C"]["ridge+b2b"])
    P["ridge_all"] = logit_model(ALL_COLS, FROZEN["logistic_C"]["ridge+all"])
    return te, P


def summarise_models(df, te_all, P_all, label, models=None):
    models = models or FROZEN["models"]
    y = df.loc[te_all, "home_win"].to_numpy(); rows = []
    for m in models:
        p = P_all[m]; s = M.summarise(y, p); lo, hi = M.bootstrap_ci(y, p, M.log_loss)
        rows.append({"model": m, "log_loss": s["log_loss"], "ll_lo": lo, "ll_hi": hi, "brier": s["brier"], "accuracy": s["accuracy"],
                     "auc": roc_auc_score(y, p), "mean_pred": p.mean(), "actual": y.mean(), "n": s["n"]})
    print(f"\n=== {label}: n={len(y)}, actual home-win rate {y.mean():.3f} ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p)); rng = np.random.default_rng(0)
    def paired(a, b):  # positive => b better than a
        dd = ll(P_all[a]) - ll(P_all[b]); bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(5000)]
        return f"{dd.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
    print("paired log-loss gains (95% CI; positive = second model better):")
    for a, b in [("constant", "log5"), ("log5", "ridge_rating"), ("log5_recalibrated", "ridge_rating"), ("constant", "ridge_rating"),
                 ("ridge_rating", "ridge_b2b"), ("ridge_rating", "ridge_all"), ("ridge_rating", "elo_rating")] + ([("ridge_rating", "ridge_progress"), ("log5", "ridge_progress")] + ([("ridge_progress", "ridge_progress_k127")] if "ridge_progress_k127" in models else []) if "ridge_progress" in models else []):
        print(f"  {a:18s} -> {b:14s} {paired(a, b)}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mode = sys.argv[1]
    src_hash = hashlib.sha256(b"".join(Path(f).read_bytes() for f in sorted(Path(__file__).parent.glob("*.py")) if f.name != "final_eval.py")).hexdigest()[:12]
    print("code hash (all src except this script):", src_hash, "| frozen config:", FROZEN["models"])

    if mode == "dry":
        dev = D.load_development()
        inputs = compute_inputs(dev)
        # causality check: inputs on dev must equal inputs computed with March appended, for every dev row
        ctx_marker = "skipped (March not loaded in dry mode)"
        tes, Ps = [], {m: [] for m in FROZEN["models"]}
        for start, end in [("2025-12-01", "2026-01-01"), ("2026-01-01", "2026-02-01"), ("2026-02-01", "2026-03-01")]:
            te, P = predict_period(dev, inputs, pd.Timestamp(start), pd.Timestamp(end)); tes.append(te)
            for m in P: Ps[m].append(P[m])
        te_all = np.concatenate(tes); P_all = {m: np.concatenate(v) for m, v in Ps.items()}
        summarise_models(dev, te_all, P_all, "DRY RUN Dec-Feb (pseudo-test months)")
        print("\nEXPECTED from earlier validation: constant 0.6937, log5 0.6574, ridge_rating 0.6419, elo 0.6416")

    elif mode == "march":
        ctx = D.load_final_test_context()                          # Oct-Mar, no April
        dev_n = int((ctx["game_date"] < D.FINAL_TEST_START).sum())
        inputs = compute_inputs(ctx)
        # causality check: dev-row inputs must not change when March is appended
        dev_only = compute_inputs(D.load_development())
        ok = (np.allclose(inputs[0][:dev_n], dev_only[0]) and np.allclose(inputs[1][:dev_n], dev_only[1])
              and np.allclose(inputs[2].iloc[:dev_n].to_numpy(float), dev_only[2].to_numpy(float)))
        print("causality check (Oct-Feb inputs identical with/without March appended):", ok)
        assert ok, "inputs are not causal - STOP"
        te, P = predict_period(ctx, inputs, D.FINAL_TEST_START, D.APRIL_START)
        tab = summarise_models(ctx, te, P, "FINAL TEST: MARCH (scored once)")
        out = ctx.loc[te, ["game_id", "game_date", "home", "away", "home_win"]].copy()
        for m, p in P.items(): out[m] = p
        out.to_csv("../outputs/march_predictions.csv", index=False); tab.to_csv("../outputs/march_summary.csv", index=False)
        print("\nsaved outputs/march_predictions.csv and march_summary.csv")
