"""Regularised logistic regression with feature-block ablation (rest, form, box score). Dec/Jan/Feb rolling validation only."""
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import baselines as B
import data as D
import elo as E
import features as F
import metrics as M
import ridge as R

df = D.load_development()
assert df["game_date"].max() < D.FINAL_TEST_START
y_all = df["home_win"].to_numpy()
folds = list(D.rolling_folds(df))
val_idx = np.concatenate([va for _, _, va in folds]); y_val = y_all[val_idx]
fold_of_val = np.concatenate([[m] * len(va) for m, _, va in folds])

feat = F.build_features(df)
d_ridge = R.run_ridge(df, 80, "none", None)
d_elo, _, _ = E.run_elo(df, 0.01, None, "none", None, kscale=0.5)
BASES = {"ridge": d_ridge, "elo": d_elo}

# --- Q2 diagnostic: how different are the two ratings? ---
print("RIDGE vs ELO ratings: corr of pre-game expected margin on validation games:", round(float(np.corrcoef(d_ridge[val_idx], d_elo[val_idx])[0, 1]), 3),
      "| sd ridge", round(float(d_ridge[val_idx].std()), 2), "sd elo", round(float(d_elo[val_idx].std()), 2))

C_GRID = [0.1, 1.0, 10.0]


def design(base, blocks):
    cols = [base] + [feat[c].to_numpy() for b in blocks for c in F.FEATURE_BLOCKS[b]]
    return np.column_stack(cols)


def predict(X, C):
    p = np.empty(len(val_idx)); pos = 0
    for _, tr, va in folds:
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=C, solver="lbfgs", max_iter=2000).fit(sc.transform(X[tr]), y_all[tr])
        p[pos:pos + len(va)] = lr.predict_proba(sc.transform(X[va]))[:, 1]; pos += len(va)
    return p


ALL = ["rest", "form", "split", "box"]
specs = [("rating only", [])] + [(f"+{b}", [b]) for b in ALL] + [("+all", ALL)] + [(f"all minus {b}", [x for x in ALL if x != b]) for b in ALL]
rows, preds = [], {}
for bname, d in BASES.items():
    for sname, blocks in specs:
        X = design(d, blocks)
        for C in C_GRID:
            p = predict(X, C); key = (bname, sname, C); preds[key] = p
            rows.append({"base": bname, "spec": sname, "C": C, **M.summarise(y_val, p)})
res = pd.DataFrame(rows)
res.to_csv("../outputs/feature_ablation_results.csv", index=False)
print(f"\nVariants scored: {len(res)} (each spec x C in {C_GRID} x 2 bases)")

ll = lambda p: -(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
rng = np.random.default_rng(0)
def paired(pa, pb):  # positive => pb better
    dd = ll(pa) - ll(pb); bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
    return dd.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

pd.set_option("display.width", 200)
for bname in BASES:
    r = res[res.base == bname]
    print(f"\n=== BASE: {bname} rating | pooled Dec-Feb log loss by spec and C ===")
    print(r.pivot_table(index="spec", columns="C", values="log_loss").reindex([s for s, _ in specs]).round(4).to_string())
    base_key = (bname, "rating only", r[r.spec == "rating only"].sort_values("log_loss").iloc[0].C)
    print(f"-- paired gain vs rating-only (C={base_key[2]}), at each spec's own best C --")
    for sname, _ in specs[1:]:
        best_c = r[r.spec == sname].sort_values("log_loss").iloc[0].C
        m_, lo, hi = paired(preds[base_key], preds[(bname, sname, best_c)])
        print(f"   {sname:16s} C={best_c:<5} gain {m_:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

# Per-fold view: pooled results hide that the Dec fold trains on only Oct-Nov
print("\nPER-FOLD GAIN vs rating-only (C=1, ridge base); positive = feature block helps")
base_p = preds[("ridge", "rating only", 1.0)]
print(f"{'spec':16s} {'Dec':>8s} {'Jan':>8s} {'Feb':>8s} {'Jan+Feb':>9s}  paired Jan+Feb CI")
jf = fold_of_val != "2025-12"
for sname, _ in specs[1:]:
    p = preds[("ridge", sname, 1.0)]; g = [M.log_loss(y_val[fold_of_val == f], base_p[fold_of_val == f]) - M.log_loss(y_val[fold_of_val == f], p[fold_of_val == f]) for f in ["2025-12", "2026-01", "2026-02"]]
    dd = (ll(base_p) - ll(p))[jf]; bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(3000)]
    print(f"{sname:16s} {g[0]:+8.4f} {g[1]:+8.4f} {g[2]:+8.4f} {dd.mean():+9.4f}  [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")

# Reference baselines
p_const = np.concatenate([B.constant_home_rate(df, tr, va) for _, tr, va in folds])
p_log5 = np.concatenate([B.log5(df, tr, va, k=10) for _, tr, va in folds])
print("\nREFERENCE constant", round(M.log_loss(y_val, p_const), 4), "log5", round(M.log_loss(y_val, p_log5), 4))

# Coefficients of the full model, fit on all development data (Oct-Feb), standardised, with bootstrap CIs
print("\nSTANDARDISED COEFFICIENTS, full model on Oct-Feb (C=1), 95% bootstrap CI over games [ridge base]")
names = ["rating"] + [c for b in ALL for c in F.FEATURE_BLOCKS[b]]
X = design(d_ridge, ALL); idx = np.arange(len(df))
def fit_coef(i):
    sc = StandardScaler().fit(X[i]); return LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[i]), y_all[i]).coef_[0]
point = fit_coef(idx); boots = np.array([fit_coef(rng.integers(0, len(idx), len(idx))) for _ in range(300)])
for n_, b_, lo, hi in zip(names, point, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)):
    print(f"   {n_:10s} {b_:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {'<-- CI excludes 0' if lo > 0 or hi < 0 else ''}")
print("   correlation of each feature with the rating (validation games):",
      {n_: round(float(np.corrcoef(X[val_idx, i], X[val_idx, 0])[0, 1]), 2) for i, n_ in enumerate(names) if i})

# Leakage perturbation test for features: change game t's own box score and result; features for game <= t must not move
t = 500; df2 = df.copy(); df2.loc[t, ["home_points", "home_turnovers", "home_rebounds", "home_fouls"]] += [40, 15, 20, 10]; df2["margin"] = df2.home_points - df2.away_points
f1, f2 = F.build_features(df), F.build_features(df2)
print("\nLEAKAGE TEST features: unchanged through game t:", bool(np.allclose(f1.iloc[:t + 1], f2.iloc[:t + 1])), "| later rows changed:", bool(not np.allclose(f1.iloc[t + 1:], f2.iloc[t + 1:])))
