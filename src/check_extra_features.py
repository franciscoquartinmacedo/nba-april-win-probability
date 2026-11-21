"""Further feature checks (rest categories, form windows, venue-specific home advantage, box score). Dec/Jan/Feb rolling validation only. Tests each block in margin space (powerful) and win/loss space."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

import data as D
import features as F
import features_extra as FX
import metrics as M
import ridge as R

df = D.load_development()
assert df["game_date"].max() < D.FINAL_TEST_START
y = df["home_win"].to_numpy(); mg = df["margin"].to_numpy().astype(float)
folds = list(D.rolling_folds(df))
vi = np.concatenate([va for _, _, va in folds]); fv = np.concatenate([[m] * len(va) for m, _, va in folds])
jf = fv != "2025-12"
feat, ex = F.build_features(df), FX.build_extra(df)
d = R.run_ridge(df, 80, "none", None)
rng = np.random.default_rng(0)

# ---------- leakage test for the new features ----------
t = 500; df2 = df.copy(); df2.loc[t, ["home_points"]] += 40
e1, e2 = FX.build_extra(df), FX.build_extra(df2)
print("LEAKAGE TEST (extra features): unchanged through game t:", bool(np.allclose(e1.iloc[:t + 1].astype(float), e2.iloc[:t + 1].astype(float))),
      "| later rows changed:", bool(not np.allclose(e1.iloc[t + 1:].astype(float), e2.iloc[t + 1:].astype(float))))


def X_of(cols):
    return np.column_stack([d] + [ (feat[c] if c in feat else ex[c]).to_numpy(dtype=float) for c in cols])


def run(cols, C=1.0):
    """returns val predictions: win prob (logistic) and margin (ridge OLS)"""
    X = X_of(cols); p = np.empty(len(vi)); mm = np.empty(len(vi)); pos = 0
    for _, tr, va in folds:
        sc = StandardScaler().fit(X[tr])
        p[pos:pos + len(va)] = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[va]))[:, 1]
        mm[pos:pos + len(va)] = Ridge(alpha=1.0).fit(sc.transform(X[tr]), mg[tr]).predict(sc.transform(X[va])); pos += len(va)
    return p, mm


def ll_game(p): return -(y[vi] * np.log(p) + (1 - y[vi]) * np.log(1 - p))
def ci(dd): bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(3000)]; return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

p0, m0 = run([]); L0 = ll_game(p0); S0 = (mg[vi] - m0) ** 2
print(f"\nBASELINE rating-only: pooled logloss {L0.mean():.4f} | margin MSE {S0.mean():.2f} (sd of margin {mg[vi].std():.2f}) | Jan+Feb logloss {L0[jf].mean():.4f} MSE {S0[jf].mean():.2f}")

def report(title, specs):
    print(f"\n=== {title} ===")
    print(f"{'spec':26s} {'dLL all':>8s} {'dLL J+F':>8s} {'CI J+F':>18s} | {'dMSE all':>8s} {'dMSE J+F':>8s} {'CI J+F':>16s}")
    for name, cols in specs:
        p, mm = run(cols); dl = L0 - ll_game(p); ds = S0 - (mg[vi] - mm) ** 2
        a, b = ci(dl[jf]); c_, e_ = ci(ds[jf])
        print(f"{name:26s} {dl.mean():+8.4f} {dl[jf].mean():+8.4f} [{a:+.4f},{b:+.4f}] | {ds.mean():+8.3f} {ds[jf].mean():+8.3f} [{c_:+.3f},{e_:+.3f}]")

# ---------- 1. rest: counts then linear vs non-linear ----------
print("\nREST COUNTS in Oct-Feb (days since own last game; first game = 3):")
for nm in ["h_rest", "a_rest"]:
    print(f"  {nm}:", ex.loc[:, nm].value_counts().sort_index().to_dict())
report("REST (positive = better than rating-only; dLL = log-loss gain, dMSE = margin MSE gain)", [
    ("linear rest_diff (old)", ["rest_diff"]),
    ("back-to-back flags only", ["home_b2b", "away_b2b"]),
    ("categories 1/2/>=3 (4 dums)", ["h_r1", "h_r2", "a_r1", "a_r2"]),
    ("log(rest) diff", ["rest_log_diff"]),
    ("categories + linear diff", ["h_r1", "h_r2", "a_r1", "a_r2", "rest_diff"]),
])

# ---------- 2. form windows ----------
report("FORM WINDOW (deviation of last-w mean margin from season mean, shrunk by n_w/(n_w+10))",
       [(f"form window {w}", [f"form{w}_diff"]) for w in FX.WINDOWS] + [("form, window 10", ["form_diff"])])

# ---------- 4. box-score, each stat and block ----------
report("BOX-SCORE (season-to-date own-minus-opponent differential, shrunk)", [
    ("turnovers only", ["tov_diff"]), ("rebounds only", ["reb_diff"]), ("fouls only", ["fou_diff"]),
    ("turnovers + rebounds", ["tov_diff", "reb_diff"]), ("all three", ["tov_diff", "reb_diff", "fou_diff"])])
report("VENUE SPLIT", [("venue split diff", ["split_diff"])])

# ---------- do box features proxy for an over-shrunk rating? ----------
print("\nPROXY TEST: does the box-score gain shrink when the rating is LESS shrunk (lam smaller)?")
for lam in [10, 20, 40, 80]:
    d_save = d; d = R.run_ridge(df, lam, "none", None)
    pa, ma = run([]); pb, mb = run(["tov_diff", "reb_diff"])
    print(f"  lam={lam:3d}: rating-only MSE {((mg[vi]-ma)**2).mean():.3f} | +tov,reb MSE {((mg[vi]-mb)**2).mean():.3f} | gain {((mg[vi]-ma)**2).mean()-((mg[vi]-mb)**2).mean():+.3f}  (logloss gain {ll_game(pa).mean()-ll_game(pb).mean():+.4f})")
    d = d_save

# ---------- 3. venue split noise: variance decomposition ----------
print("\nVENUE SPLIT NOISE: how much real between-team variation in home advantage is in Oct-Feb?")
parts = []
for tm in sorted(set(df.home)):
    hm = df[df.home == tm].margin; am = -df[df.away == tm].margin
    parts.append((tm, len(hm), len(am), hm.mean(), am.mean()))
H = pd.DataFrame(parts, columns=["team", "nh", "na", "hmean", "amean"]); H["hca"] = (H.hmean - H.amean) / 2
sig = df.margin.std()   # per-game sd of margin
H["samp_var"] = sig ** 2 * (1 / H.nh + 1 / H.na) / 4
obs_var, mean_noise = H.hca.var(ddof=1), H.samp_var.mean()
print(f"  observed sd of team HCA estimates {obs_var ** .5:.2f} pts | expected from pure sampling noise {mean_noise ** .5:.2f} pts | mean HCA {H.hca.mean():.2f}")
true_var = obs_var - mean_noise
print(f"  implied true between-team sd of HCA: {max(true_var, 0) ** .5:.2f} pts  (variance estimate {true_var:+.2f}; <=0 means indistinguishable from zero)")
half = df.game_date.median(); r1, r2 = [], []
for tm in H.team:
    a = df[df.game_date <= half]; b = df[df.game_date > half]
    f = lambda x: ((x[x.home == tm].margin.mean() + x[x.away == tm].margin.mean()) / 2)
    r1.append(f(a)); r2.append(f(b))
print(f"  split-half correlation of team HCA across teams (first half vs second half): {np.corrcoef(r1, r2)[0, 1]:+.2f}")
