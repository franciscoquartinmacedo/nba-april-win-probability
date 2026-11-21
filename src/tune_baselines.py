"""Baselines (always-home, log5, recalibrated log5): score on the Dec/Jan/Feb rolling validation. Never touches March or April."""
import numpy as np
import pandas as pd

import baselines as B
import data as D
import metrics as M

df = D.load_development()
assert df["game_date"].max() < D.FINAL_TEST_START, "locked months leaked into development data"

print("INTEGRITY CHECKS (Oct-Feb):", D.integrity_checks(df))

# Re-estimate k on development data only (an earlier ad-hoc estimate used Oct-Mar).
parts = []
for s in ("away", "home"):
    x = df[["game_date", s]].copy(); x.columns = ["date", "team"]
    x["won"] = (df["home_win"] if s == "home" else 1 - df["home_win"]).values
    parts.append(x)
L = pd.concat(parts)
fin = L.groupby("team")["won"].agg(["mean", "size"])
tal = fin["mean"].var(ddof=1) - (fin["mean"] * (1 - fin["mean"]) / fin["size"]).mean()
print(f"k re-estimated on Oct-Feb only: {0.25 / tal:.1f} (talent sd {tal ** 0.5:.3f})")

models = {
    "1 constant home rate": lambda tr, va: B.constant_home_rate(df, tr, va),
    "2 log5 (k=10)": lambda tr, va: B.log5(df, tr, va, k=10),
}
store = {name: {"y": [], "p": []} for name in models}
rows = []
for month, tr, va in D.rolling_folds(df):
    y = df.loc[va, "home_win"].to_numpy()
    for name, fn in models.items():
        p = fn(tr, va)
        store[name]["y"].append(y); store[name]["p"].append(p)
        rows.append({"fold": month, "model": name, "n_train": len(tr), **M.summarise(y, p)})

print("\nPER-FOLD"); print(pd.DataFrame(rows).round(4).to_string(index=False))

print("\nPOOLED Dec-Feb")
pooled = {}
for name, s in store.items():
    y, p = np.concatenate(s["y"]), np.concatenate(s["p"]); pooled[name] = (y, p)
    lo, hi = M.bootstrap_ci(y, p, M.log_loss)
    m = M.summarise(y, p)
    print(f"{name:22s} logloss {m['log_loss']:.4f} [{lo:.4f},{hi:.4f}]  brier {m['brier']:.4f}  acc {m['accuracy']:.3f}  n={m['n']}")

# Paired comparison: is log5 better than the constant, game by game?
y, p0 = pooled["1 constant home rate"]; _, p1 = pooled["2 log5 (k=10)"]
ll = lambda y_, p_: -(y_ * np.log(p_) + (1 - y_) * np.log(1 - p_))
d = ll(y, p0) - ll(y, p1)          # positive => log5 better
rng = np.random.default_rng(0)
boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
print(f"\nPaired log-loss gain of log5 over constant: {d.mean():.4f}  95% CI [{np.percentile(boots, 2.5):.4f}, {np.percentile(boots, 97.5):.4f}]")
