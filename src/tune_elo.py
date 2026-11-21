"""Elo settings grid, scored on the Dec/Jan/Feb rolling validation. March and April are never loaded."""
import itertools

import numpy as np
import pandas as pd

import baselines as B
import data as D
import elo as E
import metrics as M

df = D.load_development()
assert df["game_date"].max() < D.FINAL_TEST_START
y_all = df["home_win"].to_numpy()
folds = list(D.rolling_folds(df))
val_idx = np.concatenate([va for _, _, va in folds])
y_val = y_all[val_idx]

TRANSFORMS = [("none", None), ("hard", 30), ("hard", 25), ("hard", 20), ("tanh", 25), ("binary", None)]
K_INF = [0.02, 0.04, 0.06, 0.10]
K0 = [None, 5, 10, 20]            # None = fixed K ; others = decaying with k0 pseudo-games
GRID = list(itertools.product(TRANSFORMS, K_INF, K0))
print("configurations tried:", len(GRID))

pgames = np.minimum(df["home_wins"] + df["home_losses"], df["away_wins"] + df["away_losses"]).to_numpy()


def evaluate(kind, c, k_inf, k0):
    d, _, _ = E.run_elo(df, k_inf, k0, kind, c)
    p_val = np.empty(len(val_idx)); pos = 0
    for _, tr, va in folds:
        a, b = E.fit_map(d[tr], y_all[tr])
        p_val[pos:pos + len(va)] = E.predict_map(d[va], a, b); pos += len(va)
    return p_val


rows, preds = [], {}
for (kind, c), k_inf, k0 in GRID:
    p = evaluate(kind, c, k_inf, k0)
    name = f"{kind}{'' if c is None else c}|K={k_inf}|{'fixed' if k0 is None else f'k0={k0}'}"
    preds[name] = p
    m = M.summarise(y_val, p)
    rows.append({"config": name, "transform": f"{kind}{'' if c is None else c}", "K_inf": k_inf,
                 "decay": "fixed" if k0 is None else k0, **m})
res = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
res.to_csv("../outputs/elo_grid_results.csv", index=False)

pd.set_option("display.width", 200)
print("\nTOP 10 by pooled Dec-Feb log loss"); print(res.head(10).round(4).to_string(index=False))
print("\nWORST 3"); print(res.tail(3).round(4).to_string(index=False))
print(f"\nlog-loss range across all {len(res)} configs: {res.log_loss.min():.4f} .. {res.log_loss.max():.4f}; median {res.log_loss.median():.4f}")

print("\nMARGINAL EFFECTS (mean pooled log loss over the rest of the grid; lower = better)")
for col in ["transform", "K_inf", "decay"]:
    print(res.groupby(col)["log_loss"].agg(["mean", "min"]).round(4).T.to_string(), "\n")

# Reference baselines on identical games
p_const = np.concatenate([B.constant_home_rate(df, tr, va) for _, tr, va in folds])
p_log5 = np.concatenate([B.log5(df, tr, va, k=10) for _, tr, va in folds])
print("REFERENCE  constant:", round(M.log_loss(y_val, p_const), 4), " log5(k=10):", round(M.log_loss(y_val, p_log5), 4))

best = res.iloc[0]["config"]; pb = preds[best]
ll = lambda p: -(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
rng = np.random.default_rng(0)
for nm, pr in [("log5", p_log5), ("constant", p_const)]:
    dd = ll(pr) - ll(pb)
    bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(5000)]
    print(f"best Elo ({best}) gain over {nm}: {dd.mean():.4f}  95% CI [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]")

print("\nN SENSITIVITY: score only games where BOTH teams have played >= N (chosen on N=0)")
pg = pgames[val_idx]
for N in [0, 5, 10, 15, 20]:
    mk = pg >= N
    print(f"N={N:2d}  n={mk.sum():3d}  const {M.log_loss(y_val[mk], p_const[mk]):.4f}  log5 {M.log_loss(y_val[mk], p_log5[mk]):.4f}  elo {M.log_loss(y_val[mk], pb[mk]):.4f}")
