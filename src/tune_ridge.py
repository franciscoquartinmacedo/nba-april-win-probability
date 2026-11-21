"""Ridge and Elo settings grid, with a games-played-dependent probability map and a nested selection check.
Dec/Jan/Feb rolling validation only. March and April are never loaded."""
import itertools

import numpy as np
import pandas as pd

import baselines as B
import data as D
import elo as E
import metrics as M
import ridge as R

df = D.load_development()
assert df["game_date"].max() < D.FINAL_TEST_START
y_all = df["home_win"].to_numpy()
folds = list(D.rolling_folds(df))
val_idx = np.concatenate([va for _, _, va in folds]); y_val = y_all[val_idx]
fold_of_val = np.concatenate([[m] * len(va) for m, _, va in folds])
pg = np.minimum(df["home_wins"] + df["home_losses"], df["away_wins"] + df["away_losses"]).to_numpy()
U = E.early_weight(pg)

TRANSFORMS = [("none", None), ("hard", 30), ("hard", 25), ("hard", 20), ("tanh", 25), ("binary", None)]
N_GRID = [0, 5, 10, 15]
K_INF = [0.005, 0.01, 0.02, 0.04, 0.06, 0.10]
K0 = [None, 5, 10, 20, 40, 80]
LAMBDAS = [2.5, 5, 10, 20, 40, 80, 160, 320]
tname = lambda k, c: f"{k}{'' if c is None else c}"


def score_simple(d, N):
    p = np.empty(len(val_idx)); pos = 0
    for _, tr, va in folds:
        keep = tr[pg[tr] >= N]
        a, b = E.fit_map(d[keep], y_all[keep]); p[pos:pos + len(va)] = E.predict_map(d[va], a, b); pos += len(va)
    return p


def score_gp(d):
    p = np.empty(len(val_idx)); pos = 0
    for _, tr, va in folds:
        coef, b = E.fit_map_gp(d[tr], U[tr], y_all[tr]); p[pos:pos + len(va)] = E.predict_map_gp(d[va], U[va], coef, b); pos += len(va)
    return p


rows, preds = [], {}
def add(family, cfg, kind, p1, p2, maptype, N, p):
    name = f"{cfg}|map={maptype}|N={N}"; preds[name] = p
    rows.append({"family": family, "config": name, "rating_cfg": cfg, "transform": kind, "param1": p1, "param2": p2,
                 "map": maptype, "N": N, **M.summarise(y_val, p)})


for (kind, c), k_inf, k0 in itertools.product(TRANSFORMS, K_INF, K0):
    d, _, _ = E.run_elo(df, k_inf, k0, kind, c, kscale=0.5)
    cfg = f"elo|{tname(kind, c)}|K={k_inf}|{'fixed' if k0 is None else f'k0={k0}'}"
    for N in N_GRID:
        add("elo", cfg, tname(kind, c), k_inf, "fixed" if k0 is None else k0, "simple", N, score_simple(d, N))
    add("elo", cfg, tname(kind, c), k_inf, "fixed" if k0 is None else k0, "gp", 0, score_gp(d))
for (kind, c), lam in itertools.product(TRANSFORMS, LAMBDAS):
    d = R.run_ridge(df, lam, kind, c)
    cfg = f"ridge|{tname(kind, c)}|lam={lam}"
    for N in N_GRID:
        add("ridge", cfg, tname(kind, c), lam, "-", "simple", N, score_simple(d, N))
    add("ridge", cfg, tname(kind, c), lam, "-", "gp", 0, score_gp(d))

res = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
res.to_csv("../outputs/ridge_grid_results.csv", index=False)
prior = 96 + 72 + 18   # v1 (fixed-K rows are re-evaluated here, decaying rows used the old scaling) + exploratory probe
print(f"Variants scored on Dec-Feb: {len(res)}  (+ about {prior} earlier distinct evaluations => ~{len(res) + prior} total)")

pd.set_option("display.width", 220)
print("\nTOP 12 overall"); print(res[["family", "config", "log_loss", "brier", "accuracy"]].head(12).round(4).to_string(index=False))

for mp in ["simple", "gp"]:
    r = res[res["map"] == mp]
    print(f"\n[{mp} map] best elo: {r[r.family=='elo'].iloc[0]['config']}  {r[r.family=='elo'].iloc[0]['log_loss']:.4f}")
    print(f"[{mp} map] best ridge: {r[r.family=='ridge'].iloc[0]['config']}  {r[r.family=='ridge'].iloc[0]['log_loss']:.4f}")

print("\nBOUNDARY CHECK (N=0, simple map): best value of each parameter")
r0 = res[(res.N == 0) & (res["map"] == "simple")]
b_e, b_r = r0[r0.family == "elo"].iloc[0], r0[r0.family == "ridge"].iloc[0]
print("  best elo   :", b_e.config, "| k0 grid", K0, "| K_inf grid", K_INF)
print("  best ridge :", b_r.config, "| lambda grid", LAMBDAS)
print("  top-5 ridge lambdas:", [c.split('lam=')[1].split('|')[0] for c in r0[r0.family=='ridge'].config.head(5)])
print("  top-5 elo k0       :", [c.split('|')[3] for c in r0[r0.family=='elo'].config.head(5)])

print("\nSIMPLE vs GP map, same rating config (N=0): mean log loss over all rating configs")
piv = res[res.N == 0].pivot_table(index="rating_cfg", columns="map", values="log_loss")
print("  mean simple", round(piv.simple.mean(), 4), " mean gp", round(piv.gp.mean(), 4),
      "| gp better in", int((piv.gp < piv.simple).sum()), "of", len(piv), "configs; median gain", round((piv.simple - piv.gp).median(), 4))

# Baselines on same games
p_const = np.concatenate([B.constant_home_rate(df, tr, va) for _, tr, va in folds])
p_log5 = np.concatenate([B.log5(df, tr, va, k=10) for _, tr, va in folds])
print("\nREFERENCE constant", round(M.log_loss(y_val, p_const), 4), " log5", round(M.log_loss(y_val, p_log5), 4))
ll = lambda p: -(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
rng = np.random.default_rng(0)
def paired(pa, pb, mask=None):
    dd = ll(pa) - ll(pb)
    dd = dd if mask is None else dd[mask]; bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(5000)]
    return dd.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
top_s = res[res["map"] == "simple"].iloc[0]["config"]; top_g = res[res["map"] == "gp"].iloc[0]["config"]
for nm, t in [("best simple-map", top_s), ("best gp-map", top_g)]:
    m_, lo, hi = paired(p_log5, preds[t]); print(f"{nm} ({t}) gain over log5: {m_:.4f} [{lo:.4f}, {hi:.4f}]")
m_, lo, hi = paired(preds[top_s], preds[top_g]); print(f"gp-map best over simple-map best: {m_:.4f} [{lo:.4f}, {hi:.4f}]")

print("\nEARLY vs LATE calibration (best simple vs best gp config): log loss by min games played")
gpv = pg[val_idx]
for lo_, hi_ in [(0, 20), (20, 30), (30, 45), (45, 90)]:
    mk = (gpv >= lo_) & (gpv < hi_)
    if mk.sum():
        print(f"  gp {lo_:2d}-{hi_:2d} n={mk.sum():3d}  simple {M.log_loss(y_val[mk], preds[top_s][mk]):.4f}  gp {M.log_loss(y_val[mk], preds[top_g][mk]):.4f}  log5 {M.log_loss(y_val[mk], p_log5[mk]):.4f}  "
              f"mean pred s/g {preds[top_s][mk].mean():.3f}/{preds[top_g][mk].mean():.3f} act {y_val[mk].mean():.3f}")

# Nested selection: choose among ALL variants using only earlier validation folds, score on the next fold
names = list(preds.keys()); P = np.vstack([preds[n] for n in names])
LLm = -(y_val * np.log(P) + (1 - y_val) * np.log(1 - P))        # (variants, games)
nest, base_l5, base_c, picks = [], [], [], {}
for target, earlier in [("2026-01", ["2025-12"]), ("2026-02", ["2025-12", "2026-01"])]:
    sel = np.isin(fold_of_val, earlier); tgt = fold_of_val == target
    best_i = int(np.argmin(LLm[:, sel].mean(axis=1))); picks[target] = names[best_i]
    nest.append(LLm[best_i, tgt])
    base_l5.append(ll(p_log5)[tgt]); base_c.append(ll(p_const)[tgt])
nest, base_l5, base_c = map(np.concatenate, (nest, base_l5, base_c))
print("\nNESTED SELECTION (choose best of all variants on earlier folds, score on next fold; Jan+Feb only):")
for k, v in picks.items(): print(f"  {k}: picked {v}")
bs = [(base_l5 - nest)[rng.integers(0, len(nest), len(nest))].mean() for _ in range(5000)]
print(f"  nested log loss {nest.mean():.4f} | log5 {base_l5.mean():.4f} | constant {base_c.mean():.4f}")
print(f"  nested gain over log5 {np.mean(base_l5 - nest):.4f} [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]")
jf = ~(fold_of_val == "2025-12")
print(f"  (for comparison, the non-nested best config on the same Jan+Feb games: {LLm[names.index(top_s)][jf].mean():.4f})")

print("\nPER-FOLD best simple / best gp / log5 / constant")
for m in ["2025-12", "2026-01", "2026-02"]:
    mk = fold_of_val == m
    print(m, round(M.log_loss(y_val[mk], preds[top_s][mk]), 4), round(M.log_loss(y_val[mk], preds[top_g][mk]), 4),
          round(M.log_loss(y_val[mk], p_log5[mk]), 4), round(M.log_loss(y_val[mk], p_const[mk]), 4))
