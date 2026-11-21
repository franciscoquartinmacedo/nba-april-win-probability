"""April blind test, scored ONCE. Version A: everything outcome-based frozen at 31 March (schedule info only varies).
Version B: rolling inputs (frozen model, inputs update with each April game day). Fit of probability maps/logistics: Oct-Mar only."""
import numpy as np, pandas as pd
import data as D, final_eval as FE
from frozen_config import FROZEN
full = D.load_all_for_final_evaluation(); fz = D.APRIL_START
models = FROZEN["models"] + ["ridge_progress", "ridge_progress_k127"]
res = {}
for ver in ("A", "B"):
    if ver == "A":
        df = FE.freeze_records(full, fz); inputs = FE.compute_inputs(df, fz)
    else:
        df = full.copy(); df["gp_min"] = np.minimum(df["home_wins"] + df["home_losses"], df["away_wins"] + df["away_losses"]); inputs = FE.compute_inputs(df)
    te, P = FE.predict_period(df, inputs, fz, pd.Timestamp("2026-12-31"))
    tab = FE.summarise_models(df, te, P, f"APRIL version {ver} (scored once)", models)
    out = df.loc[te, ["game_id", "game_date", "home", "away", "home_win", "gp_min"]].copy()
    for m, p in P.items(): out[m] = p
    out.to_csv(f"../outputs/april_predictions_{ver}.csv", index=False); tab.to_csv(f"../outputs/april_summary_{ver}.csv", index=False); res[ver] = (df, te, P)
# A vs B paired, and calibration slopes
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
print("\n=== A vs B (paired log-loss; positive = B better) and calibration slope (1 = perfect; >1 under-confident) ===")
dfA, teA, PA = res["A"]; _, teB, PB = res["B"]; y = dfA.loc[teA, "home_win"].to_numpy()
ll = lambda p: -(y*np.log(p)+(1-y)*np.log(1-p)); rng = np.random.default_rng(0)
for m in models:
    dd = ll(PA[m]) - ll(PB[m]); bs = [dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(5000)]
    s = {}
    for v, P in (("A", PA), ("B", PB)):
        lr = LogisticRegression(C=np.inf).fit(logit(np.clip(P[m],1e-6,1-1e-6)).reshape(-1,1), y); s[v] = lr.coef_[0][0]
    print(f"{m:18s} A-B {dd.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  slope A {s['A']:.2f} B {s['B']:.2f}")
