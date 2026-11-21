"""Blindness test on a pseudo-April: freeze at 1 March, perturb March outcomes/box scores; A inputs must not move, B must."""
import numpy as np, pandas as pd
import data as D, final_eval as FE
ctx = D.load_final_test_context(); fz = D.FINAL_TEST_START
rng = np.random.default_rng(1)
def perturb(df):
    p = df.copy(); m = p["game_date"] >= fz
    for c in ["home_points","away_points","home_turnovers","away_turnovers","home_rebounds","away_rebounds","home_fouls","away_fouls"]:
        p.loc[m, c] = rng.permutation(p.loc[m, c].to_numpy())
    p.loc[m, "home_win"] = (p.loc[m, "home_points"] > p.loc[m, "away_points"]).astype(int)
    p["margin"] = p["home_points"] - p["away_points"]
    return p
def pack(df, frozen):
    d = FE.freeze_records(df, fz) if frozen else df
    r, e, f = FE.compute_inputs(d, fz if frozen else None)
    m = (df["game_date"] >= fz).to_numpy()
    return r[m], e[m], f[m].to_numpy(float), d.loc[m, ["home_wins","home_losses","away_wins","away_losses","gp_min"]].to_numpy(float) if frozen else None
a0, a1 = pack(ctx, True), pack(perturb(ctx), True)
print("A ridge/elo/features/records unchanged:", np.allclose(a0[0],a1[0]), np.allclose(a0[1],a1[1]), np.allclose(a0[2],a1[2]), np.allclose(a0[3],a1[3]))
b0, b1 = pack(ctx, False), pack(perturb(ctx), False)
print("B ridge/elo/features changed (expected True):", not np.allclose(b0[0],b1[0]), not np.allclose(b0[1],b1[1]), not np.allclose(b0[2],b1[2]))
assert np.allclose(a0[0],a1[0]) and np.allclose(a0[1],a1[1]) and np.allclose(a0[2],a1[2]) and np.allclose(a0[3],a1[3]), "version A inputs moved: look-ahead"
assert (not np.allclose(b0[0],b1[0])) and (not np.allclose(b0[1],b1[1])) and (not np.allclose(b0[2],b1[2])), "version B inputs did not move: the test is not sensitive"
print("test_blind passed")
