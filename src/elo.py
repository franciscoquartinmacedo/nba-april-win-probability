"""Margin-scale Elo.

Each team has a rating r (points). Pre-game expected home margin:
    d = r_home - r_away + H_t
H_t is the running mean of (transformed) home margins in all PRIOR games, so no
home-advantage parameter is tuned. After the game, with transformed margin m':
    err = m' - d ;  r_home += K_h * err ;  r_away -= K_a * err
Win probability is expit(a*d + b), with (a, b) fitted on training games only.

Learning rate per team after g games:
    fixed     : K = K_inf
    decaying  : K = max(K_inf, 1 / (g + 1 + k0))   # running-mean start with k0 pseudo-games of prior
All ratings start at 0 (no external information).
"""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def transform_margin(m, kind, c=None):
    m = np.asarray(m, dtype=float)
    if kind == "none":
        return m
    if kind == "hard":
        return np.clip(m, -c, c)
    if kind == "tanh":
        return c * np.tanh(m / c)
    if kind == "binary":          # win/loss only, on a +/-10 point scale
        return 10.0 * np.sign(m)
    raise ValueError(kind)


def run_elo(df, k_inf, k0=None, kind="none", c=None, kscale=1.0, freeze_idx=None):
    """Sequentially process games (df is date-ordered). Returns pre-game expected margin d and
    the games-played counts of each side BEFORE the game. Uses only past games for each prediction."""
    teams = {t: i for i, t in enumerate(sorted(set(df["home"]) | set(df["away"])))}
    h = df["home"].map(teams).to_numpy(); a = df["away"].map(teams).to_numpy()
    m = transform_margin(df["margin"].to_numpy(), kind, c)
    r = np.zeros(len(teams)); g = np.zeros(len(teams), dtype=int)
    n = len(df); d = np.empty(n); gh = np.empty(n, dtype=int); ga = np.empty(n, dtype=int)
    run_sum = 0.0
    frozen_H = None
    for t in range(n):
        if freeze_idx is not None and t == freeze_idx:
            frozen_H = run_sum / t if t > 0 else 0.0
        H = frozen_H if frozen_H is not None else (run_sum / t if t > 0 else 0.0)
        d[t] = r[h[t]] - r[a[t]] + H
        gh[t], ga[t] = g[h[t]], g[a[t]]
        if freeze_idx is not None and t >= freeze_idx:
            continue   # frozen: no rating update from this game
        err = m[t] - d[t]
        if k0 is None:
            kh = ka = k_inf
        else:
            # kscale=1.0 would over-scale (the error is shared by two teams);
            # kscale=0.5 is the running-mean scaling used here.
            kh = max(k_inf, kscale / (g[h[t]] + 1 + k0)); ka = max(k_inf, kscale / (g[a[t]] + 1 + k0))
        r[h[t]] += kh * err; r[a[t]] -= ka * err
        g[h[t]] += 1; g[a[t]] += 1
        run_sum += m[t]
    return d, gh, ga


def fit_map(d_train, y_train):
    """Fit p = expit(a*d + b) on training games only."""
    lr = LogisticRegression(C=np.inf, solver="lbfgs").fit(d_train.reshape(-1, 1), y_train)  # C=inf == no penalty
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def predict_map(d, a, b):
    return expit(a * d + b)


def early_weight(gp_min, k=10.0):
    """u = k/(k+games played by the less-experienced team): 1 at season start, -> 0 late. k=10 is an assumption."""
    return k / (k + np.asarray(gp_min, dtype=float))


def fit_map_gp(d, u, y):
    """Fit logit p = b + (a0 + a1*u) * d on training games only (3 parameters)."""
    X = np.column_stack([d, d * u])
    lr = LogisticRegression(C=np.inf, solver="lbfgs").fit(X, y)
    return lr.coef_[0].copy(), float(lr.intercept_[0])


def predict_map_gp(d, u, coef, b):
    return expit(coef[0] * d + coef[1] * d * u + b)
