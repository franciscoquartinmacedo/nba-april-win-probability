"""Expanding-window ridge rating (Bradley-Terry / SRS style, margin scale).

Before each game day, fit on ALL earlier games only:
    m'_g = c + r_home - r_away + noise ,  penalty  lam * sum(r^2)  (c unpenalised)
lam is the strength of the pull toward the league average, in 'pseudo-games'
(each game adds 1 to a team's diagonal), so it plays the same role as k0 in Elo.
"""
import numpy as np

from elo import transform_margin


def run_ridge(df, lam, kind="none", c=None, freeze_idx=None):
    """Returns pre-game predicted home margin d for every game (0 before any data exists)."""
    teams = {t: i for i, t in enumerate(sorted(set(df["home"]) | set(df["away"])))}
    T = len(teams); n = len(df)
    h = df["home"].map(teams).to_numpy(); a = df["away"].map(teams).to_numpy()
    m = transform_margin(df["margin"].to_numpy(), kind, c)
    X = np.zeros((n, T + 1)); X[:, 0] = 1.0                      # column 0 = home-advantage intercept
    X[np.arange(n), 1 + h] = 1.0; X[np.arange(n), 1 + a] = -1.0
    P = np.eye(T + 1) * lam; P[0, 0] = 0.0
    dates = df["game_date"].to_numpy()
    d = np.zeros(n)
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, n]
    for s, e in zip(starts[:-1], starts[1:]):
        if s == 0:
            continue                                              # nothing earlier to learn from
        sf = s if freeze_idx is None else min(s, freeze_idx)      # frozen: never learn from rows >= freeze_idx
        Xp, mp = X[:sf], m[:sf]
        theta = np.linalg.solve(Xp.T @ Xp + P + 1e-9 * np.eye(T + 1), Xp.T @ mp)
        d[s:e] = X[s:e] @ theta                                   # fit on strictly earlier dates
    return d
