"""Baselines 1 and 2: constant home rate and log5."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from metrics import log_loss


def constant_home_rate(df, train_idx, val_idx):
    """Baseline 1: predict the training-set home win rate for every game."""
    return np.full(len(val_idx), df.loc[train_idx, "home_win"].mean())


def shrunk_win_pct(wins, losses, k):
    """Season-to-date win% shrunk toward 0.5 with k pseudo-games (k=0 -> raw, 0/0 -> 0.5)."""
    g = wins + losses
    return (wins + 0.5 * k) / (g + k) if k > 0 else np.where(g > 0, wins / np.maximum(g, 1), 0.5)


def _log5_offset(df, k):
    """log5 in logit form is logit(p_home) - logit(p_away); the home edge is added on top."""
    ph = np.clip(shrunk_win_pct(df["home_wins"].to_numpy(), df["home_losses"].to_numpy(), k), 1e-3, 1 - 1e-3)
    pa = np.clip(shrunk_win_pct(df["away_wins"].to_numpy(), df["away_losses"].to_numpy(), k), 1e-3, 1 - 1e-3)
    return logit(ph) - logit(pa)


def log5(df, train_idx, val_idx, k=10):
    """Baseline 2: log5 with coefficient fixed at 1 and a single home-advantage constant fitted on training."""
    x_tr, y_tr = _log5_offset(df.loc[train_idx], k), df.loc[train_idx, "home_win"].to_numpy()
    res = minimize_scalar(lambda h: log_loss(y_tr, expit(x_tr + h)), bounds=(-2, 2), method="bounded")
    return expit(_log5_offset(df.loc[val_idx], k) + res.x)
