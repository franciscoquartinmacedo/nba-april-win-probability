"""Scoring rules and bootstrap intervals."""
import numpy as np

EPS = 1e-12


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def accuracy(y, p):
    return float(np.mean((p >= 0.5) == (y == 1)))


def summarise(y, p):
    return {"log_loss": log_loss(y, p), "brier": brier(y, p), "accuracy": accuracy(y, p), "n": int(len(y))}


def bootstrap_ci(y, p, fn=log_loss, n_boot=2000, seed=0):
    """95% percentile CI by resampling games. Note: ignores within-team dependence."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = [fn(y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
