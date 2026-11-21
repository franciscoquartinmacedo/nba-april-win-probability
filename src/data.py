"""Data loading, integrity checks and fold definitions.

Design rule: April (blind test) and March (final test) are LOCKED by default.
Model selection code can only reach Oct-Feb through `load_development()`.
Locked months are only reachable through explicitly named functions that the
final evaluation script calls once, after everything is frozen.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "nba-win-probability-data.csv"

FINAL_TEST_START = pd.Timestamp("2026-03-01")   # March: scored once, after freezing
APRIL_START = pd.Timestamp("2026-04-01")        # April: blind, scored once

# Rolling-origin validation: each month is predicted using only earlier games.
VALIDATION_MONTHS = ["2025-12", "2026-01", "2026-02"]


def _read() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype={"game_id": str}, parse_dates=["game_date"])
    df["margin"] = df["home_points"] - df["away_points"]      # post-game: NEVER a feature of its own game
    df["home_win"] = (df["margin"] > 0).astype(int)
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def load_development() -> pd.DataFrame:
    """Oct-Feb only. The only loader used while building/selecting models."""
    df = _read()
    return df[df["game_date"] < FINAL_TEST_START].reset_index(drop=True)


def load_final_test_march() -> pd.DataFrame:  # called once, by the final script
    df = _read()
    return df[(df["game_date"] >= FINAL_TEST_START) & (df["game_date"] < APRIL_START)].reset_index(drop=True)


def load_final_test_context() -> pd.DataFrame:  # Oct-Mar: development + March, no April. Called by the March evaluation only.
    df = _read()
    return df[df["game_date"] < APRIL_START].reset_index(drop=True)


def load_all_for_final_evaluation() -> pd.DataFrame:  # called once, by the final script
    return _read()


def integrity_checks(df: pd.DataFrame) -> dict:
    """Scripted version of the checks done by hand earlier. Works on any slice."""
    out = {"rows": len(df), "game_id_unique": bool(df["game_id"].is_unique),
           "missing_values": int(df.isna().sum().sum()), "ties": int((df["margin"] == 0).sum())}
    parts = []
    for side in ("away", "home"):
        d = df[["game_id", "game_date", side, f"{side}_wins", f"{side}_losses"]].copy()
        d.columns = ["game_id", "date", "team", "w", "l"]
        d["won"] = (df["home_win"] if side == "home" else 1 - df["home_win"]).values
        parts.append(d)
    L = pd.concat(parts).sort_values(["team", "date", "game_id"])
    L["games_before"] = L.groupby("team").cumcount()
    L["wins_before"] = L.groupby("team")["won"].cumsum() - L["won"]
    # Records in the file are season-to-date, so they only reconcile with the slice
    # if the slice starts at the season opener (true for development data).
    out["record_games_mismatch"] = int(((L["w"] + L["l"]) != L["games_before"]).sum())
    out["record_wins_mismatch"] = int((L["w"] != L["wins_before"]).sum())
    out["team_twice_same_day"] = int(L.duplicated(["team", "date"]).sum())
    return out


def rolling_folds(df: pd.DataFrame):
    """Yield (month, train_idx, val_idx). Train = every game strictly before the month."""
    for m in VALIDATION_MONTHS:
        start = pd.Timestamp(m + "-01")
        end = start + pd.offsets.MonthBegin(1)
        train = df.index[df["game_date"] < start].to_numpy()
        val = df.index[(df["game_date"] >= start) & (df["game_date"] < end)].to_numpy()
        yield m, train, val
