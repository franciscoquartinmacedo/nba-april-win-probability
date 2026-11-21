"""Leak-free pre-game features. Every value for game i uses only games played strictly before game i's date.

Assumptions (all flagged in the report; none tuned on outcomes):
  * rest days capped at 4; a team's first game is assigned 3; back-to-back = played the previous calendar day
  * form  = (last-10 mean margin - season-to-date mean margin) * n10/(n10+k)
  * split = (venue-specific mean margin - overall mean margin) * n_venue/(n_venue+k), venue = home games for the
            home team, away games for the away team
    (v1 of these two used mismatched shrinkage and correlated mechanically with strength; fixed after diagnosis)
  * box   = shrunk season-to-date average of (own - opponent) turnovers / rebounds / fouls
  * shrinkage uses K_SHRINK pseudo-games at zero (value 0 = league average / no information)
"""
import numpy as np
import pandas as pd

K_SHRINK = 10.0
REST_CAP, REST_FIRST = 4, 3
FEATURE_BLOCKS = {
    "rest": ["rest_diff", "home_b2b", "away_b2b"],
    "form": ["form_diff"],
    "split": ["split_diff"],
    "box": ["tov_diff", "reb_diff", "fou_diff"],
}


def build_features(df: pd.DataFrame, freeze_after=None) -> pd.DataFrame:
    """freeze_after=None: every game's features use all earlier games (rolling inputs, 'version B').
    freeze_after=<date>: games on/after that date add NOTHING to the outcome-based histories (form, venue split, box-score),
    so every such game sees the state as of the day before. Rest days still come from the schedule (dates only), which a
    forecaster knows in advance ('version A')."""
    hist = {}   # team -> list of (date, was_home, margin_for, tov_d, reb_d, fou_d)
    last_date = {}   # team -> date of most recent game (schedule information, not an outcome)
    out = []
    for r in df.itertuples():
        row = {}
        stats = {}
        for side, team, is_home in (("home", r.home, True), ("away", r.away, False)):
            h = hist.get(team, [])
            n = len(h)
            if team in last_date:
                last_gap = (r.game_date - last_date[team]).days
                rest = min(last_gap, REST_CAP); b2b = int(last_gap == 1)
            else:
                rest, b2b = REST_FIRST, 0
            m = np.array([x[2] for x in h]) if n else np.array([])
            # Deviation-from-own-average features, scaled by evidence (n/(n+k)). Using the SAME mean basis for both
            # terms avoids the mechanical correlation with team strength that mismatched shrinkage created.
            overall_mean = m.mean() if n else 0.0
            last10 = m[-10:]
            recent = (last10.mean() - overall_mean) * len(last10) / (len(last10) + K_SHRINK) if n else 0.0
            venue = np.array([x[2] for x in h if x[1] == is_home])
            venue_dev = (venue.mean() - overall_mean) * len(venue) / (len(venue) + K_SHRINK) if len(venue) else 0.0
            overall = overall_mean  # (kept for clarity; not used below)
            recent_dev = recent
            box = [(np.sum([x[k] for x in h]) / (n + K_SHRINK)) if n else 0.0 for k in (3, 4, 5)]
            stats[side] = dict(rest=rest, b2b=b2b, form=recent_dev, split=venue_dev, tov=box[0], reb=box[1], fou=box[2])
        H, A = stats["home"], stats["away"]
        row.update(rest_diff=H["rest"] - A["rest"], home_b2b=H["b2b"], away_b2b=A["b2b"], form_diff=H["form"] - A["form"],
                   split_diff=H["split"] - A["split"], tov_diff=H["tov"] - A["tov"], reb_diff=H["reb"] - A["reb"], fou_diff=H["fou"] - A["fou"])
        out.append(row)
        # only now add this game to histories (post-game info is used only for LATER games)
        last_date[r.home] = r.game_date; last_date[r.away] = r.game_date
        if freeze_after is not None and r.game_date >= freeze_after:
            continue            # frozen: this game's outcome and box score feed nothing
        mg = r.home_points - r.away_points
        hist.setdefault(r.home, []).append((r.game_date, True, mg, r.home_turnovers - r.away_turnovers, r.home_rebounds - r.away_rebounds, r.home_fouls - r.away_fouls))
        hist.setdefault(r.away, []).append((r.game_date, False, -mg, r.away_turnovers - r.home_turnovers, r.away_rebounds - r.home_rebounds, r.away_fouls - r.home_fouls))
    return pd.DataFrame(out, index=df.index)
