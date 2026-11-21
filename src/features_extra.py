"""Extra pre-game features for the feature checks (rest categories, multi-window form). Leak-free by construction:
each game's values use only that team's games with an earlier date."""
import numpy as np
import pandas as pd

K_SHRINK = 10.0
REST_CAP, REST_FIRST = 4, 3
WINDOWS = [3, 5, 10, 15, 20, 30]


def build_extra(df: pd.DataFrame) -> pd.DataFrame:
    hist, out = {}, []
    for r in df.itertuples():
        side = {}
        for key, team in (("h", r.home), ("a", r.away)):
            h = hist.get(team, [])
            n = len(h)
            rest = min((r.game_date - h[-1][0]).days, REST_CAP) if n else REST_FIRST
            m = np.array([x[1] for x in h]) if n else np.array([])
            d = {"rest": rest}
            for w in WINDOWS:
                last = m[-w:]
                d[f"form{w}"] = (last.mean() - m.mean()) * len(last) / (len(last) + K_SHRINK) if n else 0.0
            side[key] = d
        row = {"h_rest": side["h"]["rest"], "a_rest": side["a"]["rest"]}
        row["rest_b1"] = None
        for key in ("h", "a"):
            rs = side[key]["rest"]
            row[f"{key}_r1"] = int(rs == 1); row[f"{key}_r2"] = int(rs == 2)      # reference category: >=3 days
        row["rest_log_diff"] = np.log(side["h"]["rest"]) - np.log(side["a"]["rest"])
        for w in WINDOWS:
            row[f"form{w}_diff"] = side["h"][f"form{w}"] - side["a"][f"form{w}"]
        out.append(row)
        mg = r.home_points - r.away_points
        hist.setdefault(r.home, []).append((r.game_date, mg)); hist.setdefault(r.away, []).append((r.game_date, -mg))
    res = pd.DataFrame(out, index=df.index).drop(columns="rest_b1")
    return res
