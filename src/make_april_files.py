"""Builds the April deliverables from the stored, once-scored predictions (no model is run here).
 ../outputs/april_predictions.csv : headline forecast only (version A = inputs frozen at 31 Mar; model = ridge_rating); no outcomes.
 ../outputs/april_evaluation.csv  : same games + outcomes + all other models (extras) + version B + post-hoc ridge_progress (labelled)."""
import pandas as pd
import data as D

A = pd.read_csv("../outputs/april_predictions_A.csv", dtype={"game_id": str}, parse_dates=["game_date"])
B = pd.read_csv("../outputs/april_predictions_B.csv", dtype={"game_id": str}).set_index("game_id")
full = D.load_all_for_final_evaluation().set_index("game_id")      # April locked loader (same access path as run_april.py)
A = A.sort_values(["game_date", "game_id"]).reset_index(drop=True)

main = A[["game_id", "game_date", "away", "home"]].copy()
main["game_date"] = main.game_date.dt.strftime("%Y-%m-%d"); main["p_home"] = A.ridge_rating.round(3)
main.to_csv("../outputs/april_predictions.csv", index=False)

ev = main.drop(columns="p_home").copy()
ev["home_win"] = A.home_win; ev["margin"] = A.game_id.map(full["margin"])
ev["p_home_ridge_rating_A_HEADLINE"] = A.ridge_rating.round(4)
for m, name in [("elo_rating", "p_elo"), ("log5", "p_log5"), ("log5_recalibrated", "p_log5_recal"), ("ridge_b2b", "p_ridge_b2b"),
                ("ridge_all", "p_ridge_all"), ("constant", "p_constant"), ("ridge_progress", "p_ridge_progress_POSTHOC")]:
    ev[name] = A[m].round(4)
ev["p_ridge_rating_B_rolling_inputs"] = A.game_id.map(B["ridge_rating"]).round(4)
ev.to_csv("../outputs/april_evaluation.csv", index=False)
print(main.shape, ev.shape)
