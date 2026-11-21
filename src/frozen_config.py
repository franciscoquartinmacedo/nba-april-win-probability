"""Everything below was chosen on Oct-Feb validation (Dec/Jan/Feb rolling folds) and is FROZEN before March is scored.
Do not edit after the March run; any change would have to be reported as a post-test modification."""

FROZEN = {
    "ridge_rating": {"lambda": 80, "margin_transform": "none"},          # ridge grid: interior optimum, no cap
    "elo_robustness": {"K_inf": 0.01, "decay": "fixed", "margin_transform": "none", "kscale": 0.5},
    "log5": {"k_shrink": 10},                                            # data-derived estimate was 10-12.7 (Oct-Feb)
    "probability_map": "unpenalised 2-parameter logistic on pre-game expected margin, fitted on training games only",
    "features": {"shrink_k": 10, "rest_cap": 4, "rest_first_game": 3, "form_window": 10},
    "logistic_C": {"ridge+b2b": 1.0, "ridge+all": 0.1},                  # all: pooled-best C from the Dec-Feb feature grid
    "models": ["constant", "log5", "log5_recalibrated", "ridge_rating", "ridge_b2b", "ridge_all", "elo_rating"],
    "primary_model": "ridge_rating",
    "scoring": "March scored once; map/model fitted on games before 1 March; rating/feature INPUTS roll forward through March",
    "april": "two versions: A = everything frozen at 31 March, B = frozen model with rolling inputs",
}
