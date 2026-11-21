#!/usr/bin/env bash
# Reproduces every number, table and figure in the report from data/nba-win-probability-data.csv. Run from the repo root: bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")/src"
step() { echo; echo "=== $1 ==="; }

step "1/4 Settings, chosen on Oct-Feb only (log5, Elo, ridge, features)"
python -W ignore tune_baselines.py
python -W ignore tune_elo.py
python -W ignore tune_ridge.py
python -W ignore tune_features.py
python -W ignore check_extra_features.py

step "2/4 Final scoring: frozen settings, March scored once, April blind"
python -W ignore final_eval.py dry        # Oct-Feb check; must print 0.6937 / 0.6574 / 0.6419 / 0.6416
python -W ignore final_eval.py march
python -W ignore run_april.py
python -W ignore test_blind.py            # must print True / True (no look-ahead)
python -W ignore metrics_deep.py

step "3/4 Pooled Dec-Apr analysis and April files"
python -W ignore metrics_pooled.py
python -W ignore make_april_files.py
python -W ignore make_workbook.py

step "4/4 Figures"
python -W ignore make_figure_design.py
python -W ignore make_figure_results.py
python -W ignore make_figure_reliability.py
python -W ignore make_figure_pooled.py
echo; echo "Done. Outputs are in outputs/"
