"""Builds ../outputs/nba_april_predictions_and_extras.xlsx from stored outputs (no model is run).
Tabs: Predictions (headline, version A) | Extras (other models, outcomes). Decisions live in the report."""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

pred = pd.read_csv("../outputs/april_predictions.csv", dtype={"game_id": str}, parse_dates=["game_date"])
ev = pd.read_csv("../outputs/april_evaluation.csv", dtype={"game_id": str}, parse_dates=["game_date"])
HDR = PatternFill("solid", fgColor="F3F3F3"); WRAP = Alignment(wrap_text=True, vertical="top")
wb = Workbook()


def table(ws, header, rows, widths, wrap_cols=()):
    ws.append(header)
    for c in ws[1]: c.fill = HDR; c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical="top")
    for r in rows: ws.append(r)
    for i, w in enumerate(widths, 1): ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            if c.column in wrap_cols: c.alignment = WRAP
            else: c.alignment = Alignment(vertical="top")
    ws.freeze_panes = "A2"


wb.remove(wb.active)  # data only: decisions and missing variables live in the report

# ---------------- Predictions ----------------
ws = wb.create_sheet("Predictions")
rows = [[r.game_id, r.game_date.to_pydatetime(), r.away, r.home, float(r.p_home)] for r in pred.itertuples()]
table(ws, ["Game ID", "Date", "Away", "Home", "P(home win)"], rows, [14, 12, 8, 8, 13])
for row in ws.iter_rows(min_row=2):
    row[1].number_format = "yyyy-mm-dd"; row[4].number_format = "0.000"; row[0].number_format = "@"

# ---------------- Extras ----------------
ws = wb.create_sheet("Extras")
cols = ["game_id", "game_date", "away", "home", "home_win", "margin", "p_home_ridge_rating_A_HEADLINE", "p_elo", "p_log5", "p_log5_recal", "p_ridge_b2b", "p_ridge_all", "p_constant", "p_ridge_progress_POSTHOC", "p_ridge_rating_B_rolling_inputs"]
hdr = ["Game ID", "Date", "Away", "Home", "Home won (1/0)", "Home margin", "Ridge (headline, A)", "Elo", "Log5", "Log5 recalibrated", "Ridge + back-to-back", "Ridge + all features", "Always-home (constant)", "Ridge progress (post-hoc)", "Ridge, rolling inputs (B)"]
rows = []
for r in ev[cols].itertuples(index=False):
    rows.append([r[0], r[1].to_pydatetime(), r[2], r[3], int(r[4]), int(r[5])] + [float(x) for x in r[6:]])
table(ws, hdr, rows, [14, 12, 8, 8, 12, 12] + [14] * 9, wrap_cols=(1,))
for row in ws.iter_rows(min_row=2):
    row[1].number_format = "yyyy-mm-dd"; row[0].number_format = "@"
    for c in row[6:]: c.number_format = "0.000"
for c in ws[1]: c.alignment = Alignment(wrap_text=True, vertical="top")

wb.save("../outputs/nba_april_predictions_and_extras.xlsx")
print("saved")
