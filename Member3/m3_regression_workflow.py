"""
Member 3 (Aman Kumar Lahoti) - Who students are: regression and statistical tests.
STARTER SCRIPT. It already runs end to end and prints the iteration-1 and
iteration-2 results. Your job: make the figures marked TODO, read the results,
and write Section 6 of the report. See TEAM_INSTRUCTIONS.md, Part C.

Run from the repository root:
    python preprocess_a3.py                     # once, creates data/a3_base.csv.gz
    python Member3/m3_regression_workflow.py

Workflow (Keim diamond, unrolled):
    ITERATION 1  D1 = all features, raw score -> OLS + boosting -> fit, residuals,
                                                coefficients -> K1 (GPA dominates;
                                                ceiling edge; GPA cap)
    FEEDBACK     T  = drop GPA = 4.0 rows; median regression; GPA as covariate
    ITERATION 2  D2 = censor-aware data       -> median regression, ANCOVA ->
                                                forest plot, tutoring raw vs adjusted -> K2
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a3_common import (load_base, FEATURES, PALETTE, SEED, IMAGES_DIR,  # noqa: E402
                       set_style)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 10)
OUT = ROOT / "Member3" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
set_style()
CAT = PALETTE["cluster"]       # categorical colours, use in this order (never name this C: it
                               # would hide the C() used in statsmodels formulas)
R = {}                          # put every number you quote in the report here


def save(fig, name):
    """Save as images/M3_<name>.png (the report already expects these names)."""
    fig.savefig(IMAGES_DIR / f"M3_{name}.png")
    plt.close(fig)
    print("  saved", f"M3_{name}")


# ==========================================================================
# DATA
# ==========================================================================
train, test = load_base("train"), load_base("test")
X = FEATURES["habits"] + FEATURES["wellbeing"] + FEATURES["background"]
FORMULA = "exam_score ~ previous_gpa + " + " + ".join(X)
print(f"Train {len(train):,}  Test {len(test):,}")

# ==========================================================================
# ITERATION 1 - all features, raw score
# ==========================================================================
print("\nIteration 1")
ols = smf.ols(FORMULA, train).fit(cov_type="HC3")
gbm = HistGradientBoostingRegressor(random_state=SEED).fit(train[X + ["previous_gpa"]],
                                                           train["exam_score"])
pred_ols = ols.predict(test)
pred_gbm = gbm.predict(test[X + ["previous_gpa"]])
for name, p in [("OLS", pred_ols), ("Gradient boosting", pred_gbm)]:
    R[f"iter1_{name}_test_r2"] = float(r2_score(test.exam_score, p))
    R[f"iter1_{name}_test_mae"] = float(mean_absolute_error(test.exam_score, p))
    print(f"  {name:18s} test R2 {R[f'iter1_{name}_test_r2']:.3f}  "
          f"MAE {R[f'iter1_{name}_test_mae']:.2f}")
R["iter1_ols_coef"] = ols.params.round(4).to_dict()
R["iter1_ols_p"] = ols.pvalues.round(4).to_dict()
print("  OLS: prior GPA slope", round(ols.params["previous_gpa"], 2),
      "| tutoring", round(ols.params["access_to_tutoring"], 3),
      "p =", round(ols.pvalues["access_to_tutoring"], 3))

resid, fitted = ols.resid, ols.fittedvalues
R["iter1_share_on_ceiling_edge"] = float(np.isclose(resid, 100 - fitted).mean())
R["iter1_max_ols_prediction"] = float(fitted.max())
print(f"  residuals exactly on the 100 - fitted edge: {R['iter1_share_on_ceiling_edge']:.1%}"
      f" | max OLS prediction {R['iter1_max_ols_prediction']:.1f}")

# TODO fig01: predicted vs actual on test, one panel per model (hexbin, gridsize=50),
#   with the y = x line and test R2 in the title. save(fig, "fig01_iter1_fit")
# TODO fig02: OLS residual vs fitted (train, hexbin). Draw the line
#   residual = 100 - fitted in red: the ceiling edge. save(fig, "fig02_iter1_residuals")
# TODO fig03: OLS coefficients with 95% CI (ols.conf_int()), standardised by each
#   feature's SD so they are comparable; previous_gpa will dominate.
#   save(fig, "fig03_iter1_coefficients")

# ==========================================================================
# FEEDBACK LOOP - transformations
# ==========================================================================
# T1: drop GPA = 4.0 rows (capped predictor, 40% of students)
# T2: median regression instead of OLS (robust to the ~7% left at score 100)
# T3: test tutoring with prior GPA as a covariate (ANCOVA)
tr2 = train[train.gpa_at_ceiling == 0]
R["T1_rows_kept"] = int(len(tr2))
R["T2_share_score100_after_T1"] = float((tr2.exam_score == 100).mean())

# ==========================================================================
# ITERATION 2 - censor-aware models
# ==========================================================================
print("\nIteration 2")
qr = smf.quantreg(FORMULA, tr2).fit(q=0.5)
R["iter2_qr_coef"] = qr.params.round(4).to_dict()
R["iter2_qr_p"] = qr.pvalues.round(4).to_dict()
print("  Median regression (GPA < 4.0): tutoring", round(qr.params["access_to_tutoring"], 3),
      "p =", round(qr.pvalues["access_to_tutoring"], 3))

raw = smf.ols("exam_score ~ C(family_income_range_code) * access_to_tutoring", train).fit()
adj = smf.ols("exam_score ~ previous_gpa + C(family_income_range_code) * access_to_tutoring",
              train).fit()
a_raw, a_adj = sm.stats.anova_lm(raw, typ=2), sm.stats.anova_lm(adj, typ=2)
print("\n  ANOVA, raw score:\n", a_raw.round(4))
print("\n  ANCOVA, prior GPA as covariate:\n", a_adj.round(4))
R["anova_raw_p"] = a_raw["PR(>F)"].dropna().to_dict()
R["ancova_p"] = a_adj["PR(>F)"].dropna().to_dict()
R["gpa_by_tutoring"] = train.groupby("access_to_tutoring").previous_gpa.mean().round(3).to_dict()
print("  mean prior GPA by tutoring:", R["gpa_by_tutoring"])

# TODO fig04: forest plot of median-regression coefficients (qr.conf_int()),
#   standardised as in fig03, previous_gpa shown separately or as a note.
#   save(fig, "fig04_iter2_forest")
# TODO fig05: tutoring gap (tutored minus not) per income band, two markers per
#   band: raw score gap and GPA-adjusted gap (use the residual of
#   exam_score ~ previous_gpa). One axis in points, 95% CIs.
#   save(fig, "fig05_iter2_tutoring")

# ==========================================================================
# SAVE
# ==========================================================================
(OUT / "m3_results.json").write_text(json.dumps(R, indent=2, default=float))
print("\nSaved Member3/outputs/m3_results.json")
