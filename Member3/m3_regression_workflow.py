"""
Member 3 (Aman Kumar Lahoti) - Who students are: regression and statistical tests.

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
CAT = PALETTE["cluster"]       # categorical colours, use in this order (never name this C)
R = {}                          # every number quoted in the report is stored here

LABEL_OVERRIDES = {
    "previous_gpa": "Previous GPA",
    "family_income_range_code": "Family Income Range",
    "parental_education_level_code": "Parental Education Level",
    "internet_quality_code": "Internet Quality",
    "diet_quality_code": "Diet Quality",
}


def pretty_name(col):
    """Human-readable predictor label for plots."""
    return LABEL_OVERRIDES.get(col, col.replace("_", " ").title())


def save(fig, name):
    """Save as images/M3_<name>.png (the report already expects these names)."""
    fig.savefig(IMAGES_DIR / f"M3_{name}.png")
    plt.close(fig)
    print("  saved", f"M3_{name}.png")


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
gbm = HistGradientBoostingRegressor(random_state=SEED).fit(
    train[X + ["previous_gpa"]], train["exam_score"]
)
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
R["iter1_n_train"] = int(len(train))
R["iter1_n_test"] = int(len(test))
R["iter1_n_predictors"] = int(len(X) + 1)
R["iter1_score100_count_train"] = int((train.exam_score == 100).sum())
R["iter1_share_on_ceiling_edge"] = float(np.isclose(resid, 100 - fitted).mean())
R["iter1_max_ols_prediction"] = float(fitted.max())
ci_ols = ols.conf_int()
R["iter1_ols_tutoring_ci"] = [
    round(float(ci_ols.loc["access_to_tutoring", 0]), 4),
    round(float(ci_ols.loc["access_to_tutoring", 1]), 4),
]
print(f"  residuals exactly on the 100 - fitted edge: {R['iter1_share_on_ceiling_edge']:.1%}"
      f" | max OLS prediction {R['iter1_max_ols_prediction']:.1f}")

# ---- fig01: predicted vs actual on test (hexbin, gridsize=50) ------------
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.7))
for ax, name, p in zip(axs, ["OLS", "Gradient boosting"], [pred_ols, pred_gbm]):
    r2_val = R[f"iter1_{name}_test_r2"]
    mae_val = R[f"iter1_{name}_test_mae"]
    hb = ax.hexbin(
        test.exam_score, p, gridsize=50, cmap="Blues",
        mincnt=1, bins="log", edgecolors="none"
    )
    cb = fig.colorbar(hb, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("Student count (log scale)", fontsize=8.5, color=PALETTE["text_secondary"])
    ax.plot([35, 100], [35, 100], color=PALETTE["text_secondary"],
            ls="--", lw=1.2, label="Ideal: y = x")
    ax.axhline(100, color=PALETTE["diverging"][0], ls=":", lw=1.2, alpha=0.85,
               label="Score ceiling (100 pts)")
    ax.axvline(100, color=PALETTE["diverging"][0], ls=":", lw=1.2, alpha=0.85)
    ax.set(xlim=(35, 105), ylim=(35, 105),
           xlabel="Actual exam score (points, test)",
           ylabel="Predicted exam score (points)",
           title=f"{name}: Test R² = {r2_val:.3f}, MAE = {mae_val:.2f} pts")
    ax.legend(loc="lower right", fontsize=8.5)

fig.suptitle(
    f"Iteration 1: OLS (R² = {R['iter1_OLS_test_r2']:.3f}) and boosting "
    f"(R² = {R['iter1_Gradient boosting_test_r2']:.3f}) have similar held-out accuracy",
    fontweight="bold", x=0.01, y=0.98, ha="left", fontsize=11
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, "fig01_iter1_fit")

# ---- fig02: OLS residual vs fitted (train, hexbin) -------------------------
fig, ax = plt.subplots(figsize=(8.4, 5.0))
hb = ax.hexbin(
    fitted, resid, gridsize=55, cmap="Blues",
    mincnt=1, bins="log", edgecolors="none"
)
cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Student count (log scale)", fontsize=8.5, color=PALETTE["text_secondary"])
ax.axhline(0, color=PALETTE["text_secondary"], ls="--", lw=1.0)

# Ceiling edge line: residual = 100 - fitted (clipped cleanly by ylim)
f_grid = np.linspace(fitted.min(), fitted.max(), 200)
ax.plot(f_grid, 100 - f_grid, color=PALETTE["diverging"][0], lw=2.0,
        label="Ceiling edge: residual = 100 − fitted")

ax.annotate(
    f"{R['iter1_share_on_ceiling_edge']:.1%} of students sit exactly on this edge\n"
    f"(all students scoring 100; max OLS ŷ = {R['iter1_max_ols_prediction']:.1f})",
    xy=(88.5, 11.5), xytext=(45.5, 15.5),
    arrowprops=dict(arrowstyle="->", color=PALETTE["diverging"][0], lw=1.5),
    fontsize=8.8, weight="bold", color=PALETTE["text"],
    bbox=dict(boxstyle="round,pad=0.35", facecolor=PALETTE["surface"],
              edgecolor=PALETTE["diverging"][0], alpha=0.95)
)

ax.set(
    xlim=(fitted.min() - 1, 101.5),
    ylim=(-25, 25),
    xlabel="Fitted exam score, $\\hat{y}$ (points)",
    ylabel="OLS residual, $y - \\hat{y}$ (points)",
)
ax.set_title(
    f"Iteration 1: {R['iter1_share_on_ceiling_edge']:.1%} of OLS residuals sit on the 100 − fitted ceiling edge\n"
    f"(training split, n = {len(train):,})",
    fontsize=10.8, pad=10
)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
save(fig, "fig02_iter1_residuals")

# ---- fig03: OLS coefficients standardised by feature SD ------------------
terms_ols = [t for t in ols.params.index if t not in ["Intercept", "const"]]
std_s = train[terms_ols].std()

coef_std = ols.params[terms_ols] * std_s
ci_lo_std = ci_ols.loc[terms_ols, 0] * std_s
ci_hi_std = ci_ols.loc[terms_ols, 1] * std_s

df_coef = pd.DataFrame({
    "term": terms_ols,
    "coef_std": coef_std,
    "ci_lo": ci_lo_std,
    "ci_hi": ci_hi_std,
}).sort_values("coef_std", ascending=True)

gpa_slope = float(ols.params["previous_gpa"])
gpa_std_val = float(df_coef.loc[df_coef.term == "previous_gpa", "coef_std"].iloc[0])
other_terms = [t for t in terms_ols if t != "previous_gpa"]
max_other_abs = float(df_coef.loc[df_coef.term != "previous_gpa", "coef_std"].abs().max())
min_other_p = float(ols.pvalues[other_terms].min())

R["iter1_gpa_std_coef"] = round(gpa_std_val, 4)
R["iter1_max_non_gpa_abs_std_coef"] = round(max_other_abs, 4)
R["iter1_min_non_gpa_p"] = round(min_other_p, 4)

df_coef.to_csv(OUT / "ols_standardized_intervals.csv", index=False)
fig, ax = plt.subplots(figsize=(9.4, 6.4))
y_pos = np.arange(len(df_coef))

ax.axvline(0, color=PALETTE["text_secondary"], ls="--", lw=1.0)
for i, (_, row) in enumerate(df_coef.iterrows()):
    col = PALETTE["adjusted"] if row.term == "previous_gpa" else PALETTE["raw"]
    ax.errorbar(
        row.coef_std, i,
        xerr=[[row.coef_std - row.ci_lo], [row.ci_hi - row.coef_std]],
        fmt="o", ms=6, color=col, capsize=3, lw=1.6
    )

ax.set_yticks(y_pos)
ax.set_yticklabels([pretty_name(t) for t in df_coef["term"]], fontsize=8.8)
ax.set(
    xlabel="OLS coefficient × feature SD (exam-score points per 1 SD change, 95% HC3 CI)",
    title=(
        f"Iteration 1: Prior GPA (+{gpa_std_val:.1f} pts/SD) dominates OLS; "
        f"all {len(other_terms)} other features sit near zero"
    ),
)

ax.text(
    gpa_std_val - 0.25, len(df_coef) - 1.7,
    f"Prior GPA: {gpa_slope:+.2f} pts/unit ({gpa_std_val:+.1f} pts/SD)\n"
    f"All other {len(other_terms)} features: |β×SD| ≤ {max_other_abs:.3f}, "
    f"min p = {min_other_p:.3f}",
    va="top", ha="right", fontsize=8.5, weight="bold", color=PALETTE["text"],
    bbox=dict(boxstyle="round,pad=0.35", facecolor=PALETTE["surface"],
              edgecolor=PALETTE["adjusted"])
)

fig.tight_layout()
save(fig, "fig03_iter1_coefficients")

# ==========================================================================
# FEEDBACK LOOP - transformations
# ==========================================================================
# T1: drop GPA = 4.0 rows (capped predictor, 40% of students)
# T2: median regression instead of OLS (robust to the ~7% left at score 100)
# T3: test tutoring with prior GPA as a covariate (ANCOVA)
tr2 = train[train.gpa_at_ceiling == 0]
R["T1_rows_dropped"] = int((train.gpa_at_ceiling == 1).sum())
R["T1_gpa4_share_train"] = float((train.gpa_at_ceiling == 1).mean())
R["T1_rows_kept"] = int(len(tr2))
R["T2_share_score100_after_T1"] = float((tr2.exam_score == 100).mean())

# ==========================================================================
# ITERATION 2 - censor-aware models
# ==========================================================================
print("\nIteration 2")
qr = smf.quantreg(FORMULA, tr2).fit(q=0.5)
ci_qr = qr.conf_int()
R["iter2_qr_coef"] = qr.params.round(4).to_dict()
R["iter2_qr_p"] = qr.pvalues.round(4).to_dict()
R["iter2_qr_gpa_ci"] = [
    round(float(ci_qr.loc["previous_gpa", 0]), 4),
    round(float(ci_qr.loc["previous_gpa", 1]), 4),
]
R["iter2_qr_tutoring_ci"] = [
    round(float(ci_qr.loc["access_to_tutoring", 0]), 4),
    round(float(ci_qr.loc["access_to_tutoring", 1]), 4),
]
print("  Median regression (GPA < 4.0): tutoring", round(qr.params["access_to_tutoring"], 3),
      "p =", round(qr.pvalues["access_to_tutoring"], 3))

raw = smf.ols("exam_score ~ C(family_income_range_code) * access_to_tutoring", train).fit()
adj = smf.ols(
    "exam_score ~ previous_gpa + C(family_income_range_code) * access_to_tutoring", train
).fit()
adj_add = smf.ols(
    "exam_score ~ previous_gpa + C(family_income_range_code) + access_to_tutoring", train
).fit()

a_raw, a_adj = sm.stats.anova_lm(raw, typ=2), sm.stats.anova_lm(adj, typ=2)
print("\n  ANOVA, raw score:\n", a_raw.round(4))
print("\n  ANCOVA, prior GPA as covariate:\n", a_adj.round(4))
print("  ANCOVA (additive main effect) tutoring coef:",
      round(adj_add.params["access_to_tutoring"], 4),
      "p =", round(adj_add.pvalues["access_to_tutoring"], 4))

R["anova_raw_F"] = a_raw["F"].dropna().round(4).to_dict()
R["anova_raw_p"] = a_raw["PR(>F)"].dropna().to_dict()
R["ancova_F"] = a_adj["F"].dropna().round(4).to_dict()
R["ancova_p"] = a_adj["PR(>F)"].dropna().to_dict()
R["ancova_additive_tutoring_coef"] = float(round(adj_add.params["access_to_tutoring"], 4))
R["ancova_additive_tutoring_p"] = float(adj_add.pvalues["access_to_tutoring"])
R["gpa_by_tutoring"] = train.groupby("access_to_tutoring").previous_gpa.mean().round(3).to_dict()
print("  mean prior GPA by tutoring:", R["gpa_by_tutoring"])

# ---- fig04: forest plot of median-regression coefficients (qr.conf_int()) -
terms_qr = [t for t in qr.params.index if t not in ["Intercept", "const", "previous_gpa"]]
std_tr2 = tr2[terms_qr].std()

coef_qr_std = qr.params[terms_qr] * std_tr2
ci_qr_lo = ci_qr.loc[terms_qr, 0] * std_tr2
ci_qr_hi = ci_qr.loc[terms_qr, 1] * std_tr2
R["iter2_max_non_gpa_abs_std_coef"] = round(float(coef_qr_std.abs().max()), 4)
R["iter2_min_non_gpa_p"] = round(float(qr.pvalues[terms_qr].min()), 4)

df_qr = pd.DataFrame({
    "term": terms_qr,
    "coef_std": coef_qr_std,
    "ci_lo": ci_qr_lo,
    "ci_hi": ci_qr_hi,
}).sort_values("coef_std", ascending=True)

df_qr.to_csv(OUT / "median_non_gpa_intervals.csv", index=False)
fig, ax = plt.subplots(figsize=(9.6, 6.0))
y_pos_qr = np.arange(len(df_qr))

ax.axvline(0, color=PALETTE["text_secondary"], ls="--", lw=1.0)
for i, (_, row) in enumerate(df_qr.iterrows()):
    ax.errorbar(
        row.coef_std, i,
        xerr=[[row.coef_std - row.ci_lo], [row.ci_hi - row.coef_std]],
        fmt="o", ms=6, color=PALETTE["adjusted"], capsize=3, lw=1.6
    )

# Dynamic x-limits so bottom-right callout box never overlaps any error bar
x_left = min(-0.11, float(df_qr["ci_lo"].min()) - 0.015)
bottom_max_hi = float(df_qr.iloc[:5]["ci_hi"].max())
x_right = max(0.185, bottom_max_hi + 0.145)

ax.set_yticks(y_pos_qr)
ax.set_yticklabels([pretty_name(t) for t in df_qr["term"]], fontsize=8.8)
ax.set(
    xlabel="Median-regression coefficient × feature SD (exam-score points per 1 SD change, 95% CI)",
    xlim=(x_left, x_right),
)
ax.set_title(
    f"Iteration 2: Median regression on GPA < 4.0\n"
    f"(n = {len(tr2):,}; all {len(terms_qr)} non-GPA 95% CIs cross zero)",
    fontsize=10.8, pad=8
)

gpa_qr_val = float(qr.params["previous_gpa"])
ax.text(
    0.985, 0.04,
    f"Prior GPA slope: {gpa_qr_val:+.2f} pts/unit (p < 1e-100)\n"
    f"Tutoring: {qr.params['access_to_tutoring']:+.3f} pts (p = {qr.pvalues['access_to_tutoring']:.2f})\n"
    f"All {len(terms_qr)} non-GPA 95% CIs overlap zero",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.38", facecolor=PALETTE["surface"],
              edgecolor=PALETTE["adjusted"])
)

fig.tight_layout()
save(fig, "fig04_iter2_forest")

# ---- fig05: tutoring gap per income band (raw vs GPA-adjusted) -----------
train_copy = train.copy()
gpa_only_ols = smf.ols("exam_score ~ previous_gpa", train_copy).fit()
baseline = gpa_only_ols.predict(test)
R["gpa_only_test_r2"] = float(r2_score(test.exam_score, baseline))
R["gpa_only_test_mae"] = float(mean_absolute_error(test.exam_score, baseline))
train_copy["gpa_adj_score"] = train_copy["exam_score"] - gpa_only_ols.predict(train_copy)

tut_all = train_copy[train_copy.access_to_tutoring == 1]
notut_all = train_copy[train_copy.access_to_tutoring == 0]
overall_raw_gap = float(tut_all.exam_score.mean() - notut_all.exam_score.mean())
overall_adj_gap = float(tut_all.gpa_adj_score.mean() - notut_all.gpa_adj_score.mean())
R["tutoring_overall_raw_gap"] = round(overall_raw_gap, 4)
R["tutoring_overall_gpa_adj_gap"] = round(overall_adj_gap, 4)

income_labels = {0: "Low Income", 1: "Medium Income", 2: "High Income"}
income_order = [0, 1, 2]

gap_data = []
for code in income_order:
    sub = train_copy[train_copy.family_income_range_code == code]
    tut = sub[sub.access_to_tutoring == 1]
    notut = sub[sub.access_to_tutoring == 0]

    raw_diff = float(tut.exam_score.mean() - notut.exam_score.mean())
    raw_se = float(np.sqrt(tut.exam_score.var(ddof=1) / len(tut) +
                           notut.exam_score.var(ddof=1) / len(notut)))
    raw_ci = 1.96 * raw_se

    adj_diff = float(tut.gpa_adj_score.mean() - notut.gpa_adj_score.mean())
    adj_se = float(np.sqrt(tut.gpa_adj_score.var(ddof=1) / len(tut) +
                           notut.gpa_adj_score.var(ddof=1) / len(notut)))
    adj_ci = 1.96 * adj_se

    gap_data.append({
        "income": income_labels.get(code, str(code)),
        "n": int(len(sub)),
        "raw_diff": round(raw_diff, 4),
        "raw_ci": round(raw_ci, 4),
        "adj_diff": round(adj_diff, 4),
        "adj_ci": round(adj_ci, 4),
    })

df_gap = pd.DataFrame(gap_data)
R["tutoring_gap_by_income"] = gap_data

fig, ax = plt.subplots(figsize=(9.4, 4.3))
y_vals = np.arange(len(df_gap))[::-1]

for y_i, (_, r) in zip(y_vals, df_gap.iterrows()):
    ax.errorbar(
        r.raw_diff, y_i + 0.13, xerr=r.raw_ci, fmt="o", ms=8,
        color=PALETTE["raw"], capsize=3, lw=2,
        label="Raw score gap" if y_i == y_vals[0] else None
    )
    ax.errorbar(
        r.adj_diff, y_i - 0.13, xerr=r.adj_ci, fmt="o", ms=8,
        color=PALETTE["adjusted"], capsize=3, lw=2,
        label="GPA-adjusted gap" if y_i == y_vals[0] else None
    )

    ax.text(r.raw_diff + r.raw_ci + 0.12, y_i + 0.13, f"raw {r.raw_diff:+.2f} pts",
            va="center", fontsize=8.8, color=PALETTE["text_secondary"])
    ax.text(r.adj_diff + r.adj_ci + 0.12, y_i - 0.13, f"adj {r.adj_diff:+.2f} pts",
            va="center", fontsize=8.8, color=PALETTE["adjusted"], weight="bold")

ax.axvline(0, color=PALETTE["text_secondary"], ls="--", lw=1.0)
ax.set_yticks(y_vals)
ax.set_yticklabels([f"{r.income}\n(n = {r.n:,})" for _, r in df_gap.iterrows()], fontsize=9)
ax.set(
    xlim=(-0.6, 4.35),
    ylim=(-0.55, 2.55),
    xlabel="Tutoring gap: Tutored minus Not Tutored (exam-score points, 95% CI)",
)
ax.set_title(
    f"Iteration 2: Tutoring gap falls from {overall_raw_gap:+.2f} to {overall_adj_gap:+.2f} points after GPA adjustment",
    fontsize=10.8,
    pad=28,
)
# Horizontal legend cleanly above the axes (zero overlap with data labels)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=9)
fig.tight_layout()
save(fig, "fig05_iter2_tutoring")

# ==========================================================================
# SAVE
# ==========================================================================
(OUT / "m3_results.json").write_text(json.dumps(R, indent=2, default=float))
print("\nSaved Member3/outputs/m3_results.json")
