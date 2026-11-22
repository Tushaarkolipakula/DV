"""
preprocess_a3.py  -  shared A3 preprocessing (runs after the A1 pipeline).

Input : data/master_cleaned.csv.gz  (A1 output, 80,000 x 43)
Output: data/a3_base.csv.gz         (one row per student, with 'split' column)
        data/a3_feature_groups.json (feature groups, leakage notes, encodings)
        outputs/a3_preprocessing_report.md

Steps added on top of A1 preprocessing
  1. Re-validate the A1 file (schema, nulls, duplicates, value ranges).
  2. Prove and remove exact redundancies (columns that are formulas of others).
  3. Record target-leakage rules found in A1 so no model uses them by mistake.
  4. Flag ceiling effects (exam score = 100, GPA = 4.0) for censoring-aware work.
  5. Add prior-attainment band (gpa_band) used for stratified analysis.
  6. IQR outlier screen (flag and report, not drop: all values are in range).
  7. Encode: booleans -> 0/1, nominal -> one-hot (oh_*), ordinals keep A1 codes.
  8. Stratified 80/20 train/test split (seed 42) on performance band x dropout.

Scaling is NOT applied here: each member fits the scaler on the training
split inside their own pipeline so test data never leaks into it.

Usage:  python preprocess_a3.py   (from the repository root)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from a3_common import (RAW_FILE, BASE_FILE, META_FILE, ROOT, SEED, TEST_SIZE,
                       GPA_BAND_EDGES, GPA_BAND_LABELS, NOMINAL, FEATURES)

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
log = []


def note(msg):
    print(msg)
    log.append(msg)


# --------------------------------------------------------------------------
# 1. Load and re-validate
# --------------------------------------------------------------------------
df = pd.read_csv(RAW_FILE)
note(f"# A3 preprocessing report\n\nLoaded `{RAW_FILE.name}`: "
     f"{df.shape[0]:,} rows x {df.shape[1]} columns.\n")

assert df.isna().sum().sum() == 0, "unexpected nulls"
assert not df.duplicated().any(), "duplicate rows"
assert df["student_id"].is_unique, "duplicate student ids"

RANGES = {
    "age": (16, 28), "study_hours_per_day": (0, 12), "sleep_hours": (4, 12),
    "attendance_percentage": (40, 100), "exercise_frequency": (0, 7),
    "social_media_hours": (0, 5), "netflix_hours": (0, 4),
    "previous_gpa": (0, 4), "exam_score": (0, 100), "stress_level": (1, 10),
    "motivation_level": (1, 10), "mental_health_rating": (1, 10),
    "semester": (1, 8), "screen_time": (0, 24),
}
bad = {c: int(((df[c] < lo) | (df[c] > hi)).sum()) for c, (lo, hi) in RANGES.items()}
assert sum(bad.values()) == 0, f"out-of-range values: {bad}"
note("## 1. Validation\n\nNo nulls, no duplicate rows or IDs, and every numeric "
     "column lies inside its plausible range.\n")

# --------------------------------------------------------------------------
# 2. Exact redundancies: verify the formula, then drop the copy
# --------------------------------------------------------------------------
redundant = {
    "exam_anxiety_score": (df["exam_anxiety_score"] ==
                           np.clip(15 - df["motivation_level"], 5, 10)).all(),
    "study_hours_per_week": np.allclose(df["study_hours_per_week"],
                                        7 * df["study_hours_per_day"]),
    "leisure_screen_hours": np.allclose(df["leisure_screen_hours"],
                                        df["social_media_hours"] + df["netflix_hours"]),
}
formulas = {
    "exam_anxiety_score": "clip(15 - motivation_level, 5, 10)",
    "study_hours_per_week": "7 x study_hours_per_day",
    "leisure_screen_hours": "social_media_hours + netflix_hours",
}
note("## 2. Redundant columns removed\n\n| Column | Exact formula | Holds for all rows |\n|---|---|---|")
for c, ok in redundant.items():
    assert ok, f"{c} formula does not hold"
    note(f"| {c} | {formulas[c]} | yes |")
df = df.drop(columns=list(redundant))

# Derived columns kept for plotting but flagged as collinear/derived.
derived = {
    "previous_gpa_pct": "previous_gpa / 4 x 100",
    "performance_delta": "exam_score - previous_gpa_pct (target-derived)",
    "performance_band": "binned exam_score (target-derived)",
    "leisure_screen_share": "(social_media + netflix) / screen_time",
}
assert np.allclose(df["previous_gpa_pct"], df["previous_gpa"] / 4 * 100)
assert np.allclose(df["performance_delta"], df["exam_score"] - df["previous_gpa_pct"])
note("\nKept but flagged as derived (never use alongside their source columns, "
     "never use target-derived ones as features): " +
     ", ".join(f"`{k}`" for k in derived) + ".\n")

# --------------------------------------------------------------------------
# 3. Leakage rules found in A1
# --------------------------------------------------------------------------
rule = (df["stress_level"] > 8) & (df["motivation_level"] <= 3)
assert (rule == df["dropout_risk"]).all()
note("## 3. Leakage rules\n\n`dropout_risk` equals `(stress_level > 8) AND "
     "(motivation_level <= 3)` for all 80,000 rows. Any classifier of "
     "`dropout_risk` that sees stress and motivation is reading the label "
     f"definition. Positive rate: {df['dropout_risk'].mean():.2%}.\n")

# --------------------------------------------------------------------------
# 4. Ceiling flags
# --------------------------------------------------------------------------
df["score_at_ceiling"] = (df["exam_score"] == 100).astype(int)
df["gpa_at_ceiling"] = (df["previous_gpa"] == 4.0).astype(int)
df["both_at_ceiling"] = df["score_at_ceiling"] * df["gpa_at_ceiling"]
note("## 4. Ceiling flags\n\n"
     f"- exam_score = 100: {df['score_at_ceiling'].mean():.1%} of students\n"
     f"- previous_gpa = 4.0: {df['gpa_at_ceiling'].mean():.1%}\n"
     f"- both: {df['both_at_ceiling'].mean():.1%}\n\n"
     "Both variables are censored at the top. Linear models and residuals "
     "should exclude or separately treat these rows.\n")

# --------------------------------------------------------------------------
# 5. Prior-attainment band
# --------------------------------------------------------------------------
df["gpa_band"] = pd.cut(df["previous_gpa_pct"], GPA_BAND_EDGES,
                        labels=GPA_BAND_LABELS, include_lowest=True)
tab = df.groupby("gpa_band", observed=True)["exam_score"].agg(["size", "mean"])
note("## 5. Prior-attainment bands (previous_gpa_pct)\n\n| Band | n | Mean exam score |\n|---|---|---|")
for b, r in tab.iterrows():
    note(f"| {b} | {int(r['size']):,} | {r['mean']:.1f} |")

# --------------------------------------------------------------------------
# 6. IQR outlier screen
# --------------------------------------------------------------------------
note("\n## 6. IQR outlier screen (1.5 x IQR)\n\n| Column | Flagged rows |\n|---|---|")
num_cols = FEATURES["habits"] + FEATURES["wellbeing"] + ["previous_gpa", "screen_time"]
for c in num_cols:
    q1, q3 = df[c].quantile([.25, .75])
    k = int(((df[c] < q1 - 1.5 * (q3 - q1)) | (df[c] > q3 + 1.5 * (q3 - q1))).sum())
    if k:
        note(f"| {c} | {k:,} ({k / len(df):.2%}) |")
note("\nFlagged values are inside the instrument ranges checked in step 1, "
     "so they are genuine tail values and are kept.\n")

# --------------------------------------------------------------------------
# 7. Encoding
# --------------------------------------------------------------------------
for c in ["part_time_job", "extracurricular_participation",
          "access_to_tutoring", "dropout_risk"]:
    df[c] = df[c].astype(int)
onehot = pd.get_dummies(df[NOMINAL], prefix=[f"oh_{c}" for c in NOMINAL],
                        dtype=int)
df = pd.concat([df, onehot], axis=1)
note(f"## 7. Encoding\n\nBooleans to 0/1; {len(NOMINAL)} nominal columns "
     f"one-hot encoded into {onehot.shape[1]} `oh_*` columns; ordinal columns "
     "keep their A1 `*_code` integers.\n")

# --------------------------------------------------------------------------
# 8. Stratified split
# --------------------------------------------------------------------------
strata = df["performance_band"].astype(str) + "|" + df["dropout_risk"].astype(str)
train_idx, test_idx = train_test_split(df.index, test_size=TEST_SIZE,
                                       random_state=SEED, stratify=strata)
df["split"] = "train"
df.loc[test_idx, "split"] = "test"
chk = df.groupby("split")[["exam_score", "dropout_risk", "score_at_ceiling"]].mean()
note("## 8. Train/test split\n\nStratified on performance band x dropout flag, "
     f"seed {SEED}, test share {TEST_SIZE:.0%}.\n\n"
     "| Split | n | Mean score | Dropout rate | At ceiling |\n|---|---|---|---|---|")
for s, r in chk.iterrows():
    note(f"| {s} | {(df['split'] == s).sum():,} | {r['exam_score']:.2f} | "
         f"{r['dropout_risk']:.2%} | {r['score_at_ceiling']:.1%} |")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
df.to_csv(BASE_FILE, index=False)
meta = {
    "feature_groups": FEATURES,
    "one_hot_columns": list(onehot.columns),
    "removed_redundant": formulas,
    "derived_flagged": derived,
    "leakage": {
        "dropout_risk": "(stress_level > 8) & (motivation_level <= 3)",
        "exam_score": ["performance_band", "performance_delta"],
    },
    "ceiling_flags": ["score_at_ceiling", "gpa_at_ceiling", "both_at_ceiling"],
    "gpa_band": {"edges_pct": GPA_BAND_EDGES, "labels": GPA_BAND_LABELS},
    "seed": SEED, "test_size": TEST_SIZE,
}
with open(META_FILE, "w") as f:
    json.dump(meta, f, indent=2)
(OUT_DIR / "a3_preprocessing_report.md").write_text("\n".join(log) + "\n")
print(f"\nSaved {BASE_FILE.name} {df.shape}, {META_FILE.name}, "
      "outputs/a3_preprocessing_report.md")
