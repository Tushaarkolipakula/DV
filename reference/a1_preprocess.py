"""
DAS732 A1 - Student Habits & Academic Performance Dataset
Preprocessing script (shared foundation for all 3 team members)

What this does:
  1. Loads the raw Kaggle CSV and runs data-quality checks (nulls, dupes, range
     sanity) so the whole team can trust the same cleaned source.
  2. Fixes dtypes: booleans for Yes/No fields, ordered categories for variables
     that have a natural order (diet quality, income, education, internet
     quality) so Tableau/Python sort and color them correctly instead of
     alphabetically.
  3. Engineers a handful of derived fields that make good visualization
     targets (performance bands, age groups, sleep category, leisure-vs-total
     screen ratio, GPA-to-exam delta) — useful for "overview"/"distribution"
     tasks per Schulz et al.
  4. Splits the cleaned dataset into 3 team-member subsets along cohesive
     variable groups (as the assignment asks), each keeping the shared keys
     (student_id, exam_score, previous_gpa) so every member's charts can still
     be tied back to academic performance.
  5. Writes the master cleaned file, the data dictionary and a QA summary to
     the project root, and drops each member subset straight into that
     member's folder, which is where their plotting scripts read from.

Member 2's slice gets a second step: Member2/build_member2_dataset.py adds
dropout_risk, the behaviour-contrast columns and the Tableau bands that
member2_figures.py needs. Full order:

    python preprocess.py
    python Member2/build_member2_dataset.py
    python Member1/generate_visualizations.py
    python Member2/member2_figures.py
    python Member3/generate_9_visualizations.py

Run: python preprocess.py
Requires: pandas
"""

import pandas as pd
import numpy as np
import os

# paths are resolved against this file, so the script works from any cwd
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(ROOT, "enhanced_student_habits_performance_dataset.csv")

# each member subset lands in that member's own folder
MEMBER_DIR = {
    "member1_study_habits.csv": "Member1",
    "member2_wellbeing.csv": "Member2",
    "member3_background.csv": "Member3",
}

qa_log = []  # collects lines for the QA summary written at the end


def log(msg):
    print(msg)
    qa_log.append(msg)


# ---------------------------------------------------------------------------
# 1. LOAD + DATA QUALITY CHECKS
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)

log("=" * 70)
log("DATA QUALITY REPORT")
log("=" * 70)
log(f"Rows x Columns: {df.shape[0]} x {df.shape[1]}")

n_nulls = df.isnull().sum().sum()
log(f"Total missing values: {n_nulls}")

n_dupe_rows = df.duplicated().sum()
n_dupe_ids = df["student_id"].duplicated().sum()
log(f"Fully duplicated rows: {n_dupe_rows}")
log(f"Duplicate student_id values: {n_dupe_ids}")

# Range sanity checks - flag (not silently drop) anything outside plausible bounds
range_checks = {
    "age": (16, 30),
    "attendance_percentage": (0, 100),
    "sleep_hours": (0, 24),
    "previous_gpa": (0, 4),
    "exam_score": (0, 100),
    "stress_level": (0, 10),
    "mental_health_rating": (0, 10),
}
for col, (lo, hi) in range_checks.items():
    bad = df[(df[col] < lo) | (df[col] > hi)]
    if len(bad):
        log(f"  ! {col}: {len(bad)} rows outside [{lo}, {hi}]")
    else:
        log(f"  ok {col}: all values within [{lo}, {hi}]")

# Drop exact duplicate rows / duplicate IDs defensively, even though none were
# found in this snapshot - keeps the script safe to rerun on a refreshed export
before = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset="student_id", keep="first")
if before != len(df):
    log(f"Dropped {before - len(df)} duplicate row(s).")

# Standardize text categoricals (strip whitespace / fix casing) defensively
text_cols = df.select_dtypes(include="object").columns.tolist()
for c in text_cols:
    df[c] = df[c].astype(str).str.strip()

# ---------------------------------------------------------------------------
# 2. DTYPE FIXES
# ---------------------------------------------------------------------------

# Yes/No -> boolean
yes_no_cols = [
    "part_time_job",
    "extracurricular_participation",
    "dropout_risk",
    "access_to_tutoring",
]
for c in yes_no_cols:
    df[c] = df[c].map({"Yes": True, "No": False})

# Ordinal categoricals -> ordered pandas Category (so charts sort logically
# and can be mapped to a numeric scale if a member wants a correlation heatmap)
ordinal_map = {
    "diet_quality": ["Poor", "Fair", "Good"],
    "internet_quality": ["Low", "Medium", "High"],
    "family_income_range": ["Low", "Medium", "High"],
    "parental_education_level": ["High School", "Some College", "Bachelor", "Master", "PhD"],
}
for col, order in ordinal_map.items():
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    df[col + "_code"] = df[col].cat.codes  # numeric version for correlation/regression

# Plain nominal categoricals (no inherent order) -> category dtype for smaller
# file size / cleaner Tableau field typing
nominal_cols = ["gender", "major", "study_environment", "learning_style"]
for c in nominal_cols:
    df[c] = df[c].astype("category")

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

# Performance band - makes exam_score usable in bar/heatmap "overview" tasks
df["performance_band"] = pd.cut(
    df["exam_score"],
    bins=[-1, 59, 74, 89, 100],
    labels=["Low (<60)", "Average (60-74)", "Good (75-89)", "Excellent (90-100)"],
)

# Age group - coarser grouping for demographic comparisons
df["age_group"] = pd.cut(
    df["age"],
    bins=[15, 18, 21, 24, 30],
    labels=["16-18", "19-21", "22-24", "25-28"],
)

# Sleep category, based on commonly cited healthy sleep ranges for young adults
df["sleep_category"] = pd.cut(
    df["sleep_hours"],
    bins=[0, 5.9, 6.9, 9.0, 24],
    labels=["Insufficient (<6h)", "Borderline (6-7h)", "Recommended (7-9h)", "Excessive (>9h)"],
)

# Leisure vs total screen time - social_media + netflix is a subset of the
# broader screen_time field (checked: corr = 0.66, sm+netflix <= screen_time
# for ~99.995% of rows), so this ratio isolates "leisure share of screen time"
df["leisure_screen_hours"] = df["social_media_hours"] + df["netflix_hours"]
df["leisure_screen_share"] = (df["leisure_screen_hours"] / df["screen_time"]).clip(upper=1.0)

# GPA-to-exam delta - previous_gpa (0-4 scale) rescaled to 0-100 and compared
# to the current exam_score, as a simple "improved / declined" signal
df["previous_gpa_pct"] = df["previous_gpa"] / 4.0 * 100
df["performance_delta"] = df["exam_score"] - df["previous_gpa_pct"]

# Total weekly study hours - easier to compare against a 7-day habit cadence
df["study_hours_per_week"] = df["study_hours_per_day"] * 7

log("\nEngineered columns added: performance_band, age_group, sleep_category, "
    "leisure_screen_hours, leisure_screen_share, previous_gpa_pct, "
    "performance_delta, study_hours_per_week, *_code (ordinal encodings)")

# ---------------------------------------------------------------------------
# 4. SAVE MASTER CLEANED FILE
# ---------------------------------------------------------------------------
master_path = os.path.join(ROOT, "master_cleaned.csv")
df.to_csv(master_path, index=False)
log(f"\nSaved master cleaned file -> {master_path}  ({df.shape[0]} rows, {df.shape[1]} cols)")

# ---------------------------------------------------------------------------
# 5. SPLIT INTO 3 TEAM-MEMBER SUBSETS (cohesive variable groups)
# ---------------------------------------------------------------------------
# Shared keys every member keeps, so every subset can still be tied back to
# academic performance without re-merging the full file.
SHARED = ["student_id", "exam_score", "performance_band", "previous_gpa",
          "previous_gpa_pct", "performance_delta", "semester"]

member_groups = {
    # Member 1: study habits & daily routine
    "member1_study_habits.csv": SHARED + [
        "study_hours_per_day", "study_hours_per_week", "study_environment",
        "time_management_score", "sleep_hours", "sleep_category",
        "exercise_frequency", "diet_quality", "diet_quality_code",
        "social_media_hours", "netflix_hours", "screen_time",
        "leisure_screen_hours", "leisure_screen_share",
    ],
    # Member 2: psychological & social wellbeing
    "member2_wellbeing.csv": SHARED + [
        "mental_health_rating", "stress_level", "exam_anxiety_score",
        "motivation_level", "social_activity", "extracurricular_participation",
        "parental_support_level", "learning_style",
    ],
    # Member 3: demographics & socioeconomic / academic background
    "member3_background.csv": SHARED + [
        "age", "age_group", "gender", "major",
        "parental_education_level", "parental_education_level_code",
        "family_income_range", "family_income_range_code",
        "internet_quality", "internet_quality_code",
        "access_to_tutoring", "part_time_job", "attendance_percentage",
        "dropout_risk",
    ],
}

for fname, cols in member_groups.items():
    out_dir = os.path.join(ROOT, MEMBER_DIR[fname])
    os.makedirs(out_dir, exist_ok=True)
    df[cols].to_csv(os.path.join(out_dir, fname), index=False)
    log(f"Saved {MEMBER_DIR[fname]}/{fname} -> {len(cols)} columns")

log("\nMember 2's slice still needs Member2/build_member2_dataset.py, which adds "
    "dropout_risk, the behaviour-contrast columns and the Tableau bands. Run that "
    "before Member2/member2_figures.py.")

# ---------------------------------------------------------------------------
# 6. DATA DICTIONARY (for the report's dataset-description section)
# ---------------------------------------------------------------------------
dictionary_rows = []
descriptions = {
    "student_id": "Unique student identifier",
    "age": "Student age in years",
    "gender": "Male / Female / Other",
    "major": "Declared field of study",
    "study_hours_per_day": "Average hours spent studying per day",
    "study_hours_per_week": "Derived: study_hours_per_day * 7",
    "social_media_hours": "Average daily hours on social media",
    "netflix_hours": "Average daily hours streaming video",
    "leisure_screen_hours": "Derived: social_media_hours + netflix_hours",
    "leisure_screen_share": "Derived: leisure_screen_hours / screen_time (share of total screen time that is leisure)",
    "screen_time": "Total average daily screen time (all uses, incl. leisure)",
    "part_time_job": "Whether the student holds a part-time job (bool)",
    "attendance_percentage": "Class attendance rate (%)",
    "sleep_hours": "Average hours of sleep per night",
    "sleep_category": "Derived bucket from sleep_hours vs. recommended ranges",
    "diet_quality": "Self-reported diet quality (ordered: Poor < Fair < Good)",
    "exercise_frequency": "Exercise sessions per week (0-7)",
    "parental_education_level": "Highest parental education (ordered)",
    "internet_quality": "Home internet quality (ordered: Low < Medium < High)",
    "mental_health_rating": "Self-reported mental health, 1 (poor) - 10 (excellent)",
    "extracurricular_participation": "Participates in extracurriculars (bool)",
    "previous_gpa": "Prior-term GPA, 0-4 scale",
    "previous_gpa_pct": "Derived: previous_gpa rescaled to a 0-100 % for comparison with exam_score",
    "semester": "Current semester number",
    "stress_level": "Self-reported stress, 1-10",
    "dropout_risk": "Flagged as at-risk of dropping out (bool)",
    "social_activity": "Social outings/activities per week (0-5)",
    "screen_time_note": "See screen_time above",
    "study_environment": "Typical study location (Library, Dorm, Cafe, etc.)",
    "access_to_tutoring": "Has access to tutoring support (bool)",
    "family_income_range": "Household income bracket (ordered: Low < Medium < High)",
    "parental_support_level": "Perceived parental support, 1-10",
    "motivation_level": "Self-reported motivation, 1-10",
    "exam_anxiety_score": "Self-reported exam anxiety, 5-10",
    "learning_style": "Preferred learning style (Reading, Visual, Auditory, Kinesthetic)",
    "time_management_score": "Self-assessed time-management skill, 1-10",
    "exam_score": "TARGET: final exam score, 0-100",
    "performance_band": "Derived bucket from exam_score (Low/Average/Good/Excellent)",
    "performance_delta": "Derived: exam_score - previous_gpa_pct (improved vs. declined signal)",
    "age_group": "Derived bucket from age",
    "<col>_code": "Integer encoding of the matching ordered category, for correlation/regression use",
}
dict_df = pd.DataFrame(
    [{"column": k, "description": v} for k, v in descriptions.items()]
)
dict_path = os.path.join(ROOT, "data_dictionary.csv")
dict_df.to_csv(dict_path, index=False)
log(f"Saved data dictionary -> {dict_path}")

# ---------------------------------------------------------------------------
# 7. QA SUMMARY FILE (paste into the report's preprocessing section)
# ---------------------------------------------------------------------------
qa_path = os.path.join(ROOT, "qa_summary.txt")
with open(qa_path, "w") as f:
    f.write("\n".join(qa_log))
log(f"\nSaved QA summary -> {qa_path}")

log("\nDone. Master file, dictionary and QA summary are at the project root; "
    "member subsets are in Member1/, Member2/ and Member3/.")
