"""
a3_common.py  -  shared constants and loaders for DAS732 A3.

Every member imports from here so that all three analyses use the same
feature groups, the same train/test split and the same colours.

    from a3_common import load_base, FEATURES, PALETTE, SEED
"""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT / "images"
RAW_FILE = DATA_DIR / "master_cleaned.csv.gz"       # A1 output, gzip of the exact CSV
BASE_FILE = DATA_DIR / "a3_base.csv.gz"             # output of preprocess_a3.py
META_FILE = DATA_DIR / "a3_feature_groups.json"

SEED = 42
TEST_SIZE = 0.20

# Prior-attainment bands, defined on previous_gpa_pct (= previous_gpa / 4 * 100).
# 'Elite' is exactly GPA 4.0, which is 40% of the cohort and sits at the GPA cap.
GPA_BAND_EDGES = [0, 80, 90, 99.999, 100]
GPA_BAND_LABELS = ["Below 80", "80-90", "90-99", "Elite (4.0)"]

# Nominal columns that get one-hot encoded.
NOMINAL = ["gender", "major", "study_environment", "learning_style"]

# Member 1's specified numeric habits accompany the study-environment indicators.
# FEATURES["habits"] below remains the input list used by the earlier analyses.
HABIT_NUMERIC = [
    "study_hours_per_day", "sleep_hours", "exercise_frequency",
    "leisure_screen_share", "attendance_percentage",
]

# Feature groups (raw, non-derived columns only; see preprocess_a3.py for why).
FEATURES = {
    # What students do (Member 1)
    "habits": [
        "study_hours_per_day", "sleep_hours", "exercise_frequency",
        "attendance_percentage", "social_media_hours", "netflix_hours",
    ],
    # What students feel (Member 2)
    "wellbeing": [
        "stress_level", "motivation_level", "mental_health_rating",
        "parental_support_level", "social_activity", "time_management_score",
    ],
    # Who students are (Member 3)
    "background": [
        "age", "semester", "family_income_range_code",
        "parental_education_level_code", "internet_quality_code",
        "diet_quality_code", "access_to_tutoring", "part_time_job",
        "extracurricular_participation",
    ],
    "prior": ["previous_gpa"],
    "targets": ["exam_score", "dropout_risk", "performance_band"],
}

# Colours shared by every member's figures. Categorical order is fixed and was
# checked with a colour-blind validator: use slots in order, never cycle.
# Scatter plots with more than 3 groups should be faceted instead of coloured.
PALETTE = {
    "cluster": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#6250d6", "#e34948"],
    # ordinal blue ramp (light -> dark = low -> high prior attainment)
    "gpa_band": {"Below 80": "#86b6ef", "80-90": "#3987e5",
                 "90-99": "#1c5cab", "Elite (4.0)": "#0d366b"},
    # diverging blue <-> red with neutral grey midpoint
    "diverging": ["#e34948", "#f0efec", "#2a78d6"],
    "raw": "#8a8985",
    "adjusted": "#2a78d6",
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "grid": "#e4e3df",
}


def set_style():
    """Matplotlib defaults used by every A3 figure."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": PALETTE["surface"], "axes.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"], "savefig.dpi": 200,
        "savefig.bbox": "tight", "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelcolor": PALETTE["text_secondary"],
        "axes.edgecolor": "#b9b8b3", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": PALETTE["grid"], "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": PALETTE["text_secondary"],
        "ytick.color": PALETTE["text_secondary"], "legend.frameon": False,
        "lines.linewidth": 2,
    })


def load_base(split=None):
    """Load the A3 base table. split = None | 'train' | 'test'."""
    if not BASE_FILE.exists():
        raise FileNotFoundError(
            f"{BASE_FILE} not found. Run `python preprocess_a3.py` first.")
    df = pd.read_csv(BASE_FILE)
    df["gpa_band"] = pd.Categorical(df["gpa_band"], GPA_BAND_LABELS, ordered=True)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def load_meta():
    with open(META_FILE) as f:
        return json.load(f)
