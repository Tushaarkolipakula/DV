# A3 preprocessing report

Loaded `master_cleaned.csv.gz`: 80,000 rows x 43 columns.

## 1. Validation

No nulls, no duplicate rows or IDs, and every numeric column lies inside its plausible range.

## 2. Redundant columns removed

| Column | Exact formula | Holds for all rows |
|---|---|---|
| exam_anxiety_score | clip(15 - motivation_level, 5, 10) | yes |
| study_hours_per_week | 7 x study_hours_per_day | yes |
| leisure_screen_hours | social_media_hours + netflix_hours | yes |

Kept but flagged as derived (never use alongside their source columns, never use target-derived ones as features): `previous_gpa_pct`, `performance_delta`, `performance_band`, `leisure_screen_share`.

## 3. Leakage rules

`dropout_risk` equals `(stress_level > 8) AND (motivation_level <= 3)` for all 80,000 rows. Any classifier of `dropout_risk` that sees stress and motivation is reading the label definition. Positive rate: 1.98%.

## 4. Ceiling flags

- exam_score = 100: 25.8% of students
- previous_gpa = 4.0: 40.0%
- both: 21.4%

Both variables are censored at the top. Linear models and residuals should exclude or separately treat these rows.

## 5. Prior-attainment bands (previous_gpa_pct)

| Band | n | Mean exam score |
|---|---|---|
| Below 80 | 17,487 | 71.7 |
| 80-90 | 15,154 | 85.2 |
| 90-99 | 15,390 | 94.4 |
| Elite (4.0) | 31,969 | 98.0 |

## 6. IQR outlier screen (1.5 x IQR)

| Column | Flagged rows |
|---|---|
| study_hours_per_day | 346 (0.43%) |
| sleep_hours | 284 (0.36%) |
| mental_health_rating | 345 (0.43%) |
| previous_gpa | 201 (0.25%) |
| screen_time | 356 (0.45%) |

Flagged values are inside the instrument ranges checked in step 1, so they are genuine tail values and are kept.

## 7. Encoding

Booleans to 0/1; 4 nominal columns one-hot encoded into 18 `oh_*` columns; ordinal columns keep their A1 `*_code` integers.

## 8. Train/test split

Stratified on performance band x dropout flag, seed 42, test share 20%.

| Split | n | Mean score | Dropout rate | At ceiling |
|---|---|---|---|---|
| test | 16,000 | 89.19 | 1.98% | 26.0% |
| train | 64,000 | 89.13 | 1.98% | 25.7% |
