# DAS732 A3: Visual Analytics of Student Habits, Wellbeing and Performance

**Team:** Kolipakula Sai Tushaar, Advaya Bhardwaj, Aman Kumar Lahoti

**Deadline:** 29 November 2026, 11:59 pm IST (team target: 27 November) · **Viva/demo:** week of 30 November, all members attend

> **Advaya and Aman: start with [TEAM_INSTRUCTIONS.md](TEAM_INSTRUCTIONS.md).** It lists exactly what is left for each of you, with expected numbers.

For the three verified workflows, their results and reproduction commands, see the [analysis guide](ANALYSIS_GUIDE.md).

![Team workflow](images/A3_master_workflow.png)

## Status

| Part | Owner | Status |
|---|---|---|
| Shared A3 preprocessing (`preprocess_a3.py`, `a3_common.py`) | Tushaar | ✅ done |
| Team workflow diagram, report Sections 2–3 | Tushaar | ✅ shared diagram and sections ready; teammates can update their findings |
| Member 1: pooled and within-GPA clustering, figures, report Section 4 | Tushaar | ✅ done |
| Report skeleton (`report/main.tex`, compiles now, A1 appendix attached) | Tushaar | ✅ done |
| Member 2: classification of dropout risk, Section 5 | Advaya | Models, seven figures and workflow verified. Section 5 is a separate deliverable. |
| Member 3: regression and statistical tests, Section 6 | Aman | Models, five figures and workflow verified. See [Aman’s analysis guide](Member3/README.md). Section 6 is a separate deliverable. |
| Introduction, Synthesis, Conclusions, final PDF | Advaya | ⏳ |
| Video demo and slides | Aman (+ each member's segment) | ⏳ |

## Quick start

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Tushaarkolipakula/DV.git
cd DV
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell,
`.venv\Scripts\activate.bat` in Windows Command Prompt, or
`source .venv/bin/activate` on macOS/Linux. Then install and run:

```bash
python -m pip install -r requirements.txt
python preprocess_a3.py                    # shared data -> data/a3_base.csv.gz (~10 s)
python Member1/m1_habits_by_gpa.py         # both iterations, figures and statistical results
python -m unittest discover -s tests -v    # checks the data rules and Member 1 results
cd report
pdflatex tushaar.tex && pdflatex tushaar.tex  # standalone Member 1 report
pdflatex main.tex && pdflatex main.tex        # team draft, including the A1 appendix
```

Run everything from the repository root (except the LaTeX step). Optional: Graphviz `dot` to re-render diagrams.

## Member 1 result in brief (Tushaar)

Start with the [illustrated report](report/tushaar.pdf). The [captioned video segment](Member1/Member1_habits_demo.mp4) and [viva notes](Member1/M1_viva_and_video.md) cover the same analysis.

Among 16,000 held-out students, the global K-means cluster with higher average study time scored **4.15 points higher**. The clusters also had different prior attainment: 46% of students in the higher-study cluster had a prior GPA of 4.0, compared with 32% in the lower-study cluster.

| Comparison on held-out students | K-means gap in exam points (95% interval) | Gaussian mixture gap (95% interval) |
|---|---:|---:|
| Original global clusters | +4.15 [3.81, 4.52] | +4.19 [3.81, 4.56] |
| Original clusters given the same GPA-band composition | +0.51 [0.35, 0.69] | +0.49 [0.32, 0.65] |
| Models fitted separately inside each GPA band, then averaged by band size | +0.41 [0.24, 0.58] | −0.35 [−0.52, −0.19] |

A common GPA mix accounts for about 88% of the original K-means gap. The last row compares newly formed clusters, ordered by training-set study time within each band. In the lowest band, K-means and the mixture identify different students. The residual association depends on how habits are grouped. [Gap estimates](Member1/outputs/gpa_gaps.csv) · [Score tests and effect sizes](Member1/outputs/gpa_score_tests.csv)

| Assigned task | Where to check it |
|---|---|
| Shared input, redundancy checks, one-hot encoding, training-only scaling, seed-42 split | [Preprocessing log](outputs/a3_preprocessing_report.md) and [feature metadata](data/a3_feature_groups.json) |
| Unrolled workflow with a labelled feedback edge | [Team diagram](diagrams/A3_master_workflow.pdf) and [Tushaar’s diagram](diagrams/M1_workflow_unrolled.pdf) |
| Elbow and silhouette selection across the full cohort and all four bands | [Selection curves](images/M1_gpa_selection.png) |
| PCA maps, mean habit heatmaps and exam-score box plots for both iterations | [Illustrated report, section 4](report/tushaar.pdf) and [figure files](images/) |
| Cluster × GPA-band table and independent refits | [Composition table](Member1/outputs/gpa_mix.csv), [held-out assignments](Member1/outputs/gpa_test_assignments.csv.gz) and [analysis script](Member1/m1_habits_by_gpa.py) |

The earlier four-feature sensitivity analysis remains available in `Member1/m1_clustering_workflow.py`. Its `M1_fig*` images and `m1_results.json` use a different feature set and GPA adjustment. The current report uses the `M1_gpa_*` figures and `gpa_*` result files.

## Repository layout

| Path | Contents |
|---|---|
| `preprocess_a3.py` | Shared preprocessing: validation, redundancy removal, leakage and ceiling flags, GPA bands, one-hot encoding, training-only standardization, stratified 80/20 split (seed 42) |
| `a3_common.py` | `load_base()`, feature groups, shared colours, `set_style()`: **everyone loads data through this** |
| `data/master_cleaned.csv.gz` | A1 output (input to A3), decompresses to the original A1 CSV byte for byte |
| `data/a3_feature_groups.json` | Feature groups, leakage rules, one-hot names |
| `outputs/a3_preprocessing_report.md` | Log of every preprocessing check and count |
| `Member1/` | Tushaar: `m1_habits_by_gpa.py`, `outputs/` (all numbers, tables, cluster labels), viva + video notes |
| `Member2/` | Advaya: classification workflow `m2_classification_workflow.py` |
| `Member3/` | Aman: regression workflow, results and figure guide |
| `images/` | All three members’ analysis figures and workflow diagrams |
| `diagrams/` | Graphviz sources for every workflow diagram (M2/M3 templates ready) |
| `report/` | `main.tex` + `sections/` (one file per section, owners marked at the top of each) |
| `reference/` | A3 brief, A1 report (used as the Appendix), A1 preprocessing script and data dictionary |
| `tests/` | `unittest` checks for the data rules and reproducibility of Member 1's results |
| `TEAM_INSTRUCTIONS.md` | Step-by-step tasks for Advaya and Aman |
