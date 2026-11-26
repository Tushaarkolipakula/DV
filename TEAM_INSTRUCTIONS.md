# Team instructions: what Advaya and Aman need to do

**Deadline:** Sunday 29 November 2026, 11:59 pm IST (hard, no late submissions). **Aim to submit on Friday 27 November.**
**Viva/demo:** week of 30 November. **Everyone must attend**: an absent member gets 0 for A3, and marks are given per person based on their contribution and viva answers.

This file is the current team plan. Clustering is Tushaar's strand; Advaya owns classification and Aman owns regression. Follow these assignments instead of any earlier plan.

## Who does what

| Person | Strand (method family the brief asks for) | Report sections | Other |
|---|---|---|---|
| **Tushaar** | Habits: unsupervised clustering. **DONE** | 2, 3, 4 **DONE** | records his own video segment |
| **Advaya** | Wellbeing: supervised **classification** of `dropout_risk` | 5 (yours), plus 1 Introduction, 7 Synthesis, 8 Conclusions | **assembles and compiles the final report** |
| **Aman** | Background: supervised **regression** + statistical tests on `exam_score` | 6 (yours) | **makes the video and the slides** |

Advaya’s classification starter and figure slots are ready. Aman’s regression models, five figures and workflow diagram are complete. See [his analysis guide](Member3/README.md) for the verified outputs.

---

## Part A: setup (both, about 10 minutes)

```bash
git clone https://github.com/Tushaarkolipakula/DV.git
cd DV
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python preprocess_a3.py                      # about 10 s, creates data/a3_base.csv.gz
```

Then run your starter script:

- Advaya: `python Member2/m2_classification_workflow.py` (about 20 s)
- Aman: `python Member3/m3_regression_workflow.py` (about 5 s)

It prints your iteration-1 and iteration-2 results. Your numbers should match the "expected numbers" below. If they don't, tell Tushaar before going further.

**Rules (so the three strands fit together):**

1. Load data only through `load_base("train")` / `load_base("test")` from `a3_common.py`. Never make your own split.
2. Fit scalers and models on **train** only. Report metrics on **test**.
3. Save figures with the `save()` function already in your script. It writes `images/M2_...png` or `images/M3_...png` with the exact names the report expects.
4. Call `set_style()` (already done) and use `CAT` (the categorical colours) in order: `CAT[0]`, `CAT[1]`, and so on. For "before vs after" charts use `PALETTE["raw"]` (grey) for before and `PALETTE["adjusted"]` (blue) for after, like Tushaar's figures.
5. Every chart: a title that states the finding, labelled axes with units, one y-axis only (never a twin axis), and a legend when there are two or more series.
6. Put every number you quote in the report into the `R` dictionary. The script saves it to `MemberX/outputs/mX_results.json`, so the report can be checked against it.
7. Edit only your own files (see Part E), so nobody gets merge conflicts.

Look at Tushaar's figures in `images/M1_*.png` and his current analysis in `Member1/m1_habits_by_gpa.py` for working examples of every chart type you need (heatmap, PR-style curves, bar charts, dot plots with confidence intervals).

---

## Part B: Advaya (classification of dropout risk)

### B1. Make the 6 figures

Every `# TODO fig..` comment in `Member2/m2_classification_workflow.py` says exactly what to draw. Summary:

| File (auto-named by `save`) | What it shows | Chart |
|---|---|---|
| `M2_fig01_rule_heatmap` | dropout rate per stress (rounded) × motivation cell, train split | heatmap, sequential blue, annotated, rule box drawn |
| `M2_fig02_iter1_pr_roc` | PR (left) and ROC (right) for the 3 models, test split | line curves, AUC in legend, base-rate line on PR |
| `M2_fig03_iter1_tree` | depth-2 tree (left) + top-10 random-forest importances (right) | `plot_tree` + horizontal bars |
| `M2_fig04_iter2_pr_before_after` | PR of the best model, iteration 1 vs 2, base-rate line | 2 lines on one axis (grey = before, blue = after) |
| `M2_fig05_iter2_importance` | permutation importance, leak-free logistic model | horizontal bars, top 10 |
| `M2_fig06_proxy_check` | dropout rate by `gpa_band`, and by rounded `mental_health_rating` | two bar/dot panels |

### B2. Expected numbers (from Tushaar's check; yours should match)

| | Logistic | Tree | Forest |
|---|---|---|---|
| Iteration 1 PR-AUC (test) | 0.880 | 1.000 | 1.000 |
| Iteration 2 PR-AUC (test) | 0.075 | 0.043 | 0.065 |
| Iteration 2 ROC-AUC (test) | 0.701 | 0.693 | 0.697 |

- Base rate is 1.98%.
- The depth-2 tree splits on `stress_level <= 8.05`, then `motivation_level <= 3.50`.
- Top leak-free features: `previous_gpa` (0.128), then `mental_health_rating` (0.057), then everything else at 0.01 or below.
- Prior GPA correlates 0.27 with motivation. Mental health drops at stress 7 (A1 Fig 23).

**What it means (your K1 and K2):**

- **K1:** the label is a definition, so the models read it back. That is label leakage. Logistic regression is "only" 0.88 because a straight line can't draw the AND-shaped rule.
- **K2:** without the rule's inputs, the flag is barely predictable, and what remains comes from proxies of motivation (prior GPA) and stress (mental health). The flag measures feelings, not behaviour.

### B3. Workflow diagram

Edit the two `TODO` boxes in `diagrams/M2_workflow_unrolled.dot` (K1 and K2: one short line each with a key number). Then render it:

```bash
cd diagrams
dot -Tpng -Gdpi=170 M2_workflow_unrolled.dot -o ../images/M2_workflow_unrolled.png
```

No Graphviz installed? Paste the `.dot` file into https://dreampuf.github.io/GraphvizOnline, download the PNG, and save it as `images/M2_workflow_unrolled.png`.

Also open `diagrams/A3_master_workflow.dot` and update the wellbeing result in node `w2` when your interpretation is final. Regenerate both the PNG for the README and the PDF used in the report with the commands at the top of the DOT file.

### B4. Write Section 5

Write it in `report/sections/05_member2_advaya.tex`: replace each red `\todo{...}` / `\todoi{...}` with real text. The TODO boxes already say what goes where. Copy the **style** of Section 4 (`04_member1_tushaar.tex`):

- driving question, then method summary;
- a task/rationale table: one row per figure, saying *why that chart type*. The rubric asks for "rationale behind visualizations".
- Iteration 1, then K1, then the feedback loop (say which inference caused which change), then Iteration 2, then K2;
- captions that state the finding and the number, not just "PR curve".

### B5. Assemble the report (Advaya only)

1. Write `01_introduction.tex`, then the Member 2 and Member 3 paragraphs and the closing paragraph in `07_synthesis.tex`, then items 2–4 in `08_conclusions.tex`. Each file's TODO says what to write.
2. Fill your row in `09_contributions.tex`, and get Aman to fill his.
3. Compile:
   ```bash
   cd report
   pdflatex main.tex
   pdflatex main.tex
   ```
   Or upload the whole repo to Overleaf and set `report/main.tex` as the main file. The A1 report is attached automatically as the Appendix.
4. Search the PDF for "TODO" and "missing image". **Both must return nothing.**
5. Read the PDF once end to end. Check that all figures appear, the numbers in the text match the `outputs/*.json` files, and there are no `??` references.

---

## Part C: Aman (regression and statistical tests)

### C1. Reproduce the five completed figures

Run `python Member3/m3_regression_workflow.py` after preprocessing.

| File | What it shows | Chart |
|---|---|---|
| `M3_fig01_iter1_fit` | predicted vs actual on test, OLS and boosting | 2 hexbin panels + y = x line, R² in titles |
| `M3_fig02_iter1_residuals` | OLS residual vs fitted (train) + red line residual = 100 − fitted | hexbin |
| `M3_fig03_iter1_coefficients` | OLS coefficients × feature SD, with 95% CI | dot-and-whisker (forest) plot |
| `M3_fig04_iter2_forest` | median-regression coefficients × SD, 95% CI, GPA < 4.0 rows | forest plot |
| `M3_fig05_iter2_tutoring` | tutoring gap per income band: raw (grey) vs GPA-adjusted (blue), 95% CI | dot plot on one axis (like Tushaar's Fig 9) |

### C2. Expected numbers

| What | Value |
|---|---|
| OLS test R² / MAE | 0.871 / 3.19 |
| Gradient boosting test R² / MAE | 0.872 / 3.23 |
| OLS prior-GPA slope | 23.35 points per GPA unit |
| OLS tutoring coefficient | −0.028 (p = 0.40) |
| Every other OLS coefficient | |b| < 0.03, not significant |
| Residuals exactly on the 100 − fitted edge | 25.7% (= students scoring 100) |
| Largest OLS prediction | 98.7 (never above 100) |
| Rows after dropping GPA = 4.0 | 38,466, of which 7.2% score 100 |
| Median regression tutoring | 0.004 (p = 0.94) |
| ANOVA raw: tutoring | p < 1e-100 |
| ANOVA raw: income × tutoring | p = 0.052 |
| ANCOVA with GPA covariate: tutoring | p = 0.43 |
| Mean prior GPA, tutored vs not | 3.662 vs 3.542 |

**What it means:**

- **K1:** the exam is prior GPA plus noise. The ceiling shows as a hard residual edge, not as predictions above 100, and 40% of students sit at one GPA value (4.0).
- **K2:** the +2.8-point tutoring effect from A1 disappears once prior GPA is controlled. Tutored students already had higher GPAs. Tutoring and prior GPA are recorded together, so the data cannot tell which came first. Say this, because the rubric checks that inferences respect time order.

### C3. Diagram and Section 6

The individual diagram and the master diagram include the verified results. For the separate team report, follow B4 using `report/sections/06_member3_aman.tex`.

### C4. Video demo and slides (Aman only)

- **Length:** about 9–10 minutes. Each member presents their own strand, about 3 minutes each.
- **Tushaar's segment:** use the [captioned segment](Member1/Member1_habits_demo.mp4) for the visuals. Tushaar can record his voice-over using the [viva notes](Member1/M1_viva_and_video.md). His current report figures begin with `M1_gpa_`.
- **Slide order:**
  1. title;
  2. question and dataset;
  3. master workflow (`images/A3_master_workflow.png`);
  4. Tushaar's segment;
  5. Advaya's segment;
  6. Aman's segment;
  7. synthesis;
  8. conclusions.
- **For each figure slide:** one sentence of finding as the slide title and the figure full-width. The brief says PowerPoint support in the video "will be appreciated".
- **Show the feedback loop on screen:** pause on each member's workflow diagram at the red "FEEDBACK LOOP" edge and say what inference caused which transformation. This is a graded item.

---

## Part D: final submission checklist (from the A3 brief)

- [ ] Report PDF: workflow diagram with the "Feedback loop" edge (done), rationale for the visualizations, mining/analysis methods, A1 report as Appendix (done automatically).
- [ ] `images/` folder: all `M1_*`, `M2_*`, `M3_*`, `A3_master_workflow.png`.
- [ ] Video demo file.
- [ ] No "TODO" or "missing image" anywhere in the PDF.
- [ ] Every member can explain every figure in their own section, **and** the master workflow.
- [ ] Submitted by Friday 27 November (hard deadline Sunday 29, 11:59 pm).

## Part E: git without conflicts

```bash
git pull                       # always first
# ... work ...
git add <only your files>
git commit -m "Member 2: figures 1-3"
git pull --rebase
git push
```

| Person | Edits only |
|---|---|
| Advaya | `Member2/`, `images/M2_*`, `diagrams/M2_*`, `report/sections/05_*`, `01_*`, `07_*`, `08_*` |
| Aman | `Member3/`, `images/M3_*`, `diagrams/M3_*`, `report/sections/06_*` |
| Shared, one at a time (tell the group first) | `09_*`, `11_*`, `diagrams/A3_master_workflow.dot`, `report/main.tex` |

Do not edit `preprocess_a3.py`, `a3_common.py`, `Member1/` or sections 02–04. If you think something there is wrong, message Tushaar.

## Part F: quick answers

- **Do I need statsmodels?** Aman, yes (it's in `requirements.txt`). Advaya, no.
- **Where are numbers for the text?** In `MemberX/outputs/mX_results.json`, which your script writes.
- **Can I change the models?** Yes, if you can justify it in the viva. Keep the two-iteration structure and the feedback-loop transformation.
- **Where's the A1 report?** `reference/DV_Assignment1_report.pdf`. The A3 brief is `reference/DAS732_T1-26-27_A3.pdf`.
