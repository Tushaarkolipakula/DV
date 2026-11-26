# DAS732 A3 · Student habits, wellbeing and performance

The three analyses use the same 80,000 A1 records. Open [the team workflow](images/A3_master_workflow.png) to see what each first iteration revealed and which transformation followed. The individual workflow PNGs are in `images/`, and their editable Graphviz sources are in `diagrams/`.

| Member | Start here | What the second iteration tests |
|---|---|---|
| Tushaar · habits | [Workflow](images/M1_workflow_unrolled.png) · [Score gaps](Member1/outputs/gpa_gaps.csv) | Refit K-means and Gaussian mixtures independently within four prior-GPA bands. Giving the original clusters a common GPA mix reduces the K-means score gap from 4.15 to 0.51 points. Fresh band fits give +0.41 for K-means and −0.35 for the mixture. Local groups depend on the model. |
| Advaya · wellbeing | [Workflow](images/M2_workflow_unrolled.png) · [Model comparison](Member2/outputs/model_metrics.csv) | Remove stress and motivation, which define the dropout-risk flag, then refit three classifiers. The tree’s held-out average precision falls from 1.000 to about 0.043. The positive rate is 316 of 16,000 (1.975%). This measures recovery of a constructed flag. Actual dropout was not observed. |
| Aman · background | [Workflow](images/M3_workflow_unrolled.png) · [Numerical results](Member3/outputs/m3_results.json) | Diagnose the score ceiling, fit median regression below the GPA ceiling, and compare tutoring across income bands with and without prior-GPA adjustment. The 2.76-point raw gap falls to approximately −0.03. The data establish an association and leave the timing of tutoring unresolved. |

## Reproduce the analysis

Use a Python environment with the dependencies in `requirements.txt`. Run these commands from this folder:

```shell
python -m pip install -r requirements.txt
python preprocess_a3.py
python Member1/m1_habits_by_gpa.py
python Member2/m2_classification_workflow.py
python Member3/m3_regression_workflow.py
python -m unittest discover -s tests -v
```

The shared script rebuilds the base table from `data/master_cleaned.csv.gz`. It removes redundant fields including exam anxiety, records the target definitions, fixes a stratified 64,000/16,000 split with seed 42, and learns category encodings and numeric scales on training rows. The preprocessing log is in `outputs/`. Each member’s numerical results are in their own `outputs/` folder.

## Read the figures

- **Habits:** The `M1_gpa_*` figures cover the assigned analysis. The elbow and silhouette curves guide model selection. PCA maps and profile heatmaps show how the fitted groups differ. Score boxes, a cluster × GPA composition table and confidence intervals explain how the score gap changes. The earlier `M1_fig*` series is a supplementary sensitivity analysis.
- **Wellbeing:** Figures 1–3 expose the flag definition using a heatmap, precision–recall and ROC curves, a tree and forest importances. Figures 4–6 compare the same classifiers after removal, rank the remaining predictors and show flag prevalence across GPA and mental-health groups. Figure 7 shows classification counts at a 0.5 score threshold. PR summaries use average precision; class-weighted scores are used for ranking.
- **Background:** Figures 1–3 show held-out predictions, training residuals and standardized OLS coefficients. Figures 4–5 show median-regression intervals and income-specific tutoring gaps. Median regression uses 38,466 uncapped training rows; income and tutoring tests use all 64,000 training rows.

The A1 data provide the habit measurements, prior GPA, flag inputs and exam scores required for these comparisons. No external dataset is required. The written report with its A1 appendix, the team video and individual attendance at the live viva are separate assignment requirements.
