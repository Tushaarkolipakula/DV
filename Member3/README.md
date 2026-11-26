# Aman’s A3 regression analysis

A1 Figure 24 shows a roughly 2.8-point exam-score gap between students with and without tutoring across income bands. Figure 25 shows a strong association between prior GPA and exam score. This analysis asks whether the tutoring gap persists after accounting for prior GPA.

From the project root, install the dependencies and reproduce the figures:

```bash
python -m pip install -r requirements.txt
python preprocess_a3.py
python Member3/m3_regression_workflow.py
```

The pipeline uses 64,000 training students and 16,000 held-out students. No additional data download is needed.

| Evidence | What to look for |
|---|---|
| Figure 1, held-out predictions | OLS and gradient boosting give similar test R² (0.871 and 0.872). GPA alone gives 0.871 and a mean absolute error of 3.19 points. |
| Figure 2, training residuals | Students scoring 100 lie on the line where residual equals 100 minus fitted score. They comprise 25.7% of training observations. |
| Figure 3, OLS coefficients | One training standard deviation of prior GPA corresponds to about 10.8 fitted exam points. Bars show HC3 confidence intervals. |
| Figure 4, median regression | Removing the GPA ceiling leaves 38,466 training students. The tutoring coefficient is +0.004 points, with a 95% interval of −0.114 to +0.123. Median regression still models the observed, capped score. |
| Figure 5, tutoring by income | Each row compares mean exam scores, then mean residuals from the training GPA-only regression. Bars show approximate 95% intervals from group variances, conditional on the fitted GPA adjustment. |

The separate ANOVA and ANCOVA use all 64,000 training students. The raw tutoring association is strong. After adjustment, the type-II tutoring test gives p = 0.427 and the income-by-tutoring interaction gives p = 0.149. The overall descriptive gap falls from +2.76 to −0.03 points. Tutoring access and GPA were observed together, so the data cannot establish the direction of a causal effect.

Figures are in `images/M3_*.png`. The editable unrolled workflow is in `diagrams/M3_workflow_unrolled.dot`, with PDF and PNG renders. `Member3/outputs/m3_results.json` contains fit metrics, tests and income-band contrasts. The two CSVs alongside it contain the standardized coefficient estimates and confidence limits used in Figures 3 and 4.
