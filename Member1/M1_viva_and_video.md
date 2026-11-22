# Member 1 (Tushaar): viva notes and video script

## The story in four sentences

I clustered students on their habits to see whether "good-habit" profiles exist and whether they help in the exam beyond prior GPA. On the six raw habits there are no real clusters: the silhouette equals shuffled data and partitions change with the seed, and the small 2-point score gap follows prior GPA. That inference drove the feedback loop: I rebuilt the feature space (leisure share instead of raw leisure hours, dropped attendance) and replaced the raw score with a GPA-adjusted score. In iteration 2 a stable study-focused vs leisure-heavy split appears with a 4.2-point raw advantage, but that advantage is exactly its prior-GPA advantage and is 0.0 points (CI ±0.13) once GPA is held fixed, on the unseen test split too.

## Numbers to know by heart

| What | Value |
|---|---|
| Max correlation between any two raw habits | 0.007 |
| Iteration 1 silhouette (k=4), real vs shuffled | 0.117 vs 0.116 |
| Iteration 1 seed stability (k=4) | ARI 0.24 |
| Iteration 1 score gap / effect size | 2.0 pts, eta² = 0.0028 |
| Iteration 2 stability (k=2) | ARI 0.99, silhouette 0.233 vs null 0.194 |
| Cluster centroids | 5.5 vs 2.5 study h/day; leisure share 37% vs 60% |
| Raw score gap (train / test) | +4.22 / +4.14, d = 0.37 / 0.36 |
| GPA-adjusted gap (test) | -0.01, 95% CI [-0.13, +0.13], p = 0.92 |
| GPA bins alone explain | R² = 0.870 of test score |
| GBM habits -> score / GPA / adjusted (test R²) | 0.070 / 0.080 / 0.000 |
| Students at GPA 4.0, of which score 100 | 39.9%, 53.6% |

## Likely viva questions

**Why K-means and not DBSCAN or hierarchical?** 64,000 rows rule out hierarchical clustering (O(n²) memory). DBSCAN needs density gaps, and Fig 1-2 show there are none in D1. K-means plus a GMM BIC check covers spherical and elliptical groups. The point of iteration 1 was to test whether any structure exists, and the shuffled null answers that whatever the algorithm.

**What is the shuffled null?** Each column is permuted on its own. That keeps every variable's distribution but breaks any relation between variables. If real data clusters no better than that, the clusters come from the marginal shapes, not from students sharing a pattern.

**What is ARI stability?** Run K-means six times with different seeds and one start each; adjusted Rand index measures how often pairs of students land together across runs. 1 means identical partitions, 0 means chance agreement.

**Why is ARI 0.00 at k=2 in iteration 1?** The cloud is round, so every direction is an equally good cut and each seed picks a different one.

**Isn't choosing features after seeing them "cheating"?** T1 used two things: the cluster profiles (which variables K-means split on) and A1's finding that the leisure *share*, not raw hours, relates to performance. That is the human-in-the-loop step the diamond model describes. The test of T1 is not the score; it is that the new space has stable structure (ARI 0.99) that beats the null, which the raw space did not.

**Why did the refined space cluster when the raw one didn't?** Leisure share and study hours correlate at -0.64, partly by construction because screen time includes study time. That correlation creates a real axis (study-dominant vs leisure-dominant screen use). I mention this openly in the report.

**Why bin-mean adjustment instead of regression residuals?** The score–GPA relation bends at the top because of the 100 ceiling. Bins of 0.1 GPA follow any shape without assuming one. Means are learned on train only and applied to test, so there is no leakage.

**What about the ceiling?** 40% have GPA 4.0, and over half of those score 100, so their adjusted score is capped. I repeated the test without them: gap +0.01, CI [-0.19, +0.20].

**Why does the Below-80 band still show 2.5 points?** That band spans GPA 1.64 to 3.20. Inside it, study-focused students have GPA 0.10 higher, worth about 2.4 points at roughly 24 exam points per GPA unit in that range. After fine adjustment the gap is -0.01.

**Does this mean studying doesn't matter?** No. It means this dataset can't separate habits from prior attainment. Habits predict prior GPA (R² 0.08) as well as the exam (0.07). Prior GPA comes before the exam, and the habits are measured once, so the defensible reading is that habits travel with a student's long-run trajectory. The data is cross-sectional and supports no causal claim either way.

**What's the "correctness w.r.t. time" point?** Prior GPA precedes the exam, so it can be a cause of the exam score or share causes with it, but not the reverse. Habits are a single snapshot, so we cannot order them relative to prior GPA.

**How does this connect to the team?** Same confound as A1's Fig 19 partial correlations, now shown with a different method (clustering + adjustment) and confirmed on a held-out split. The Below-80 result also suggests re-checking A1 Fig 20 with fine-grained adjustment.

## Video script (about 3 minutes, one slide per line)

1. **Workflow slide (M1_workflow_unrolled.png).** "My strand asks whether students form habit profiles and whether a good profile helps beyond prior GPA. Here is my diamond, unrolled: iteration one on the top row, the feedback loop in red, iteration two below."
2. **Fig 1.** "The six raw habits don't correlate with each other at all, and three of them don't correlate with the score."
3. **Fig 2.** "K-means finds no preferred k, and the silhouette sits exactly on the shuffled-data line. Different seeds give different clusters."
4. **Fig 3 + Fig 4.** "The four clusters slice one round cloud along exercise and leisure hours, never study hours. They differ by two points, with eta-squared of 0.003."
5. **Fig 5.** "The lower clusters just hold more low-GPA students. Inside a GPA band the gap almost disappears."
6. **Workflow slide, red edge.** "That inference is my feedback loop. I transform the features, swapping raw leisure hours for leisure share and dropping attendance, and I transform the target into a GPA-adjusted score, learned on train only. I also flag the GPA-4.0 ceiling."
7. **Fig 6 + Fig 7.** "Now k equals two is stable, ARI 0.99, and beats the null. The clusters are study-focused, five and a half hours a day, and leisure-heavy, two and a half."
8. **Fig 8.** "Study-focused students score 4.2 points higher. They also arrive with 4.5 points more prior GPA. Hold GPA fixed and the gap is zero, on the test set and without the ceiling group."
9. **Fig 9.** "Even the 2.5-point gap in the lowest band is GPA spread inside that band."
10. **Fig 10 + close.** "A gradient-boosting check agrees: habits predict prior GPA, not the exam beyond it. Habits mark a student's trajectory; this data shows no exam-specific benefit."
