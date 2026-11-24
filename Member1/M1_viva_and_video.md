# Tushaar's viva and video notes

**Question.** When students are grouped by their habits alone, how large is the exam-score difference between the profiles, and how much remains after accounting for the groups' different prior GPAs?

The [captioned demo segment](Member1_habits_demo.mp4) runs about 2 minutes 11 seconds. It follows the question from the pooled profiles through the GPA composition check, the band refits and the GMM sensitivity result. The guide below gives a longer spoken version for the live demo or a voice-over with additional figure pauses.

## Three-minute recording guide

| Time | Show | What to explain |
|---|---|---|
| 0:00–0:25 | [Member workflow](../images/M1_workflow_unrolled.png) | A1 found that daily study hours correlated with exam score at 0.24, while prior GPA correlated at 0.93. I wanted to know whether a combination of habits still distinguished scores among students with similar starting points. |
| 0:25–0:55 | [Elbow and silhouette](../images/M1_gpa_selection.png), [pooled PCA](../images/M1_gpa_pca_global.png) | I clustered five numeric habits and one categorical study environment. I used 64,000 training students to fit the scaling and models. The elbow suggested three groups, but the silhouette peaked at two. The value of 0.159 shows substantial overlap. K-means and the mixture agreed fairly closely overall, at ARI 0.73. |
| 0:55–1:25 | [Pooled average habits](../images/M1_gpa_profiles_global.png), [score boxes](../images/M1_gpa_scores_global.png) | On the 16,000 held-out students, the higher-study group averaged 5.46 study hours and 37% leisure screen time, compared with 2.52 hours and 60%. It scored 91.02 rather than 86.86. The rank test detects the difference, but its effect size is only 0.025. Point to the overlap in the boxes. |
| 1:25–1:55 | [GPA composition](../images/M1_gpa_mix.png), then the **Feedback loop** edge on the workflow | In the higher-study cluster, 46% entered with a perfect GPA. In the lower-study cluster it was 32%. I gave the same two clusters a common GPA-band mix. Their 4.15-point gap fell to 0.51. That prompted me to fit a new pair of models inside each of the four bands. |
| 1:55–2:25 | [Within-band PCA](../images/M1_gpa_pca_bands.png), [habit profiles](../images/M1_gpa_profiles_bands.png), [score boxes](../images/M1_gpa_scores_bands.png) | I repeated scaling, selection, model fitting and plotting using each band's training students. K-means still separated higher and lower study time. The lowest band retained a 2.01-point gap. In each of the other three bands the estimate was within 0.20 points of zero. In the GPA-4.0 band both cluster medians reached 100. |
| 2:25–3:00 | [Contrasts and intervals](../images/M1_gpa_gaps.png) and [GMM profiles by band](../images/M1_gpa_profiles_gmm_bands.png) | With the band-specific labels, the weighted K-means gap was 0.41 points. The mixture gave −0.35. In the lowest GPA band the mixture split dormitory and café students from the other environments. Its two groups differed by only a quarter hour of average study time. Both algorithms agree that most of the pooled score gap tracks prior GPA. What remains depends on which profile we compare. |

## Numbers to check against the figures

All score contrasts are mean exam points, higher-study cluster minus lower-study cluster. Labels are ordered by training mean study time. Intervals resample held-out students within bands and condition on the fitted models.

| Comparison | K-means, 95% CI | GMM, 95% CI |
|---|---|---|
| All held-out students, global labels | 4.15 [3.81, 4.52] | 4.19 [3.81, 4.56] |
| Same labels, common GPA mix | 0.51 [0.35, 0.69] | 0.49 [0.32, 0.65] |
| New labels within each band, weighted across bands | 0.41 [0.24, 0.58] | −0.35 [−0.52, −0.19] |
| Lowest GPA band, new labels | 2.01 [1.46, 2.53] | −1.54 [−2.08, −1.01] |

## Questions to rehearse

**What exactly changed in the feedback loop?** The cluster-by-GPA table showed unequal starting attainment. I split the modelling population by prior GPA and fitted new scalers, K-means, GMM and PCA inside each band. Every student has one global assignment and one assignment from their own band's model. The four local fits all selected two clusters, but the label identities can change.

**What do the two GPA comparisons measure?** Holding global labels fixed isolates the association with composition across the four bands. The K-means gap fell by 3.64 points, or about 88%. Refitting measures something else. It asks whether the profiles found among students of similar prior attainment differ in score. I combine the four local differences using each band's share of the test set.

**Which habits went into clustering?** Daily study hours, sleep, weekly exercise, leisure screen share and attendance, plus one-hot study environment. I standardize each numeric field using training values. I center the environment indicators and scale them as a block, giving one original categorical variable the same total variance as one numeric input. Scores only enter the later comparisons. Prior GPA only defines the bands.

**Why two clusters if the elbow is at three?** Both plots carry information. The rule admits candidate values within 0.01 of the best silhouette and then prefers the candidate nearest the elbow. The pooled silhouette was 0.159 at two and 0.134 at three, so three missed the tolerance. These are descriptive profiles with overlap.

**Why can the algorithms disagree?** K-means groups students by distance to centroids. A diagonal Gaussian mixture allows different spreads along each axis. It can lock onto the encoded environment categories. This happened in the lowest band, where the algorithms' ARI was 0.004. The mixture's higher-study cluster was entirely café/dorm students. Other environments made up the other cluster. Compare their study means of 3.52 and 3.27 hours with the K-means means of 4.60 and 1.85 hours.

**What does the score test show?** Kruskal–Wallis compares score ranks and corrects for tied values such as 100. The pooled K-means test had H = 393.46 with one degree of freedom and a rank effect size of 0.0245. In each algorithm only the lowest GPA band retained a significant difference after Holm adjustment across its four band tests. The bootstrap intervals report the size and direction of the mean differences in exam points.

**Why avoid a causal claim?** Each student contributes one observation. Broad GPA bands still admit different prior GPAs within them, prior GPA and scores have upper bounds, and the profile definitions change after refitting. The results describe these students at the recorded time. They cannot estimate how changing a habit would affect the same student's score.

## Live demo

From the repository root, run the shared preprocessing and then the assigned analysis:

~~~bash
python preprocess_a3.py
python Member1/m1_habits_by_gpa.py
~~~

Open the [score contrasts](outputs/gpa_gaps.csv), [rank tests](outputs/gpa_score_tests.csv) and [profile means](outputs/gpa_profiles.csv) alongside the figures if asked to trace a number. The A1 background is in Figures 2, 3 and 8 of the [A1 report](../reference/DV_Assignment1_report.pdf).
