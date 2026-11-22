"""
Member 1 (Kolipakula Sai Tushaar) - What students do.
Visual analytics workflow with one feedback loop (Keim et al. diamond, unrolled).

Driving question
    Do students form distinct habit profiles, and does belonging to a
    "good-habits" profile give an exam advantage beyond prior attainment?

ITERATION 1  (data D1 = six raw habit variables, target = raw exam score)
    V1  fig01 habit correlation matrix
    M1  K-means k = 2..8 with silhouette vs. a column-shuffled null,
        seed-stability (ARI), GMM BIC, Hopkins statistic             -> fig02
    V1  fig03 cluster map (PCA) + centroid profiles
    V1  fig04 exam score by cluster (Kruskal-Wallis, eta^2)
    V1  fig05 cluster x prior-GPA composition and within-band scores
    K1  partitions are unstable and split on leisure hours that carry no
        score signal; the score gap that exists follows prior GPA.

FEEDBACK LOOP  (transformation of the data, driven by K1)
    T1  feature space: drop attendance, social-media hours, streaming hours;
        add leisure_screen_share  -> D2 = 4 habit variables
    T2  target: GPA-conditional score  adj = score - mean(score | GPA bin),
        bin means learned on the training split only
    T3  censoring: flag GPA = 4.0 rows; re-test with them excluded

ITERATION 2  (data D2, targets raw and adjusted score)
    M2  K-means re-selected on D2 (stability jumps to ARI ~ 1 at k = 2) -> fig06
    V2  fig07 refined cluster map and profiles
    V2  fig08 raw vs. GPA-adjusted cluster gap, bootstrap CIs, test replication
    V2  fig09 within-band small multiples
    M2  gradient boosting check (habits -> score / GPA / adjusted)      -> fig10
    K2  habit profiles mark the prior-attainment trajectory; they carry no
        exam-specific advantage once prior GPA is held fixed.

Usage (from the repository root):
    python preprocess_a3.py
    python Member1/m1_clustering_workflow.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import adjusted_rand_score, silhouette_score, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a3_common import (load_base, FEATURES, PALETTE, SEED, IMAGES_DIR,  # noqa: E402
                       GPA_BAND_LABELS, set_style)

OUT = ROOT / "Member1" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
set_style()
rng = np.random.default_rng(SEED)
C = PALETTE["cluster"]
DIV = LinearSegmentedColormap.from_list("div", PALETTE["diverging"])
R = {}                                    # every number quoted in the report


def save(fig, name):
    fig.savefig(IMAGES_DIR / f"{name}.png")
    plt.close(fig)
    print("  saved", name)


def ci95(x):
    x = np.asarray(x)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def eta_sq_kruskal(H, k, n):
    return (H - k + 1) / (n - k)


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def hopkins(X, m=1000):
    """~0.5 for uniformly random data, -> 1 for strongly clustered data."""
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    samp = X[rng.choice(len(X), m, replace=False)]
    unif = rng.uniform(X.min(0), X.max(0), (m, X.shape[1]))
    w = nn.kneighbors(samp, 2)[0][:, 1]
    u = nn.kneighbors(unif, 1)[0][:, 0]
    return float(u.sum() / (u.sum() + w.sum()))


def k_selection(X, ks, tag):
    """Inertia, silhouette vs. shuffled null, seed stability, GMM BIC."""
    idx = rng.choice(len(X), 8000, replace=False)
    Xnull = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        kn = KMeans(k, n_init=10, random_state=SEED).fit(Xnull)
        seeds = [KMeans(k, n_init=1, random_state=s).fit_predict(X) for s in range(6)]
        ari = np.mean([adjusted_rand_score(seeds[0], s) for s in seeds[1:]])
        gmm = GaussianMixture(k, random_state=SEED).fit(X[idx])
        rows.append(dict(k=k, inertia=km.inertia_,
                         silhouette=silhouette_score(X[idx], km.labels_[idx]),
                         silhouette_null=silhouette_score(Xnull[idx], kn.labels_[idx]),
                         stability_ari=ari, gmm_bic=gmm.bic(X[idx])))
        print(f"  [{tag}] k={k} sil={rows[-1]['silhouette']:.3f} "
              f"null={rows[-1]['silhouette_null']:.3f} ARI={ari:.2f}")
    return pd.DataFrame(rows)


# ==========================================================================
# DATA
# ==========================================================================
train, test = load_base("train"), load_base("test")
F1 = FEATURES["habits"]
LABEL = {"study_hours_per_day": "Study h/day", "sleep_hours": "Sleep h",
         "exercise_frequency": "Exercise /wk", "attendance_percentage": "Attendance %",
         "social_media_hours": "Social media h", "netflix_hours": "Streaming h",
         "leisure_screen_share": "Leisure screen share",
         "previous_gpa": "Prior GPA", "exam_score": "Exam score"}
print(f"Train {len(train):,}  Test {len(test):,}")

# ==========================================================================
# ITERATION 1 - VISUALISE: habit correlation structure
# ==========================================================================
print("\nIteration 1")
cols = F1 + ["previous_gpa", "exam_score"]
corr = train[cols].corr()
R["it1_max_abs_r_between_habits"] = float(
    corr.loc[F1, F1].where(~np.eye(len(F1), dtype=bool)).abs().max().max())
R["it1_r_with_score"] = corr.loc[F1, "exam_score"].round(3).to_dict()

fig, ax = plt.subplots(figsize=(6.8, 5.6))
im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), [LABEL[c] for c in cols], rotation=40, ha="right")
ax.set_yticks(range(len(cols)), [LABEL[c] for c in cols])
ax.grid(False)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color=PALETTE["text"])
ax.axhline(5.5, color="white", lw=2); ax.axvline(5.5, color="white", lw=2)
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_title("Iteration 1 input: the six raw habits are mutually uncorrelated", loc="left")
save(fig, "M1_fig01_habit_correlations")

# ==========================================================================
# ITERATION 1 - MODEL: K-means on six standardised raw habits
# ==========================================================================
sc1 = StandardScaler().fit(train[F1])
X1 = sc1.transform(train[F1])
ks = range(2, 9)
sel1 = k_selection(X1, ks, "raw")
sel1.to_csv(OUT / "it1_k_selection.csv", index=False)
R["it1_hopkins"] = hopkins(X1)
R["it1_k_selection"] = sel1.round(4).to_dict("records")


def plot_k_selection(sel_list, labels, colors, title, name):
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for sel, lab, col in zip(sel_list, labels, colors):
        axs[0].plot(sel.k, sel.inertia / sel.inertia.iloc[0], "-o", ms=5, color=col, label=lab)
        axs[1].plot(sel.k, sel.silhouette, "-o", ms=5, color=col, label=f"{lab}: real")
        axs[1].plot(sel.k, sel.silhouette_null, "--", color=col, lw=1.5,
                    label=f"{lab}: shuffled null")
        axs[2].plot(sel.k, sel.stability_ari, "-o", ms=5, color=col, label=lab)
        axs[3].plot(sel.k, (sel.gmm_bic - sel.gmm_bic.iloc[0]) / 1000, "-o", ms=5,
                    color=col, label=lab)
    axs[0].set(title="Elbow (inertia, k=2 = 1)", xlabel="k")
    axs[1].set(title="Silhouette vs. null", xlabel="k")
    axs[2].set(title="Seed stability (mean ARI)", xlabel="k", ylim=(0, 1.05))
    axs[3].set(title="GMM BIC change vs. k=2 (thousands)", xlabel="k")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, name)


plot_k_selection([sel1], ["Raw habits (D1)"], [C[0]],
                 "Iteration 1 model selection: no k is preferred, and the partitions "
                 "are no better than shuffled data", "M1_fig02_k_selection_raw")

K1 = 4   # every criterion is flat, so k is set for readable profiles
km1 = KMeans(K1, n_init=10, random_state=SEED).fit(X1)
# order clusters by mean exam score so labels are stable across runs
order = train.assign(c=km1.labels_).groupby("c").exam_score.mean().sort_values(ascending=False).index
remap = {old: new for new, old in enumerate(order)}
train["cl1"] = pd.Series(km1.labels_).map(remap).values
test["cl1"] = pd.Series(km1.predict(sc1.transform(test[F1]))).map(remap).values
cent1 = pd.DataFrame(km1.cluster_centers_, columns=F1).rename(index=remap).sort_index()
R["it1_k"] = K1
R["it1_centroids_z"] = cent1.round(2).to_dict("index")
R["it1_centroid_spread_z"] = (cent1.max() - cent1.min()).round(2).to_dict()

# ---- fig03: PCA map faceted by cluster + centroid profile ----------------
pca1 = PCA(2, random_state=SEED).fit(X1)
P1 = pca1.transform(X1)
R["it1_pca_var"] = pca1.explained_variance_ratio_.round(3).tolist()
fig = plt.figure(figsize=(14, 4.2))
gs = fig.add_gridspec(1, K1 + 2, width_ratios=[1] * K1 + [0.15, 1.7])
lim = np.percentile(np.abs(P1), 99.5)
for c in range(K1):
    ax = fig.add_subplot(gs[0, c])
    ax.hexbin(P1[:, 0], P1[:, 1], gridsize=40, cmap="Greys", bins="log",
              extent=(-lim, lim, -lim, lim), alpha=0.35, linewidths=0)
    m = train.cl1.values == c
    ax.hexbin(P1[m, 0], P1[m, 1], gridsize=40, bins="log", mincnt=1,
              extent=(-lim, lim, -lim, lim), linewidths=0,
              cmap=LinearSegmentedColormap.from_list("c", ["#ffffff", C[c]]))
    ax.set(title=f"Cluster {c + 1}  (n={m.sum():,})", xticks=[], yticks=[],
           xlabel="PC1", ylabel="PC2" if c == 0 else "")
    ax.grid(False)
ax = fig.add_subplot(gs[0, K1 + 1])
im = ax.imshow(cent1.values, cmap=DIV, vmin=-1.2, vmax=1.2, aspect="auto")
ax.set_xticks(range(len(F1)), [LABEL[f] for f in F1], rotation=35, ha="right")
ax.set_yticks(range(K1), [f"Cluster {c + 1}" for c in range(K1)])
ax.grid(False)
for i in range(K1):
    for j in range(len(F1)):
        ax.text(j, i, f"{cent1.values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="centroid (z)")
ax.set_title("Centroid profiles", loc="left")
fig.suptitle(f"Iteration 1 clusters (k={K1}) carve one round cloud; they split on exercise, "
             "social-media and streaming hours, never on study hours", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "M1_fig03_cluster_map_raw")

# ---- fig04: score by cluster ----------------------------------------------
groups = [train.exam_score[train.cl1 == c] for c in range(K1)]
H, p = stats.kruskal(*groups)
R["it1_kruskal_H"], R["it1_kruskal_p"] = float(H), float(p)
R["it1_eta2"] = float(eta_sq_kruskal(H, K1, len(train)))
R["it1_cluster_score"] = {c + 1: round(float(g.mean()), 2) for c, g in enumerate(groups)}
R["it1_cluster_gpa"] = train.groupby("cl1").previous_gpa.mean().round(3).rename(lambda c: c + 1).to_dict()
R["it1_score_gap"] = float(max(g.mean() for g in groups) - min(g.mean() for g in groups))

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axs[0]
parts = ax.violinplot(groups, positions=range(K1), showextrema=False, widths=0.8)
for b, col in zip(parts["bodies"], C):
    b.set_facecolor(col); b.set_alpha(0.35); b.set_edgecolor(col)
ax.set_xticks(range(K1), [f"Cluster {c + 1}" for c in range(K1)])
ax.set(ylabel="Exam score", title="Score distributions overlap almost completely")
ax = axs[1]
for c, g in enumerate(groups):
    ax.errorbar(c, g.mean(), yerr=ci95(g), fmt="o", ms=8, color=C[c], capsize=4, lw=2)
    ax.annotate(f"{g.mean():.1f}", (c, g.mean()), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=9)
ax.set_xticks(range(K1), [f"C{c + 1}" for c in range(K1)])
ax.set(ylabel="Mean exam score (95% CI)", xlim=(-0.5, K1 - 0.3),
       title=f"Gap {R['it1_score_gap']:.1f} pts, eta² = {R['it1_eta2']:.4f}")
fig.suptitle(f"Iteration 1: clusters differ in score (Kruskal-Wallis H = {H:.0f}, "
             f"p < 1e-30) but the effect is tiny", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "M1_fig04_score_by_cluster_raw")

# ---- fig05: confound check against prior GPA ------------------------------
comp = pd.crosstab(train.cl1, train.gpa_band, normalize="index")[GPA_BAND_LABELS]
within = train.pivot_table("exam_score", "gpa_band", "cl1", observed=True)
R["it1_share_below80"] = comp["Below 80"].round(3).rename(lambda c: c + 1).to_dict()
R["it1_within_band_gap"] = (within.max(1) - within.min(1)).round(2).to_dict()

fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axs[0]
left = np.zeros(K1)
for b in GPA_BAND_LABELS:
    ax.barh(range(K1), comp[b], left=left, color=PALETTE["gpa_band"][b],
            edgecolor=PALETTE["surface"], lw=2, label=b, height=0.7)
    for i, v in enumerate(comp[b]):
        ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                color="white" if b != "Below 80" else PALETTE["text"])
    left += comp[b].values
ax.set_yticks(range(K1), [f"Cluster {c + 1}" for c in range(K1)])
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(False); ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8,
          title="Prior-GPA band (previous_gpa_pct)", title_fontsize=8)
ax.set_title("Lower-scoring clusters hold more low-GPA students", loc="left")
ax = axs[1]
for c in range(K1):
    ax.plot(within[c].values, range(len(GPA_BAND_LABELS)), "o", ms=8, color=C[c],
            label=f"Cluster {c + 1}", alpha=0.9)
for i, b in enumerate(GPA_BAND_LABELS):
    ax.text(within.loc[b].max() + 1.0, i, f"gap {within.loc[b].max() - within.loc[b].min():.2f}",
            va="center", fontsize=8.5, color=PALETTE["text_secondary"])
ax.set_yticks(range(len(GPA_BAND_LABELS)), GPA_BAND_LABELS)
ax.set(xlabel="Mean exam score", xlim=(68, 104),
       title="Inside each band the gap shrinks to 0.2 pts or less, except 'Below 80'")
ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Iteration 1 inference: the cluster gap follows prior-GPA composition",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "M1_fig05_gpa_confound_raw")

# ==========================================================================
# FEEDBACK LOOP - TRANSFORM THE DATA
# ==========================================================================
print("\nFeedback loop: transformations")
# T1 feature space
F2 = ["study_hours_per_day", "sleep_hours", "exercise_frequency", "leisure_screen_share"]
R["T1_dropped"] = ["attendance_percentage", "social_media_hours", "netflix_hours"]
R["T1_added"] = ["leisure_screen_share"]
R["T1_r_share_study"] = float(train[["leisure_screen_share", "study_hours_per_day"]].corr().iloc[0, 1])

# T2 GPA-conditional score, bin means learned on train only
GPA_BINS = np.r_[np.arange(1.6, 3.95, 0.1), 3.9999, 4.0001]
train["gpa_bin"] = pd.cut(train.previous_gpa, GPA_BINS, include_lowest=True)
test["gpa_bin"] = pd.cut(test.previous_gpa, GPA_BINS, include_lowest=True)
bin_mean = train.groupby("gpa_bin", observed=False).exam_score.mean()
train["adj_score"] = train.exam_score - train.gpa_bin.map(bin_mean).astype(float)
test["adj_score"] = test.exam_score - test.gpa_bin.map(bin_mean).astype(float)
R["T2_n_bins"] = int(bin_mean.notna().sum())
R["T2_r2_score_on_gpa_bins_test"] = float(r2_score(
    test.exam_score, test.gpa_bin.map(bin_mean).astype(float)))
R["T2_adj_sd_train"] = float(train.adj_score.std())

# T3 censoring flag (already in base table: gpa_at_ceiling)
R["T3_share_gpa_ceiling"] = float(train.gpa_at_ceiling.mean())
R["T3_score_ceiling_within_gpa4"] = float(train.loc[train.gpa_at_ceiling == 1,
                                                    "score_at_ceiling"].mean())

# ==========================================================================
# ITERATION 2 - MODEL: K-means re-selected on D2
# ==========================================================================
print("\nIteration 2")
sc2 = StandardScaler().fit(train[F2])
X2 = sc2.transform(train[F2])
sel2 = k_selection(X2, ks, "refined")
sel2.to_csv(OUT / "it2_k_selection.csv", index=False)
R["it2_hopkins"] = hopkins(X2)
R["it2_k_selection"] = sel2.round(4).to_dict("records")
plot_k_selection([sel1, sel2], ["Raw habits (D1)", "Refined habits (D2)"], [PALETTE["raw"], C[0]],
                 "Iteration 2 model selection: after the feature transformation, k = 2 "
                 "is stable (ARI ~ 1) and beats the null", "M1_fig06_k_selection_refined")

K2 = int(sel2.loc[sel2.stability_ari.idxmax(), "k"])
km2 = KMeans(K2, n_init=10, random_state=SEED).fit(X2)
order2 = train.assign(c=km2.labels_).groupby("c").exam_score.mean().sort_values(ascending=False).index
remap2 = {old: new for new, old in enumerate(order2)}
train["cl2"] = pd.Series(km2.labels_).map(remap2).values
test["cl2"] = pd.Series(km2.predict(sc2.transform(test[F2]))).map(remap2).values
cent2_raw = pd.DataFrame(sc2.inverse_transform(km2.cluster_centers_), columns=F2).rename(index=remap2).sort_index()
NAMES2 = ["Study-focused", "Leisure-heavy"] if K2 == 2 else [f"Cluster {i + 1}" for i in range(K2)]
R["it2_k"] = K2
R["it2_cluster_names"] = NAMES2
R["it2_centroids"] = cent2_raw.round(3).to_dict("index")
R["it2_sizes"] = train.cl2.value_counts().sort_index().to_dict()
R["it2_ari_it1_vs_it2"] = float(adjusted_rand_score(train.cl1, train.cl2))

# ---- fig07: refined cluster map ------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axs[0]
s = rng.choice(len(train), 6000, replace=False)
for c in range(K2):
    m = train.cl2.values[s] == c
    ax.scatter(train.study_hours_per_day.values[s][m], train.leisure_screen_share.values[s][m],
               s=9, color=C[c], alpha=0.45, lw=0, label=f"{NAMES2[c]} (n={R['it2_sizes'][c]:,})")
    ax.scatter(cent2_raw.loc[c, "study_hours_per_day"], cent2_raw.loc[c, "leisure_screen_share"],
               s=160, color=C[c], edgecolor="white", lw=2, zorder=5, marker="D")
ax.set(xlabel="Study hours per day", ylabel="Leisure share of screen time",
       title="Refined space: clusters separate along study vs. leisure screen use")
ax.legend(fontsize=8.5, markerscale=2.5, loc="upper right")
ax = axs[1]
z2 = pd.DataFrame(km2.cluster_centers_, columns=F2).rename(index=remap2).sort_index()
w = 0.36
for c in range(K2):
    ax.barh(np.arange(len(F2)) + (c - 0.5) * w, z2.loc[c], height=w, color=C[c],
            label=NAMES2[c], edgecolor=PALETTE["surface"], lw=2)
    for j, f in enumerate(F2):
        ax.text(z2.loc[c, f] + (0.04 if z2.loc[c, f] >= 0 else -0.04), j + (c - 0.5) * w,
                f"{cent2_raw.loc[c, f]:.2f}", va="center", fontsize=8,
                ha="left" if z2.loc[c, f] >= 0 else "right")
ax.axvline(0, color="#b9b8b3", lw=1)
ax.set_yticks(range(len(F2)), [LABEL[f] for f in F2]); ax.invert_yaxis()
ax.set(xlabel="Centroid (z); labels show raw units", xlim=(-1.3, 1.3),
       title="Sleep and exercise do not separate the clusters")
ax.legend(fontsize=8.5, loc="center right")
fig.suptitle(f"Iteration 2 clusters (k={K2}), refit on the transformed habit space",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "M1_fig07_cluster_map_refined")


# ---- statistics: raw vs adjusted gap with bootstrap CIs ------------------
def boot_gap(df, col, n=1000):
    a = df.loc[df.cl2 == 0, col].values
    b = df.loc[df.cl2 == 1, col].values
    g = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def tests(df, col):
    a = df.loc[df.cl2 == 0, col]; b = df.loc[df.cl2 == 1, col]
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b)
    return dict(welch_t=float(t.statistic), welch_p=float(t.pvalue),
                mwu_p=float(u.pvalue), cohens_d=float(cohens_d(a, b)))


no4_tr = train[train.gpa_at_ceiling == 0]
no4_te = test[test.gpa_at_ceiling == 0]
rows = [
    ("Prior GPA, % scale\n(train)", train, "previous_gpa_pct"),
    ("Raw score\n(train)", train, "exam_score"),
    ("Raw score\n(test)", test, "exam_score"),
    ("GPA-adjusted\n(train)", train, "adj_score"),
    ("GPA-adjusted\n(test)", test, "adj_score"),
    ("GPA-adjusted,\nGPA < 4.0 (test)", no4_te, "adj_score"),
]
gaps = []
for lab, df, col in rows:
    est, lo, hi = boot_gap(df, col)
    gaps.append(dict(label=lab.replace("\n", " "), est=est, lo=lo, hi=hi, **tests(df, col)))
gaps = pd.DataFrame(gaps)
gaps.to_csv(OUT / "it2_gap_tests.csv", index=False)
R["it2_gaps"] = gaps.round(5).to_dict("records")

# ---- fig08: raw vs adjusted gap ------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.3))
cols8 = [PALETTE["gpa_band"]["90-99"], PALETTE["raw"], PALETTE["raw"],
         C[0], C[0], C[0]]
y = np.arange(len(gaps))[::-1]
for yi, (_, r), col in zip(y, gaps.iterrows(), cols8):
    ax.plot([0, r.est], [yi, yi], color=col, lw=2, alpha=0.5)
    ax.errorbar(r.est, yi, xerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", ms=9,
                color=col, capsize=4, lw=2)
    ax.text(max(r.hi, 0) + 0.15, yi, f"{r.est:+.2f}  [{r.lo:+.2f}, {r.hi:+.2f}]   d = {r.cohens_d:+.3f}",
            va="center", fontsize=8.5)
ax.axvline(0, color="#7a7975", lw=1)
ax.set_yticks(y, [r[0] for r in rows])
ax.set(xlim=(-1, 9.5), xlabel=f"{NAMES2[0]} minus {NAMES2[1]} (exam-score points, 95% bootstrap CI)")
ax.set_title("Iteration 2 inference: the 4-point habit gap is the prior-GPA gap; "
             "holding GPA fixed it is zero", loc="left")
fig.tight_layout()
save(fig, "M1_fig08_raw_vs_adjusted_gap")

# ---- fig09: within-band gaps, raw vs adjusted (test split) -----------------
band_tab = []
for b in GPA_BAND_LABELS:
    d = test[test.gpa_band == b]
    raw = boot_gap(d, "exam_score")
    adj = boot_gap(d, "adj_score")
    gpa = d.loc[d.cl2 == 0, "previous_gpa"].mean() - d.loc[d.cl2 == 1, "previous_gpa"].mean()
    band_tab.append(dict(band=b, n=len(d), raw=raw[0], raw_lo=raw[1], raw_hi=raw[2],
                         adj=adj[0], adj_lo=adj[1], adj_hi=adj[2], gpa_gap=gpa))
band_tab = pd.DataFrame(band_tab)
R["it2_within_band_test"] = band_tab.round(4).to_dict("records")

fig, ax = plt.subplots(figsize=(10, 4.0))
y = np.arange(len(GPA_BAND_LABELS))[::-1]
for yi, (_, r) in zip(y, band_tab.iterrows()):
    for val, lo, hi, col, dy, lab in [(r.raw, r.raw_lo, r.raw_hi, PALETTE["raw"], 0.14, "Raw score"),
                                      (r.adj, r.adj_lo, r.adj_hi, C[0], -0.14, "GPA-adjusted score")]:
        ax.errorbar(val, yi + dy, xerr=[[val - lo], [hi - val]], fmt="o", ms=8,
                    color=col, capsize=3, lw=2, label=lab if yi == y[0] else None)
    ax.text(3.35, yi, f"raw {r.raw:+.2f}  ->  adjusted {r.adj:+.2f}\n"
            f"prior-GPA gap inside band {r.gpa_gap:+.3f}", va="center", fontsize=8.5,
            color=PALETTE["text_secondary"])
ax.axvline(0, color="#7a7975", lw=1)
ax.set_yticks(y, [f"{b}\n(n={n:,})" for b, n in zip(band_tab.band, band_tab.n)])
ax.set(xlim=(-1.0, 7.4), xlabel=f"{NAMES2[0]} minus {NAMES2[1]} (points, test split, 95% bootstrap CI)",
       ylabel="Prior-GPA band")
ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 1.0))
ax.set_title("Iteration 2: the 2.5-point gap left in the lowest band is GPA spread inside the "
             "band; it vanishes after adjustment", loc="left", fontsize=10.5)
fig.tight_layout()
save(fig, "M1_fig09_within_band_refined")

# ==========================================================================
# ITERATION 2 - SUPERVISED CHECK
# ==========================================================================
print("  gradient boosting check")
targets = [("exam_score", "Exam score"), ("previous_gpa", "Prior GPA"),
           ("adj_score", "GPA-adjusted score")]
sup = {}
imps = {}
for col, lab in targets:
    m = HistGradientBoostingRegressor(random_state=SEED, max_iter=200).fit(train[F2], train[col])
    sup[lab] = float(r2_score(test[col], m.predict(test[F2])))
    if col in ("previous_gpa", "adj_score"):
        pi = permutation_importance(m, test[F2], test[col], n_repeats=5,
                                    random_state=SEED, scoring="r2")
        imps[lab] = pd.Series(pi.importances_mean, index=F2)
R["it2_gbm_test_r2"] = {k: round(v, 4) for k, v in sup.items()}
R["it2_gbm_perm_importance"] = {k: v.round(4).to_dict() for k, v in imps.items()}

fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.2]})
ax = axs[0]
labs = list(sup)
cols10 = [PALETTE["raw"], PALETTE["gpa_band"]["90-99"], C[0]]
ax.barh(range(3), [sup[l] for l in labs], color=cols10, height=0.6)
for i, l in enumerate(labs):
    ax.text(max(sup[l], 0) + 0.003, i, f"R² = {(0.0 if abs(sup[l]) < 5e-4 else sup[l]):.3f}",
            va="center", fontsize=9)
ax.set_yticks(range(3), labs); ax.invert_yaxis()
ax.set(xlim=(-0.01, 0.12), xlabel="Test-set R² from the four refined habits",
       title="Habits predict prior GPA, not the exam beyond it")
ax = axs[1]
w = 0.38
for i, (lab, col) in enumerate([("Prior GPA", PALETTE["gpa_band"]["90-99"]),
                                ("GPA-adjusted score", C[0])]):
    ax.barh(np.arange(len(F2)) + (i - 0.5) * w, imps[lab].values, height=w, color=col,
            label=lab, edgecolor=PALETTE["surface"], lw=2)
ax.set_yticks(range(len(F2)), [LABEL[f] for f in F2]); ax.invert_yaxis()
ax.axvline(0, color="#b9b8b3", lw=1)
ax.set(xlabel="Permutation importance (drop in test R²)",
       title="Which habits carry the prior-GPA signal")
ax.legend(fontsize=8.5, loc="lower right")
fig.suptitle("Iteration 2 supervised check (gradient boosting, fitted on train, scored on test)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "M1_fig10_supervised_check")

# ==========================================================================
# SAVE
# ==========================================================================
keep = ["student_id", "split", "gpa_band", "previous_gpa", "exam_score",
        "adj_score", "cl1", "cl2"] + F1 + ["leisure_screen_share"]
assign = pd.concat([train[keep], test[keep]]).sort_values("student_id")
assign["cl1"] += 1
assign["cluster_it2_name"] = assign.cl2.map(dict(enumerate(NAMES2)))
assign.rename(columns={"cl1": "cluster_it1", "cl2": "cluster_it2"}).to_csv(
    OUT / "m1_cluster_assignments.csv.gz", index=False)


def tidy(o):
    if isinstance(o, dict):
        return {str(k): tidy(v) for k, v in o.items()}
    if isinstance(o, list):
        return [tidy(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


(OUT / "m1_results.json").write_text(json.dumps(tidy(R), indent=2))
print("\nSaved outputs to", OUT.relative_to(ROOT))
