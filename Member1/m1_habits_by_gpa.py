"""Tushaar's assigned analysis. Run from the repo root after preprocess_a3.py.

Fit unsupervised models on training habits only. Use exam scores only for
held-out comparisons, and previous GPA only for stratification. Cluster IDs
are local to a model; order by training centroids' mean study hours for display.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a3_common import GPA_BAND_LABELS, HABIT_NUMERIC as NUMERIC, IMAGES_DIR, PALETTE, ROOT, SEED, load_base
NUM_LABELS = ["Study\nh/day", "Sleep\nh/night", "Exercise\ntimes/wk", "Leisure\nshare", "Attend.\n%"]
K_VALUES = np.arange(1, 9)
MODELS = ["K-means", "GMM"]
GLOBAL = "All students"
SCOPES = [GLOBAL, *GPA_BAND_LABELS]
COLORS = PALETTE["cluster"]
OUTPUT = ROOT / "Member1" / "outputs"


def encode(train, test, env):
    """One unit of training variance per numeric feature and for environment in total."""
    scaler = StandardScaler().fit(train[NUMERIC])
    env_mean = train[env].mean().to_numpy()
    env_scale = np.sqrt(train[env].var(ddof=0).sum())
    if env_scale == 0:
        env_scale = 1.0
    def transform(rows):
        return np.column_stack([scaler.transform(rows[NUMERIC]),
                                (rows[env].to_numpy() - env_mean) / env_scale])
    return transform(train), transform(test)


def fit_scope(scope, train, test, env):
    x, xt = encode(train, test, env)
    rng = np.random.default_rng(SEED)
    sil_ids = rng.choice(len(train), size=min(1800, len(train)), replace=False)
    selection, fits = [], {}
    for k in K_VALUES:
        model = KMeans(n_clusters=int(k), n_init=10, random_state=SEED).fit(x)
        fits[k] = model
        sil = np.nan if k == 1 else silhouette_score(x[sil_ids], model.labels_[sil_ids])
        selection.append(dict(scope=scope, k=k, inertia_per_train=model.inertia_/len(x),
                              silhouette=sil))
    curve = pd.DataFrame(selection)
    inertia = curve.inertia_per_train.to_numpy()
    # Largest vertical distance below the chord between normalized endpoints.
    reduction = (inertia[0] - inertia) / (inertia[0] - inertia[-1])
    elbow = int(K_VALUES[np.argmax(reduction - (K_VALUES-1)/(K_VALUES[-1]-1))])
    tied = curve.loc[curve.silhouette >= curve.silhouette.max() - .01, "k"]
    k = int(min(tied, key=lambda n: (abs(n - elbow), n)))
    curve["elbow"] = elbow
    curve["chosen_k"] = k
    # Diagonal Gaussian mixtures approximate this mixed numeric/indicator space.
    # Bounded covariance prevents zero-variance indicator dimensions from collapsing.
    gmm = GaussianMixture(k, covariance_type="diag", reg_covar=.01,
                          n_init=5, max_iter=250, random_state=SEED).fit(x)
    if not gmm.converged_:
        raise RuntimeError(f"GMM did not converge for {scope}")
    assigned = test[["student_id", "gpa_band", "exam_score"]].copy()
    assigned["scope"] = scope
    profiles, tests = [], []
    for name, model in zip(MODELS, [fits[k], gmm]):
        train_labels = model.predict(x)
        order = train.groupby(train_labels)["study_hours_per_day"].mean().sort_values().index
        if len(order) != k:
            raise ValueError(f"Empty training group: {scope}, {name}")
        lookup = np.empty(k, dtype=int)
        lookup[order] = np.arange(1, k + 1)
        labels = lookup[model.predict(xt)]
        assigned[name] = labels
        if len(np.unique(labels)) != k:
            raise ValueError(f"Empty held-out group: {scope}, {name}")
        for cluster in range(1, k+1):
            group = test.loc[labels == cluster]
            means = group[NUMERIC + env].mean().to_dict()
            profiles.append(dict(scope=scope, model=name, cluster=cluster,
                                 n_test=len(group), mean_score=group.exam_score.mean(),
                                 median_score=group.exam_score.median(), **means))
        groups = [test.loc[labels == c, "exam_score"] for c in range(1,k+1)]
        h, p = kruskal(*groups)
        tests.append(dict(scope=scope, model=name, n_train=len(train), n_test=len(test),
                          k=k, H=h, df=k-1, p=p,
                          eta2_H=max(0., (h-k+1)/(len(test)-k))))
    pca = PCA(n_components=2, svd_solver="full").fit(x)
    # Display the same random subset for both models; fit and tests used all rows.
    ids = np.sort(rng.choice(len(xt), min(1400, len(xt)), replace=False))
    projection = pca.transform(xt[ids])
    score = adjusted_rand_score(assigned["K-means"], assigned["GMM"])
    return dict(scope=scope, curve=curve, assigned=assigned, profiles=profiles,
                tests=tests, k=k, elbow=elbow, ari=score, ids=ids, xy=projection,
                explained=pca.explained_variance_ratio_,
                env_scale_ref=train[NUMERIC+env].std(ddof=0),
                env_mean_ref=train[NUMERIC+env].mean())


def holm(p_values):
    p = np.asarray(p_values)
    order = np.argsort(p)
    corrected = np.empty(len(p))
    corrected[order] = np.minimum(1, np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)))
    return corrected


def compare_gaps(global_run, band_runs):
    """Use the high-minus-low study-time clusters, chosen without score information.

    Keep global labels fixed for direct standardization. For the band-refit
    contrast, rank local labels anew by training study time and combine only
    their signed contrasts, weighting by the held-out band shares. Bootstrap
    resamples the same students within each band for all three estimates,
    conditional on the fitted models and observed composition.
    """
    whole = global_run["assigned"].set_index("student_id")
    local = pd.concat([run["assigned"] for run in band_runs]).set_index("student_id")
    assert whole.index.is_unique and local.index.is_unique and set(whole.index)==set(local.index)
    rng = np.random.default_rng(SEED)
    rows = []
    for alg in MODELS:
        max_g = global_run["k"]
        parts = []
        for run in band_runs:
            a = whole.loc[whole.gpa_band == run["scope"]]
            parts.append((a.exam_score.to_numpy(), a[alg].to_numpy(),
                          local.loc[a.index, alg].to_numpy(), run["k"]))
        def evaluate(resample):
            global_n = np.zeros(2)
            global_sum = np.zeros(2)
            standardized = refit = 0.
            local_gaps = []
            for score, labels, local_labels, max_l in parts:
                if resample:
                    idx = rng.integers(len(score), size=len(score))
                    score, labels, local_labels = score[idx], labels[idx], local_labels[idx]
                masks = [labels == 1, labels == max_g]
                counts = np.array([m.sum() for m in masks])
                sums = np.array([score[m].sum() for m in masks])
                if not counts.all():
                    raise ValueError("Cannot standardize a missing band × cluster cell")
                global_n += counts
                global_sum += sums
                w = len(score)/len(whole)
                standardized += w*np.diff(sums/counts)[0]
                gap = score[local_labels==max_l].mean() - score[local_labels==1].mean()
                refit += w*gap
                local_gaps.append(gap)
            raw = np.diff(global_sum/global_n)[0]
            return [raw, standardized, refit, raw-standardized, *local_gaps]
        point = evaluate(False)
        ci = np.quantile([evaluate(True) for _ in range(1000)], [.025,.975], axis=0)
        for i, label in enumerate(["Global", "Same labels, common GPA mix", "Refit in each GPA band",
                                   "Global minus standardized", *GPA_BAND_LABELS]):
            rows.append(dict(model=alg, comparison=label, gap=point[i], lo=ci[0,i], hi=ci[1,i]))
    return pd.DataFrame(rows)


def save(fig, name):
    fig.savefig(IMAGES_DIR / f"M1_gpa_{name}.png", dpi=165, bbox_inches="tight")
    plt.close(fig)


def plot_selection(runs):
    fig, axes = plt.subplots(2, len(runs), figsize=(15, 5), layout="constrained")
    for j, run in enumerate(runs):
        d = run["curve"]
        for i, (col, label) in enumerate([("inertia_per_train", "Inertia per student"),
                                         ("silhouette", "Mean silhouette")]):
            ax = axes[i,j]
            ax.plot(d.k, d[col], "o-", c="#586f83", markersize=4)
            k = run["k"]
            ax.scatter(k, d.loc[d.k == k,col], s=74, c="#c54b21", zorder=3)
            if i == 0:
                ax.axvline(run["elbow"], color="#888", ls=":", zorder=0)
                ax.set_title(run["scope"])
            else:
                ax.set_xlabel("Number of clusters")
            if j==0:
                ax.set_ylabel(label)
            ax.set_xticks([1,2,3,4,5,6,7,8])
    fig.suptitle("Choosing k on training habits  |  dot = selected k, dotted line = elbow", fontsize=15)
    save(fig, "selection")


def plot_pca(runs, suffix):
    fig, axes = plt.subplots(len(runs), 2, squeeze=False, figsize=(9, 3.25*len(runs)),
                             layout="constrained")
    for axes_row, run in zip(axes, runs):
        for ax, model in zip(axes_row, MODELS):
            labels = run["assigned"][model].to_numpy()[run["ids"]]
            for c in range(1, run["k"] + 1):
                points = run["xy"][labels == c]
                ax.scatter(*points.T, s=8, color=COLORS[c-1], alpha=.55, label=f"C{c}", linewidths=0)
            v1, v2 = run["explained"]*100
            ax.set(xlabel=f"PC1 ({v1:.0f}% variance)", ylabel=f"PC2 ({v2:.0f}% variance)",
                   title=f"{run['scope']}  ·  {model}")
            ax.legend(ncol=min(4, run["k"]), fontsize=8, loc="upper right")
    save(fig, f"pca_{suffix}")


def plot_scores(runs, suffix):
    fig, axes = plt.subplots(len(runs), 2, squeeze=False, figsize=(9, 2.65*len(runs)),
                             layout="constrained", sharey=True)
    for axes_row, run in zip(axes, runs):
        for ax, model in zip(axes_row, MODELS):
            a = run["assigned"]
            groups = [a.loc[a[model]==n, "exam_score"] for n in range(1,run["k"]+1)]
            boxes = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=.6,
                               medianprops={"color":"#151515", "linewidth":1.6})
            for box, color in zip(boxes["boxes"], COLORS):
                box.set(facecolor=color, alpha=.65)
            ax.set_xticks(range(1,run["k"]+1), [f"C{i+1}\nn={len(g):,}" for i,g in enumerate(groups)])
            ax.set(title=f"{run['scope']}  ·  {model}", ylabel="Exam score (points)", ylim=(32,103))
    save(fig, f"scores_{suffix}")


def plot_profiles(runs, env, suffix, model):
    cols = NUMERIC+env
    names = NUM_LABELS+[e.removeprefix("oh_study_environment_").replace("Co-Learning Group","Co-learn").replace("Quiet Room","Quiet rm.") for e in env]
    data = []
    for run in runs:
        for row in run["profiles"]:
            if row["model"] == model:
                avg = pd.Series({f:row[f] for f in cols})
                effect = (avg - run["env_mean_ref"])/run["env_scale_ref"].replace(0,np.nan)
                data.append((run["scope"], row["cluster"], avg.to_numpy(), effect.to_numpy()))
    fig, ax = plt.subplots(figsize=(11, 1.8+.51*len(data)), layout="constrained")
    grid = ax.imshow(np.stack([r[3] for r in data]), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    for i,(scope,c,mean,z) in enumerate(data):
        for j,v in enumerate(mean):
            label = f"{v:.1f}" if j<5 and j!=3 else (f"{v:.2f}" if j==3 else f"{100*v:.0f}%")
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if abs(z[j])>1 else "#121212")
        if i and data[i-1][0]!=scope: ax.axhline(i-.5, color="white", lw=3)
    ax.set_yticks(range(len(data)), [f"{s}  C{c}" for s,c,*_ in data])
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_title(f"{model} habit profiles on held-out students  |  cells show means or shares")
    ax.grid(False)
    fig.colorbar(grid, ax=ax, shrink=.85, label="SDs from training mean\nwithin the fitting population")
    prefix = "profiles" if model == "K-means" else "profiles_gmm"
    save(fig, f"{prefix}_{suffix}")


def plot_mix_and_gap(mix, gaps):
    table = mix.loc[mix.model == "K-means"]
    shares = table.pivot(index="cluster", columns="gpa_band", values="within_cluster_pct")[GPA_BAND_LABELS]
    counts = table.pivot(index="cluster", columns="gpa_band", values="n_test")[GPA_BAND_LABELS]
    fig, ax = plt.subplots(figsize=(7.8, 2+.6*len(shares)), layout="constrained")
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=shares.to_numpy().max(), aspect="auto")
    for i in range(len(shares)):
        for j in range(len(shares.columns)):
            v = shares.iloc[i,j]
            ax.text(j,i,f"{v:.1f}% ({counts.iloc[i,j]:,})",ha="center",va="center",
                    color="white" if v>.6*shares.to_numpy().max() else "#132238")
    ax.set_xticks(range(4),GPA_BAND_LABELS)
    ax.set_yticks(range(len(shares)), [f"C{c}" for c in shares.index])
    ax.set(title="Prior GPA composition of each global K-means cluster",xlabel="Prior GPA band (percent of maximum)",ylabel="Global cluster")
    ax.grid(False)
    save(fig,"mix")
    sub = gaps.loc[gaps.model=="K-means"].iloc[[0,1,2,4,5,6,7]]
    fig, ax = plt.subplots(figsize=(8,4.2),layout="constrained")
    for i, r in enumerate(sub.itertuples()):
        ax.plot([r.lo,r.hi],[i,i],color="#325779",lw=2)
        ax.scatter(r.gap,i,c="#b54a24" if i==0 else "#325779",s=32,zorder=3)
    ax.axvline(0,color="#777",ls=":")
    ax.set_yticks(range(len(sub)),sub.comparison)
    ax.invert_yaxis()
    ax.set(xlabel="Higher-study minus lower-study cluster mean exam score (points)",
           title="Held-out score gaps with 95% bootstrap intervals")
    save(fig,"gaps")


def main():
    OUTPUT.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    df = load_base()
    env = sorted(c for c in df if c.startswith("oh_study_environment_"))
    if not env or not (df[env].sum(axis=1)==1).all():
        raise ValueError("Expected one active study-environment indicator per student")
    runs = []
    for scope in SCOPES:
        rows = df if scope==GLOBAL else df.loc[df.gpa_band==scope]
        train = rows.loc[rows.split=="train"]
        test = rows.loc[rows.split=="test"]
        run = fit_scope(scope,train,test,env)
        print(f"{scope}: k={run['k']}, elbow={run['elbow']}, held-out ARI={run['ari']:.3f}", flush=True)
        runs.append(run)
    def write(name, frame):
        frame.to_csv(OUTPUT/f"gpa_{name}.csv",index=False,float_format="%.10g")
    write("selection",pd.concat([r["curve"] for r in runs]))
    write("profiles",pd.DataFrame([row for run in runs for row in run["profiles"]]))
    tests = pd.DataFrame([dict(**row, kmeans_gmm_ari=run["ari"]) for run in runs for row in run["tests"]])
    tests["p_holm_within_model"] = np.nan
    for model in MODELS:
        mask = (tests.scope!=GLOBAL)&(tests.model==model)
        tests.loc[mask,"p_holm_within_model"] = holm(tests.loc[mask,"p"])
    write("score_tests",tests)
    all_assigned = pd.concat([r["assigned"] for r in runs],ignore_index=True)
    all_assigned.to_csv(OUTPUT/"gpa_test_assignments.csv.gz",index=False,
                        compression={"method":"gzip","mtime":0})
    mix = []
    for model in MODELS:
        by = runs[0]["assigned"].groupby([model,"gpa_band"],observed=False)
        tab = by.exam_score.agg(n_test="size",mean_score="mean").reset_index().rename(columns={model:"cluster"})
        tab["within_cluster_pct"] = 100*tab.n_test/tab.groupby("cluster").n_test.transform("sum")
        tab["model"] = model
        mix.append(tab)
    mix = pd.concat(mix,ignore_index=True)
    write("mix",mix)
    gaps = compare_gaps(runs[0],runs[1:])
    write("gaps",gaps)
    plot_selection(runs)
    for selected,suffix in [(runs[:1],"global"),(runs[1:],"bands")]:
        plot_pca(selected,suffix)
        plot_scores(selected,suffix)
        for model in MODELS:
            plot_profiles(selected,env,suffix,model)
    plot_mix_and_gap(mix,gaps)
    print(gaps.to_string(index=False),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
