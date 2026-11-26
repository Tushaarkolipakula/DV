"""
Member 2 (Advaya Bhardwaj) - What students feel: classification of dropout risk.
Run from the repository root:
    python preprocess_a3.py                       # once, creates data/a3_base.csv.gz
    python Member2/m2_classification_workflow.py

Workflow (Keim diamond, unrolled):
    ITERATION 1  D1 = all features            -> models -> PR/ROC, tree -> K1 (leakage)
    FEEDBACK     T  = remove stress_level and motivation_level (the rule's inputs)
    ITERATION 2  D2 = leak-free features      -> models -> PR before/after,
                                                 importance, proxy check -> K2
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a3_common import (load_base, load_meta, FEATURES, PALETTE, ROOT, SEED,
                       IMAGES_DIR, GPA_BAND_LABELS, set_style)

OUT = ROOT / "Member2" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
set_style()
CAT = PALETTE["cluster"]
R = {}


def save(fig, name):
    """Save as images/M2_<name>.png (the report already expects these names)."""
    fig.savefig(IMAGES_DIR / f"M2_{name}.png")
    plt.close(fig)
    print("  saved", f"M2_{name}")


def models():
    """The three classifiers. class_weight='balanced' handles the 2% positives."""
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        # Suppress numerical splits in leaves that already contain a single class.
        "Decision tree": DecisionTreeClassifier(
            max_depth=3, min_impurity_decrease=1e-10, class_weight="balanced", random_state=SEED),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
    }


def evaluate(feats, tag):
    """Fit every model on train, score on test. Returns metrics and fitted models."""
    rows, fitted, probs = [], {}, {}
    for name, m in models().items():
        m.fit(train[feats], train[TARGET])
        p = m.predict_proba(test[feats])[:, 1]
        rows.append(dict(iteration=tag, model=name,
                         pr_auc=average_precision_score(test[TARGET], p),
                         roc_auc=roc_auc_score(test[TARGET], p),
                         f1_at_0_5=f1_score(test[TARGET], p >= 0.5)))
        counts = confusion_matrix(test[TARGET], p >= 0.5, labels=[0, 1]).ravel()
        rows[-1].update(zip(("tn", "fp", "fn", "tp"), counts))
        fitted[name], probs[name] = m, p
        print(f"  [{tag}] {name:20s} PR-AUC {rows[-1]['pr_auc']:.3f}  "
              f"ROC-AUC {rows[-1]['roc_auc']:.3f}  F1 {rows[-1]['f1_at_0_5']:.3f}")
    return pd.DataFrame(rows), fitted, probs


# ==========================================================================
# DATA
# ==========================================================================
train, test = load_base("train"), load_base("test")
meta = load_meta()
TARGET = "dropout_risk"
ALL = (FEATURES["habits"] + FEATURES["wellbeing"] + FEATURES["background"]
       + FEATURES["prior"] + meta["one_hot_columns"])
R["base_rate_test"] = float(test[TARGET].mean())
print(f"Train {len(train):,}  Test {len(test):,}  positive rate {R['base_rate_test']:.2%}")

# ==========================================================================
# ITERATION 1 - all features
# ==========================================================================
print("\nIteration 1")

res1, fit1, prob1 = evaluate(ALL, "iteration 1")


tree2 = fit1["Decision tree"]
print(export_text(tree2, feature_names=ALL))
R["iter1_tree_rules"] = export_text(tree2, feature_names=ALL)

# ==========================================================================
# FEEDBACK LOOP - transformation T1
# ==========================================================================
LEAK = ["stress_level", "motivation_level"]   # inputs of the dropout rule
D2 = [f for f in ALL if f not in LEAK]
R["T1_removed"] = LEAK

# ==========================================================================
# ITERATION 2 - leak-free features
# ==========================================================================
print("\nIteration 2")
res2, fit2, prob2 = evaluate(D2, "iteration 2")


pi = permutation_importance(fit2["Logistic regression"], test[D2], test[TARGET],
                            n_repeats=5, random_state=SEED, scoring="roc_auc")
imp = pd.Series(pi.importances_mean, index=D2).sort_values(ascending=False)
print("\nTop leak-free features (drop in test ROC-AUC):\n", imp.head(8).round(4))
R["iter2_perm_importance_top8"] = imp.head(8).round(5).to_dict()

R["corr_gpa_motivation"] = float(train[["previous_gpa", "motivation_level"]].corr().iloc[0, 1])
R["corr_mental_stress"] = float(train[["mental_health_rating", "stress_level"]].corr().iloc[0, 1])

# ==========================================================================
# Visual evidence
def panels(rows=1, cols=1, size=(7, 4)):
    return plt.subplots(rows, cols, figsize=size, layout="constrained")


def prevalence_line(ax):
    ax.axhline(R["base_rate_test"], color=PALETTE["raw"], ls=":",
               label=f"Flag prevalence {R['base_rate_test']:.2%}")


def pr_curve(ax, scores, label, color, style="-"):
    precision, recall, _ = precision_recall_curve(test[TARGET], scores)
    ax.step(recall, precision, where="post", label=label, color=color, ls=style)
    ax.set(xlabel="Recall", ylabel="Precision", ylim=(-.025, 1.05))


# Right-closed bins put 8.0 below the rule and 8.1 above it.
rates = train.pivot_table(index=np.ceil(train.stress_level).astype(int),
                          columns="motivation_level", values=TARGET, aggfunc="mean")
fig, ax = panels(size=(7, 5))
ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.grid(False)
ax.set(xticks=range(10), xticklabels=range(1, 11), yticks=range(10),
       yticklabels=["1"] + [f"({i-1}, {i}]" for i in range(2, 11)],
       xlabel="Motivation score", ylabel="Stress score interval",
       title="Only high stress with low motivation triggers the flag")
for (r, c), value in np.ndenumerate(rates.to_numpy()):
    ax.text(c, r, f"{value:.0%}" if pd.notna(value) else "—", ha="center", va="center",
            fontsize=8, color="white" if value > .5 else PALETTE["text"])
ax.add_patch(plt.Rectangle((-.49, 7.5), 2.98, 1.98, fill=False, ec=CAT[1], lw=2))
save(fig, "fig01_rule_heatmap")

fig, (left, right) = panels(cols=2, size=(11, 4))
for (name, scores), color, style in zip(prob1.items(), CAT, ["-", "--", ":"]):
    row = res1.set_index("model").loc[name]
    pr_curve(left, scores, f"{name} · {row.pr_auc:.3f}", color, style)
    fpr, tpr, _ = roc_curve(test[TARGET], scores)
    right.plot(fpr, tpr, color=color, ls=style, label=f"{name} · {row.roc_auc:.3f}")
prevalence_line(left)
left.set_title("Tree and forest attain perfect AP")
right.set(title="All three rank positives highly", xlabel="False positive rate", ylabel="True positive rate")
left.legend(title="Average precision")
right.legend(title="ROC area")
save(fig, "fig02_iter1_pr_roc")

fig, (left, right) = panels(cols=2, size=(13, 5))
plot_tree(tree2, feature_names=ALL, class_names=["Clear", "Flagged"], impurity=False,
          label="root", precision=2, ax=left, fontsize=8)
left.set_title("The fitted tree recovers the rule")
forest_imp = pd.Series(fit1["Random forest"].feature_importances_, index=ALL)
forest_imp.nlargest(10).sort_values().plot.barh(ax=right, color=CAT[0])
right.set(title="Two features dominate the forest", xlabel="Share of impurity decrease")
save(fig, "fig03_iter1_tree")

fig, axes = panels(cols=3, size=(14, 4))
for ax, name in zip(axes, prob1):
    for label, scores, color in [("Before", prob1[name], PALETTE["raw"]),
                                 ("After", prob2[name], PALETTE["adjusted"])]:
        ap = average_precision_score(test[TARGET], scores)
        pr_curve(ax, scores, f"{label} · AP {ap:.3f}", color)
    prevalence_line(ax)
    ax.set_title(name)
    ax.legend(fontsize=8)
fig.suptitle("Removing the rule inputs reduces precision across models")
save(fig, "fig04_iter2_pr_before_after")

fig, ax = panels()
imp.head(10).sort_values().plot.barh(ax=ax, color=CAT[0])
ax.set(title="GPA and mental health carry the strongest remaining signals",
       xlabel="Mean decrease in test ROC area after shuffling")
save(fig, "fig05_iter2_importance")

fig, axes = panels(cols=2, size=(10, 4))
for ax, groups, xlabel in zip(axes, [test.gpa_band, test.mental_health_rating.round().astype(int)],
                              ["Prior GPA band (% of maximum)", "Mental health rating (rounded)"]):
    rate = test.groupby(groups, observed=True)[TARGET].mean().mul(100)
    if groups.name == "gpa_band":
        rate = rate.reindex(GPA_BAND_LABELS)
    rate.plot.bar(ax=ax, color=CAT[0], rot=0)
    ax.set(xlabel=xlabel, ylabel="Flagged test students (%)")
fig.suptitle("Flag prevalence varies across prior GPA and mental health")
save(fig, "fig06_proxy_check")

fig, axes = panels(rows=2, cols=3, size=(11, 7))
for ax, row in zip(axes.flat, pd.concat([res1, res2]).itertuples()):
    counts = np.array([[row.tn, row.fp], [row.fn, row.tp]])
    ax.imshow(counts / counts.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for (r, c), count in np.ndenumerate(counts):
        ax.text(c, r, f"{count:,}", ha="center", va="center",
                color="white" if count > counts[r].sum() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Clear", "Flagged"],
           yticklabels=["Clear", "Flagged"], xlabel="Predicted", ylabel="Actual",
           title=f"{row.model} · {row.iteration[-1]}")
fig.suptitle("Held-out classifications at the 0.5 threshold")
save(fig, "fig07_confusion_matrices")

R["test_positives"] = int(test[TARGET].sum())
R["predictors_before"] = ALL
R["predictors_after"] = D2
# SAVE
# ==========================================================================
pd.concat([res1, res2]).to_csv(OUT / "model_metrics.csv", index=False)
R["metrics"] = pd.concat([res1, res2]).round(4).to_dict("records")
(OUT / "m2_results.json").write_text(json.dumps(R, indent=2, default=float))
print("\nSaved Member2/outputs/model_metrics.csv and m2_results.json")
