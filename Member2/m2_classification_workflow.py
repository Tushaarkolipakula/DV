"""
Member 2 (Advaya Bhardwaj) - What students feel: classification of dropout risk.
STARTER SCRIPT. It already runs end to end and prints the iteration-1 and
iteration-2 model metrics. Your job: make the figures marked TODO, read the
results, and write Section 5 of the report. See TEAM_INSTRUCTIONS.md, Part B.

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
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a3_common import (load_base, load_meta, FEATURES, PALETTE, SEED,  # noqa: E402
                       IMAGES_DIR, GPA_BAND_LABELS, set_style)

OUT = ROOT / "Member2" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
set_style()
CAT = PALETTE["cluster"]       # categorical colours, use in this order (never name this C: it
                               # would hide the C() used in statsmodels formulas)
R = {}                          # put every number you quote in the report here


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
        "Decision tree": DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", random_state=SEED),
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
# TODO fig01: heatmap of dropout rate for every stress x motivation cell (train).
#   Round stress to integers; pivot_table(values=TARGET, index=stress, columns=motivation).
#   Sequential blue colormap; annotate the rate; draw the rule box (stress > 8, motivation <= 3).
#   save(fig, "fig01_rule_heatmap")

res1, fit1, prob1 = evaluate(ALL, "iteration 1")

# TODO fig02: two panels, PR curves (left) and ROC curves (right) for the 3 models.
#   Use precision_recall_curve / roc_curve on test[TARGET] and prob1[name].
#   Add a horizontal dashed line at the base rate on the PR panel. Label each
#   curve with its AUC. save(fig, "fig02_iter1_pr_roc")

tree2 = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(train[ALL], train[TARGET])
print(export_text(tree2, feature_names=ALL))
R["iter1_tree_rules"] = export_text(tree2, feature_names=ALL)
# TODO fig03: left = plot_tree(tree2, feature_names=ALL, filled=True, ax=ax);
#   right = top-10 feature_importances_ of the random forest (fit1["Random forest"]).
#   save(fig, "fig03_iter1_tree")

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

# TODO fig04: PR curves of the best model, iteration 1 vs iteration 2, with the
#   base-rate line. One axis. save(fig, "fig04_iter2_pr_before_after")

pi = permutation_importance(fit2["Logistic regression"], test[D2], test[TARGET],
                            n_repeats=5, random_state=SEED, scoring="roc_auc")
imp = pd.Series(pi.importances_mean, index=D2).sort_values(ascending=False)
print("\nTop leak-free features (drop in test ROC-AUC):\n", imp.head(8).round(4))
R["iter2_perm_importance_top8"] = imp.head(8).round(5).to_dict()
# TODO fig05: horizontal bar chart of imp.head(10). save(fig, "fig05_iter2_importance")

# TODO fig06: two panels, dropout rate by gpa_band (GPA_BAND_LABELS order) and
#   by rounded mental_health_rating. These show WHY some signal remains:
#   prior GPA correlates with motivation, and mental health drops at stress 7.
#   save(fig, "fig06_proxy_check")
R["corr_gpa_motivation"] = float(train[["previous_gpa", "motivation_level"]].corr().iloc[0, 1])
R["corr_mental_stress"] = float(train[["mental_health_rating", "stress_level"]].corr().iloc[0, 1])

# ==========================================================================
# SAVE
# ==========================================================================
pd.concat([res1, res2]).to_csv(OUT / "model_metrics.csv", index=False)
R["metrics"] = pd.concat([res1, res2]).round(4).to_dict("records")
(OUT / "m2_results.json").write_text(json.dumps(R, indent=2, default=float))
print("\nSaved Member2/outputs/model_metrics.csv and m2_results.json")
