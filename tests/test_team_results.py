"""Check that Member 2 and Member 3 outputs describe the shared cohorts."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from a3_common import load_base

ROOT = Path(__file__).resolve().parents[1]


class TeamEvidence(unittest.TestCase):
    def test_classification_counts_and_removed_features(self):
        test = load_base("test")
        folder = ROOT / "Member2/outputs"
        meta = json.loads((folder / "m2_results.json").read_text())
        metrics = pd.read_csv(folder / "model_metrics.csv")
        self.assertEqual(len(metrics), 6)
        positives = test.dropout_risk.sum()
        np.testing.assert_array_equal(metrics.tp + metrics.fn, np.full(6, positives))
        np.testing.assert_array_equal(metrics.tn + metrics.fp, np.full(6, len(test) - positives))
        self.assertEqual(set(meta["predictors_before"]) - set(meta["predictors_after"]),
                         {"stress_level", "motivation_level"})
        self.assertTrue(set(meta["predictors_after"]).isdisjoint(
            {"dropout_risk", "exam_anxiety_score", "z_stress_level", "z_motivation_level"}))
        tree = metrics.query("iteration == 'iteration 1' and model == 'Decision tree'").iloc[0]
        self.assertEqual(tree.fp + tree.fn, 0)

    def test_regression_cohorts_and_raw_tutoring_gaps(self):
        train = load_base("train")
        results = json.loads((ROOT / "Member3/outputs/m3_results.json").read_text())
        self.assertEqual(results["iter1_n_train"], len(train))
        self.assertEqual(results["T1_rows_kept"], (train.previous_gpa < 4).sum())
        for row in results["tutoring_gap_by_income"]:
            band = train[train.family_income_range == row["income"].split()[0]]
            self.assertEqual(len(band), row["n"])
            means = band.groupby("access_to_tutoring").exam_score.mean()
            self.assertAlmostEqual(means[1] - means[0], row["raw_diff"], places=4)
