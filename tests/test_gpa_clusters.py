"""Cross-check Member 1's published tables against the fixed test split and scores."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a3_common import GPA_BAND_LABELS, load_base


class TestGpaClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = ROOT / "Member1" / "outputs"
        cls.assignments = pd.read_csv(out / "gpa_test_assignments.csv.gz")
        cls.held_out = load_base("test").set_index("student_id")
        cls.gaps = pd.read_csv(out / "gpa_gaps.csv").set_index(["model", "comparison"])

    def test_assignments_cover_each_test_student_exactly_twice(self):
        for global_fit in (True, False):
            a = self.assignments.loc[(self.assignments.scope == "All students") == global_fit]
            a = a.set_index("student_id").sort_index()
            ref = self.held_out.sort_index()
            self.assertTrue(a.index.equals(ref.index))
            np.testing.assert_array_equal(a.exam_score, ref.exam_score)
            np.testing.assert_array_equal(a.gpa_band, ref.gpa_band.astype(str))
            if not global_fit:
                np.testing.assert_array_equal(a.scope, a.gpa_band)

    def test_score_contrasts_use_common_weights_and_local_clusters(self):
        pooled = self.assignments[self.assignments.scope == "All students"]
        local = self.assignments[self.assignments.scope != "All students"]
        for model in ("K-means", "GMM"):
            def contrast(rows):
                groups = rows.groupby(model).exam_score.mean().sort_index()
                return groups.iloc[-1] - groups.iloc[0]

            common_mix = refitted = 0.
            for band in GPA_BAND_LABELS:
                group = pooled[pooled.gpa_band == band]
                w = len(group) / len(pooled)
                common_mix += w * contrast(group)
                local_gap = contrast(local[local.scope == band])
                self.assertAlmostEqual(local_gap, self.gaps.loc[(model, band), "gap"], places=7)
                refitted += w * local_gap
            for comparison, value in [("Global", contrast(pooled)),
                                      ("Same labels, common GPA mix", common_mix),
                                      ("Refit in each GPA band", refitted)]:
                self.assertAlmostEqual(value, self.gaps.loc[(model, comparison), "gap"], places=7)
