"""
Checks for the shared preprocessing and Member 1's saved results.
Run from the repository root (after python preprocess_a3.py):

    python -m unittest discover -s tests -v

The last test re-runs the Member 1 pipeline into a temporary copy of the repo and
compares every saved number with the committed Member1/outputs/m1_results.json.
It takes about a minute.
"""
import hashlib
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a3_common import load_base, load_meta, RAW_FILE  # noqa: E402

A1_SHA256 = "ba4476704af3849e8c7213de4f274ae16216f931fc0890904fd893ac2b9b448f"


class TestSourceData(unittest.TestCase):
    def test_source_is_unchanged_a1_output(self):
        with gzip.open(RAW_FILE, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), A1_SHA256)


class TestBaseTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_base()
        cls.meta = load_meta()

    def test_shape_and_ids(self):
        self.assertEqual(len(self.df), 80_000)
        self.assertTrue(self.df["student_id"].is_unique)

    def test_split(self):
        counts = self.df["split"].value_counts()
        self.assertEqual(counts["train"], 64_000)
        self.assertEqual(counts["test"], 16_000)

    def test_redundant_columns_removed(self):
        for c in ["exam_anxiety_score", "study_hours_per_week", "leisure_screen_hours"]:
            self.assertNotIn(c, self.df.columns)

    def test_dropout_rule_holds(self):
        rule = (self.df["stress_level"] > 8) & (self.df["motivation_level"] <= 3)
        self.assertTrue((rule.astype(int) == self.df["dropout_risk"]).all())

    def test_ceiling_flags(self):
        self.assertAlmostEqual(self.df["score_at_ceiling"].mean(), 0.25765, places=4)
        self.assertAlmostEqual(self.df["gpa_at_ceiling"].mean(), 0.3996125, places=6)

    def test_one_hot_rows_sum_to_one(self):
        oh = self.meta["one_hot_columns"]
        for prefix in ["oh_gender", "oh_major", "oh_study_environment", "oh_learning_style"]:
            cols = [c for c in oh if c.startswith(prefix + "_")]
            self.assertTrue((self.df[cols].sum(axis=1) == 1).all(), prefix)


class TestMember1Results(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = json.loads((ROOT / "Member1/outputs/m1_results.json").read_text())

    def test_headline_numbers(self):
        g = {r["label"]: r for r in self.R["it2_gaps"]}
        self.assertAlmostEqual(g["Raw score (train)"]["est"], 4.2171, places=3)
        self.assertAlmostEqual(g["GPA-adjusted (test)"]["est"], -0.00654, places=3)
        self.assertLess(g["GPA-adjusted (test)"]["hi"], 0.2)
        self.assertGreater(g["GPA-adjusted (test)"]["lo"], -0.2)
        self.assertEqual(self.R["it2_k"], 2)
        self.assertGreater(self.R["it2_k_selection"][0]["stability_ari"], 0.95)

    def test_pipeline_reproduces_saved_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for item in ["a3_common.py", "preprocess_a3.py", "data", "Member1"]:
                src = ROOT / item
                (shutil.copytree if src.is_dir() else shutil.copy)(src, tmp / item)
            shutil.rmtree(tmp / "Member1" / "outputs")
            for script in ["preprocess_a3.py", "Member1/m1_clustering_workflow.py"]:
                subprocess.run([sys.executable, script], cwd=tmp, check=True,
                               capture_output=True)
            new = json.loads((tmp / "Member1/outputs/m1_results.json").read_text())
        old = self.R

        def compare(a, b, path="R"):
            if isinstance(a, dict):
                self.assertEqual(set(a), set(b), path)
                for k in a:
                    compare(a[k], b[k], f"{path}.{k}")
            elif isinstance(a, list):
                self.assertEqual(len(a), len(b), path)
                for i, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, f"{path}[{i}]")
            elif isinstance(a, float):
                self.assertTrue(np.isclose(a, b, rtol=1e-6, atol=1e-9), f"{path}: {a} vs {b}")
            else:
                self.assertEqual(a, b, path)

        compare(old, new)


if __name__ == "__main__":
    unittest.main()
