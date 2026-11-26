"""NumPy-only retrieval fixtures; no checkpoint, PyTorch or GPU is exercised.

The isolated numeric helpers are loaded from evaluate_test.py without importing
its model runtime. Keep this test beside that file, or in the repository tests
directory. These fixtures validate ranks and report formatting, not inference.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import numpy as np


HERE = Path(__file__).resolve()
CANDIDATES = [HERE.with_name("evaluate_test.py"),
              HERE.parents[1] / "src" / "mzsgo_fa_benchmark" / "evaluate_test.py",
              HERE.parents[1] / "mzsgo_fa_benchmark" / "evaluate_test.py"]
SOURCE = next((path for path in CANDIDATES if path.is_file()), None)
if SOURCE is None:
    raise FileNotFoundError("Place this test beside evaluate_test.py or in the repository tests directory")
HELPERS = {"require", "general_retrieval_metrics", "general_retrieval_latex_row"}
NODES = [node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body
         if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
if {node.name for node in NODES} != HELPERS:
    raise RuntimeError("The general retrieval helpers are missing from evaluate_test.py")
NAMESPACE = {"np": np}
exec(compile(ast.Module(body=NODES, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
metrics = NAMESPACE["general_retrieval_metrics"]
latex_row = NAMESPACE["general_retrieval_latex_row"]


class GeneralRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.scores = np.asarray([[.9, .8, .7, .6, .5], [.1, .4, .5, .8, .9]], dtype=np.float32)
        self.labels = np.asarray([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=np.uint8)
        self.seen = np.asarray([True, True, False, False, False])

    def test_full_universe_ranks_and_target_masks_match_hand_calculation(self):
        result = metrics(self.scores, self.labels, self.seen, ks=(1, 2, 3, 50))
        expected = {"overall": (2 / 3, .75, .25, .5, .75),
                    "seen_targets": (.625, .625, .5, .5, .5),
                    "unseen_targets": (5 / 12, 5 / 12, 0., .5, 1.)}
        for group, values in expected.items():
            actual = [result[group][key] for key in ("mAP", "MRR", "recall@1", "recall@2", "recall@3")]
            np.testing.assert_allclose(actual, values, rtol=0, atol=1e-14)
            self.assertEqual(result[group]["recall@50"], 1.)
            self.assertEqual(result[group]["n_candidates"], 5)
        self.assertEqual(result["overall"]["positive_rank_median"], 2.5)

    def test_empty_target_group_is_null_and_group_average_excludes_empty_proteins(self):
        empty = metrics(self.scores, self.labels, np.ones(5, dtype=bool))["unseen_targets"]
        self.assertEqual(empty["n_proteins"], 0)
        self.assertEqual(empty["n_positive_pairs"], 0)
        self.assertIsNone(empty["mAP"])
        self.assertIsNone(empty["recall@50"])
        json.dumps(empty, allow_nan=False)
        labels = self.labels.copy()
        labels[1, 1] = 0
        seen = metrics(self.scores, labels, self.seen)["seen_targets"]
        self.assertEqual(seen["n_proteins"], 1)
        self.assertEqual(seen["mAP"], 1.)

    def test_ties_keep_saved_go_order_without_collapsing_unseen_ranking(self):
        result = metrics(np.zeros((1, 4), dtype=np.float32), np.array([[0, 0, 1, 0]]),
                         np.array([True, True, False, True]), ks=(1, 2, 3))
        self.assertEqual(result["unseen_targets"]["mAP"], 1 / 3)
        self.assertEqual(result["unseen_targets"]["recall@2"], 0.)
        self.assertEqual(result["unseen_targets"]["recall@3"], 1.)

    def test_protein_averaged_recall_is_distinct_from_macro_term_recall(self):
        scores = np.asarray([[3., 2., 1.], [3., 2., 1.]], dtype=np.float32)
        labels = np.asarray([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        overall = metrics(scores, labels, np.ones(3, dtype=bool), ks=(1,))["overall"]
        self.assertEqual(overall["recall@1"], .75)
        self.assertEqual(overall["macro_term_recall@1"], .5)

    def test_paper_row_uses_measured_counts_and_required_metric_order(self):
        result = metrics(self.scores, self.labels, self.seen)
        row = latex_row("CC", result).splitlines()[-1]
        self.assertEqual(row, "CC & 2 & 5 & 0.6667 & 1.0000 & 1.0000 & 1.0000 & 1.0000 " + chr(92) * 2)

    def test_invalid_cutoffs_fail(self):
        for values in ((0,), (1, 1), (True,)):
            with self.subTest(ks=values), self.assertRaises(ValueError):
                metrics(self.scores, self.labels, self.seen, ks=values)


if __name__ == "__main__":
    unittest.main()
