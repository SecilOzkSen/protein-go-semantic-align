import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "script" / "mzsgo" / "evaluate_mzsgo_temporal.py"
SPEC = importlib.util.spec_from_file_location("mzsgo_eval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TemporalMetricTests(unittest.TestCase):
    def test_perfect_temporal_scores(self):
        truth = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
        scores = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        metrics = MODULE.mzsgo_protein_metrics(truth, scores)
        self.assertEqual(metrics["fmax"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_retrieval_metrics_distinguish_coverage_and_recall(self):
        ranked = np.asarray([[1, 9], [8, 2]], dtype=np.int64)
        truths = [{1, 2}, {2}]
        metrics = MODULE.retrieval_metrics(ranked, truths, [1, 2], [1, 2])
        self.assertEqual(metrics["candidate_coverage@1"], 0.5)
        self.assertEqual(metrics["unseen_recall@1"], 0.25)
        self.assertEqual(metrics["candidate_coverage@2"], 1.0)
        self.assertEqual(metrics["unseen_recall@2"], 0.75)


if __name__ == "__main__":
    unittest.main()
