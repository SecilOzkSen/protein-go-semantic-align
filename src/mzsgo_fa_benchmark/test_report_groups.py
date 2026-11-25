"""Synthetic NumPy tests for RAW/TPR group reporting and provenance checks.

The utility fixture embeds the original evaluate_annotations function from the
audited MZSGO reference. It needs only NumPy when called with NumPy arrays.
No dataset, checkpoint, encoder, network, training or GPU is used.
"""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mzsgo_fa_benchmark import report_groups as report
from mzsgo_fa_benchmark.evaluate_mzsgo_generic import mzsgo_metrics


REFERENCE_EVALUATE_ANNOTATIONS = 'def evaluate_annotations(gold, hypo, ontology_name=\'test\'):\n    temp_gold = []\n    temp_hypo = []\n    \n    for i in range(len(gold)):\n        g = gold[i].cpu().numpy() if hasattr(gold[i], \'cpu\') else gold[i]\n        h = hypo[i].cpu().numpy() if hasattr(hypo[i], \'cpu\') else hypo[i]\n        \n        if np.sum(g) > 0:\n            temp_gold.append(g)\n            temp_hypo.append(h)\n            \n    print(f"{ontology_name}: {len(temp_gold)} proteins with annotations")\n    \n    if len(temp_gold) == 0:\n        return 0.0, 0.0, 0.0, 0.0, 0.0\n\n    y_true = np.array(temp_gold)\n    y_pred = np.array(temp_hypo)\n    \n    thresholds = np.linspace(0, 1, 101)\n    avg_prec_list = []\n    avg_rec_list = []\n    f_list = []\n    \n    N = y_true.shape[0]\n    \n    for t in thresholds:\n        y_pred_binary = (y_pred >= t).astype(int)\n        \n        tp = np.sum((y_true == 1) & (y_pred_binary == 1), axis=1)\n        fp = np.sum((y_true == 0) & (y_pred_binary == 1), axis=1)\n        fn = np.sum((y_true == 1) & (y_pred_binary == 0), axis=1)\n        \n        pred_count = tp + fp\n        \n        has_pred_mask = pred_count > 0\n        n_with_pred = np.sum(has_pred_mask)\n        \n        if n_with_pred > 0:\n            p_vals = tp[has_pred_mask] / pred_count[has_pred_mask]\n            avg_prec = np.sum(p_vals) / n_with_pred\n        else:\n            avg_prec = 0.0\n            \n        real_count = tp + fn\n        r_vals = tp / real_count\n        avg_rec = np.sum(r_vals) / N \n        \n        avg_prec_list.append(avg_prec)\n        avg_rec_list.append(avg_rec)\n        \n        if (avg_prec + avg_rec) > 0:\n            f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)\n        else:\n            f1 = 0.0\n        f_list.append(f1)\n        \n    fmax = max(f_list)\n    max_idx = np.argmax(f_list)\n    best_p = avg_prec_list[max_idx]\n    best_r = avg_rec_list[max_idx]\n    best_t = thresholds[max_idx]\n    \n    prec_array = np.array(avg_prec_list)\n    rec_array = np.array(avg_rec_list)\n    \n    sorted_indices = np.argsort(rec_array)\n    rec_sorted = rec_array[sorted_indices]\n    prec_sorted = prec_array[sorted_indices]\n    \n    aupr = np.trapz(prec_sorted, rec_sorted)\n    \n    return fmax, best_p, best_r, aupr, best_t'


class GroupReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.raw = self.base / "raw"
        self.tpr = self.base / "tpr"
        self.repo = self.base / "MZSGO"
        self.raw.mkdir()
        self.tpr.mkdir()
        (self.repo / "utils").mkdir(parents=True)
        self.util_path = self.repo / "utils" / "util.py"
        self.util_path.write_text("import numpy as np\n\n" + REFERENCE_EVALUATE_ANNOTATIONS + "\n", encoding="utf-8")
        self.labels = np.asarray([[1, 0, 1, 0, 0], [0, 1, 0, 0, 0],
                                  [0, 0, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=np.uint8)
        self.scores = np.asarray([[.8, .1, .6, .2, 0.], [.3, .7, .6, .1, .1],
                                  [.2, .4, .5, .8, .3], [.8, .6, .3, .2, .4]], dtype=np.float32)
        self.propagated = self.scores.copy()
        self.propagated[:, 0] = np.maximum(self.scores[:, 0], self.scores[:, 2])
        self.propagated[:, 1] = np.maximum(self.scores[:, 1], self.scores[:, 3])
        self.pids = ["test-a", "test-b", "test-c", "test-d"]
        self.gids = [f"GO:{i:07d}" for i in range(1, 6)]
        self.train = {"train-a": self.gids[:2], "train-b": [self.gids[0]]}
        self.pool = {**self.train, "validation-a": [self.gids[2]]}
        self.write_json(self.raw / "protein_ids.json", self.pids)
        self.write_json(self.raw / "go_ids.json", self.gids)
        self.write_json(self.raw / "checkpoint_train_labels.json", self.train)
        self.write_json(self.raw / "benchmark_train_labels.json", self.pool)
        self.write_npz()
        np.save(self.raw / "scores_01.npy", self.scores)
        np.save(self.tpr / "scores_01_tpr.npy", self.propagated)
        self.metrics = {
            "mode": "general", "branch": "MF", "score_transform": "linear",
            "n_proteins": 4, "n_candidates": 5, "checkpoint_sha256": "c" * 64,
            "full_train_refit": False,
            "general_original_mzsgo": mzsgo_metrics(self.labels, self.scores),
            "metric_source_sha256": report.sha256_file(self.util_path),
        }
        self.write_json(self.raw / "metrics.json", self.metrics)
        self.tpr_metadata = {
            "branch": "MF", "n_proteins": 4, "n_candidates": 5,
            "ground_truth_propagated": False,
            "source_scores_sha256": report.sha256_file(self.raw / "scores_01.npy"),
            "source_raw_dir": str(self.raw),
            "n_changed_scores": int(np.count_nonzero(self.propagated > self.scores + 1e-7)),
            "raw": mzsgo_metrics(self.labels, self.scores),
            "propagated": mzsgo_metrics(self.labels, self.propagated),
            "ontology_path": "fixture-go-2023.obo", "ontology_sha256": "d" * 64,
        }
        self.write_json(self.tpr / "metrics_tpr.json", self.tpr_metadata)
        self.manifest = {
            "format": report.MANIFEST_FORMAT, "branch": "MF", "n_proteins": 4,
            "n_candidates": 5, "checkpoint_sha256": "c" * 64,
        }
        self.refresh_manifest()
        self.args = SimpleNamespace(raw_dir=self.raw, tpr_dir=self.tpr, mzsgo_repo=self.repo)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def write_npz(self, *, labels=None, pids=None, gids=None):
        np.savez_compressed(self.raw / "evaluation_data.npz",
                            labels=self.labels if labels is None else labels,
                            protein_ids=np.asarray(self.pids if pids is None else pids),
                            go_ids=np.asarray(self.gids if gids is None else gids))

    def refresh_manifest(self):
        self.manifest["files"] = {filename: report.sha256_file(self.raw / filename) for filename in report.RAW_FILES}
        self.write_json(self.raw / "evaluation_manifest.json", self.manifest)

    def execute(self):
        console = io.StringIO()
        with redirect_stdout(console):
            result = report.execute(self.args)
        return result, console.getvalue()

    def assert_no_outputs(self):
        self.assertFalse((self.raw / report.REPORT_NAME).exists())
        self.assertFalse((self.tpr / report.REPORT_NAME).exists())

    def test_primary_and_diagnostic_partitions_use_distinct_truthful_training_bases(self):
        original_npz_hash = report.sha256_file(self.raw / "evaluation_data.npz")
        result, console = self.execute()
        for variant in ("raw", "tpr"):
            value = result[variant]
            self.assertEqual(value["training_basis"], report.PRIMARY_BASIS)
            self.assertEqual((value["n_seen_test_go"], value["n_unseen_test_go"]), (2, 2))
            self.assertEqual(value["seen"]["proteins_with_positive"], 3)
            self.assertEqual(value["unseen"]["proteins_with_positive"], 2)
            self.assertEqual(value["overall"]["n_terms"], 5)
            self.assertEqual(value["overall"]["positive_pairs"], 6)
            diagnostic = value["benchmark_training_pool_partition"]
            self.assertEqual(diagnostic["training_basis"], report.DIAGNOSTIC_BASIS)
            self.assertEqual((diagnostic["n_seen_test_go"], diagnostic["n_unseen_test_go"]), (3, 1))
            self.assertEqual(diagnostic["unseen"]["proteins_with_positive"], 1)
            self.assertIs(diagnostic["diagnostic_only"], True)
            self.assertIs(diagnostic["checkpoint_refit_on_this_pool"], False)
            self.assertIs(value["checkpoint_refit_on_train_plus_validation"], False)
            self.assertIs(value["ground_truth_propagated"], False)
            s, u = value["seen"]["AUPR"], value["unseen"]["AUPR"]
            self.assertAlmostEqual(value["H"], 2 * s * u / (s + u), places=14)
            self.assertEqual(diagnostic["overall"], value["overall"])
            self.assertLessEqual(value["metric_parity"]["unseen"]["max_absolute_difference"], 1e-5)
        self.assertEqual(report.sha256_file(self.raw / "evaluation_data.npz"), original_npz_hash)
        np.testing.assert_array_equal(np.load(self.raw / "scores_01.npy"), self.scores)
        np.testing.assert_array_equal(np.load(self.tpr / "scores_01_tpr.npy"), self.propagated)
        self.assertIn("PRIMARY PARTITION", console)
        self.assertIn("DIAGNOSTIC PARTITION ONLY, NO REFIT", console)
        self.assertIn("| RAW |", console)
        self.assertIn("| TPR |", console)
        for directory in (self.raw, self.tpr):
            payload = json.loads((directory / report.REPORT_NAME).read_text())
            self.assertEqual(payload["format"], report.REPORT_FORMAT)
            self.assertNotIn("NaN", (directory / report.REPORT_NAME).read_text())

    def test_hash_mismatch_stops_before_any_report(self):
        np.save(self.raw / "scores_01.npy", self.scores + .001)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.execute()
        self.assert_no_outputs()

    def test_manifest_path_traversal_is_rejected(self):
        self.manifest["files"]["../outside.json"] = "a" * 64
        self.write_json(self.raw / "evaluation_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "basename"):
            self.execute()
        self.assert_no_outputs()

    def test_id_order_mismatch_is_not_hidden_by_equal_dimensions(self):
        self.write_npz(pids=list(reversed(self.pids)))
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "protein ID order"):
            self.execute()
        self.assert_no_outputs()

    def test_go_order_and_duplicates_are_rejected(self):
        for changed in (list(reversed(self.gids)), [self.gids[0]] * 5):
            with self.subTest(gids=changed):
                self.write_npz(gids=changed)
                self.refresh_manifest()
                with self.assertRaisesRegex(ValueError, "GO ID order|Duplicate"):
                    self.execute()
                self.assert_no_outputs()

    def test_nonbinary_gold_and_dimension_metadata_fail(self):
        labels = self.labels.copy()
        labels[0, 0] = 2
        self.write_npz(labels=labels)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "binary"):
            self.execute()
        self.write_npz()
        self.manifest["n_proteins"] = 3
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "n_proteins mismatch"):
            self.execute()
        self.assert_no_outputs()

    def test_training_test_overlap_and_mismatched_pool_annotations_fail(self):
        train = {**self.train, self.pids[0]: [self.gids[0]]}
        pool = {**self.pool, self.pids[0]: [self.gids[0]]}
        self.write_json(self.raw / "checkpoint_train_labels.json", train)
        self.write_json(self.raw / "benchmark_train_labels.json", pool)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.execute()
        self.write_json(self.raw / "checkpoint_train_labels.json", self.train)
        pool = deepcopy(self.pool)
        pool["train-a"] = [self.gids[0]]
        self.write_json(self.raw / "benchmark_train_labels.json", pool)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.execute()
        self.assert_no_outputs()

    def test_tpr_source_hash_directory_and_gold_policy_are_checked(self):
        for key, value, expected in (
            ("source_scores_sha256", "a" * 64, "score SHA256"),
            ("source_raw_dir", str(self.base / "different"), "directory mismatch"),
            ("ground_truth_propagated", True, "preserve original gold"),
        ):
            with self.subTest(key=key):
                changed = deepcopy(self.tpr_metadata)
                changed[key] = value
                self.write_json(self.tpr / "metrics_tpr.json", changed)
                with self.assertRaisesRegex(ValueError, expected):
                    self.execute()
                self.assert_no_outputs()

    def test_tpr_scores_must_not_decrease_and_counts_must_match(self):
        lowered = self.propagated.copy()
        lowered[0, 0] = self.scores[0, 0] - .1
        np.save(self.tpr / "scores_01_tpr.npy", lowered)
        with self.assertRaisesRegex(ValueError, "lowered"):
            self.execute()
        np.save(self.tpr / "scores_01_tpr.npy", self.propagated)
        self.tpr_metadata["n_changed_scores"] += 1
        self.write_json(self.tpr / "metrics_tpr.json", self.tpr_metadata)
        with self.assertRaisesRegex(ValueError, "changed-score count"):
            self.execute()
        self.assert_no_outputs()

    def test_original_utility_source_hash_is_required(self):
        self.util_path.write_text(self.util_path.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "metric source SHA256"):
            self.execute()
        self.assert_no_outputs()

    def test_refit_metadata_cannot_be_relabelled_as_non_refit(self):
        self.metrics["full_train_refit"] = True
        self.write_json(self.raw / "metrics.json", self.metrics)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "non-refit"):
            self.execute()
        self.assert_no_outputs()

    def test_generic_original_semantic_disagreement_is_not_reported_as_parity(self):
        self.util_path.write_text(self.util_path.read_text() + "\n_original = evaluate_annotations\ndef evaluate_annotations(gold, hypo, ontology_name='test'):\n    values = list(_original(gold, hypo, ontology_name))\n    values[3] += 0.02\n    return values\n")
        self.metrics["metric_source_sha256"] = report.sha256_file(self.util_path)
        self.write_json(self.raw / "metrics.json", self.metrics)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "generic/original.*parity failed"):
            self.execute()
        self.assert_no_outputs()

    def test_raw_and_tpr_recorded_overall_metrics_are_rechecked(self):
        for section in ("raw", "propagated"):
            with self.subTest(section=section):
                changed = deepcopy(self.tpr_metadata)
                changed[section]["Fmax"] += .01
                self.write_json(self.tpr / "metrics_tpr.json", changed)
                with self.assertRaisesRegex(ValueError, "parity failed"):
                    self.execute()
                self.assert_no_outputs()

    def test_empty_unseen_partition_returns_null_without_nan_or_refit(self):
        all_train = {"train-a": self.gids}
        self.write_json(self.raw / "checkpoint_train_labels.json", all_train)
        self.write_json(self.raw / "benchmark_train_labels.json", all_train)
        self.refresh_manifest()
        result, _ = self.execute()
        for variant in ("raw", "tpr"):
            self.assertIsNone(result[variant]["unseen"])
            self.assertIsNone(result[variant]["H"])
            self.assertEqual(result[variant]["n_unseen_test_go"], 0)

    def test_existing_output_is_never_overwritten(self):
        destination = self.raw / report.REPORT_NAME
        destination.write_text("existing report\n")
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(destination.read_text(), "existing report\n")
        self.assertFalse((self.tpr / report.REPORT_NAME).exists())

    def test_concurrent_second_output_preserved_and_own_first_output_rolled_back(self):
        native_link = os.link
        calls = []
        def racing_link(source, destination):
            calls.append(Path(destination))
            if len(calls) == 2:
                Path(destination).write_text("other process output\n")
                raise FileExistsError("Output appeared after preflight")
            return native_link(source, destination)
        with patch.object(report.os, "link", side_effect=racing_link):
            with self.assertRaises(FileExistsError):
                self.execute()
        self.assertFalse((self.raw / report.REPORT_NAME).exists())
        self.assertEqual((self.tpr / report.REPORT_NAME).read_text(), "other process output\n")
        self.assertEqual(list(self.raw.glob("*.tmp")), [])
        self.assertEqual(list(self.tpr.glob("*.tmp")), [])

    def test_cli_accepts_required_paths_and_fails_cleanly_on_missing_input(self):
        parsed = report.parser().parse_args(["--raw-dir", str(self.raw), "--tpr-dir", str(self.tpr)])
        self.assertEqual(parsed.mzsgo_repo, Path("/workspace/MZSGO"))
        (self.raw / "evaluation_manifest.json").unlink()
        with redirect_stderr(io.StringIO()) as error:
            code = report.main(["--raw-dir", str(self.raw), "--tpr-dir", str(self.tpr), "--mzsgo-repo", str(self.repo)])
        self.assertEqual(code, 1)
        self.assertIn("[ERROR]", error.getvalue())
        self.assert_no_outputs()


if __name__ == "__main__":
    unittest.main()
