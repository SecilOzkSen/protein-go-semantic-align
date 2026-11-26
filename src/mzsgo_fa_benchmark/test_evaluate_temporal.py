"""Temporal cohort/scope fixtures; no real dataset, model run or GPU required."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.mzsgo_fa_benchmark import evaluate_temporal as temporal
from src.mzsgo_fa_benchmark.evaluate_test import sha256
from src.mzsgo_fa_benchmark.evaluate_mzsgo_generic import mzsgo_metrics
from test_report_groups import REFERENCE_EVALUATE_ANNOTATIONS


def original_retrieval_reference(s, y, cols, ks):
    """Verbatim algorithm from supplied mzsgo_temporal_v2_original(4).py."""
    n, g = s.shape
    order = np.argsort(-s, axis=1, kind='stable')
    inv = np.empty_like(order)
    inv[np.arange(n)[:, None], order] = np.arange(g)[None, :]
    rank_list = []
    aps = []
    mrr = []
    recall = {k: [] for k in ks}
    term_hits = {k: np.zeros(y.shape[1], dtype=np.int64) for k in ks}
    term_count = y.sum(0)
    for i in range(n):
        positives = np.flatnonzero(y[i])
        ranks = np.sort(inv[i, cols[positives]] + 1)
        if not len(ranks): raise RuntimeError('Temporal test protein has no positives')
        rank_list.extend(ranks.tolist())
        mrr.append(1 / float(ranks[0]))
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
        for k in ks:
            hits = inv[i, cols[positives]] < min(k, g)
            recall[k].append(float(hits.mean()))
            term_hits[k][positives] += hits.astype(np.int64)
    result = {'temporal_positive_mAP': float(np.mean(aps)),
              'temporal_positive_MRR': float(np.mean(mrr)),
              'temporal_positive_rank_median': float(np.median(rank_list)),
              'temporal_positive_rank_p90': float(np.quantile(rank_list, .9))}
    present = term_count > 0
    for k in ks:
        result[f'temporal_recall@{k}'] = float(np.mean(recall[k]))
        result[f'macro_temporal_term_recall@{k}'] = float(np.mean(term_hits[k][present] / term_count[present]))
    return result


class TemporalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        proc = self.data / "processed" / "mf"
        proc.mkdir(parents=True)
        self.annotation_path = proc / "pid_to_go_temporal_mf.json"
        self.labels = {"temporal-b": [5], "temporal-a": [3]}
        self.annotation_path.write_text(json.dumps(self.labels))
        self.bank_path = self.root / "base.pt"
        self.bank = {"format": "mzsgo_frozen_segment_parent_v1", "go_ids": [6, 2, 5, 1, 4, 3],
                     "segment_names": ["name", "definition", "is_a"],
                     "segment_embs": torch.zeros(6, 3, 4),
                     "segment_present": torch.ones(6, 3, dtype=torch.bool),
                     "parent_embs": torch.zeros(6, 2, 4),
                     "parent_present": torch.zeros(6, 2, dtype=torch.bool),
                     "metadata": {"canonical_jsonl_sha256": "canonical-fixture",
                                  "candidate_ids_sha256": "candidate-fixture",
                                  "temporal_ids_sha256": "temporal-fixture"}}
        torch.save(self.bank, self.bank_path)
        self.resources = {"candidate_go_ids": [6, 2, 5, 1, 4, 3], "temporal_go_ids": [6, 3, 5],
                          "active_go_ids": [1, 2, 4], "train_ids": ["train"], "val_ids": ["validation"],
                          "temporal_test_ids": ["temporal-a", "temporal-b", "test-without-temporal-label"],
                          "train_annotations": {"train": [1, 2]}, "base_bank": self.bank,
                          "base_bank_path": self.bank_path,
                          "input_sha256": {"canonical_text": "canonical-fixture", "candidate_ids": "candidate-fixture",
                                           "temporal_ids": "temporal-fixture", "frozen_bank": sha256(self.bank_path)},
                          "audit": {"fixture": True}}
        self.checkpoint = self.root / "best_primary.pt"
        self.checkpoint.write_bytes(b"synthetic fixed checkpoint fixture marker")
        self.bundle = {"identity": {"branch": "MF"},
                       "cfg": {"monitor_metric": "protein_wise_map"},
                       "meta": {"epoch": 25, "selection": {"best_primary_epoch": 25, "best_primary": .71},
                                "validation_metrics": {"protein_wise_map": .71}},
                       "checkpoint_sha256": sha256(self.checkpoint), "source_audit": {"fixture": True}}
        self.original_repo = self.root / "MZSGO"
        (self.original_repo / "utils").mkdir(parents=True)
        (self.original_repo / "utils" / "util.py").write_text("import numpy as np\n" + REFERENCE_EVALUATE_ANNOTATIONS + "\n")
        self.out = self.root / "temporal_output"
        self.probabilities = np.asarray([[.98, .91, .80, .90, .20, .60],
                                         [.99, .92, .30, .95, .70, .60]], dtype=np.float32)
        self.raw = (self.probabilities * np.float32(2) - np.float32(1)).astype(np.float32)

    def cohort(self):
        return temporal.prepare_temporal_cohort(self.bundle, self.resources, self.annotation_path)

    def test_all_candidates_and_zero_positive_temporal_columns_are_retained(self):
        pids, gids, targets, y, cols, audit = self.cohort()
        self.assertEqual(pids, ["temporal-a", "temporal-b"])
        self.assertEqual(gids, [1, 2, 3, 4, 5, 6])
        self.assertEqual(targets, [3, 5, 6])
        np.testing.assert_array_equal(y, [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(cols, [2, 4, 5])
        self.assertEqual(audit["n_recorded_ids_without_temporal_annotation"], 1)
        self.assertEqual(audit["n_temporal_go_with_test_positives"], 2)
        self.assertEqual(self.resources["evaluation_annotations"], self.labels)

    def test_general_or_mixed_ground_truth_is_rejected(self):
        self.annotation_path.write_text(json.dumps({"temporal-a": [1, 3]}))
        with self.assertRaisesRegex(ValueError, "non-temporal GO"):
            self.cohort()

    def test_unrecorded_temporal_protein_is_rejected(self):
        self.annotation_path.write_text(json.dumps({"unrecorded": [3]}))
        with self.assertRaisesRegex(ValueError, "outside the checkpoint-audited"):
            self.cohort()

    def test_training_protein_overlap_is_rejected(self):
        self.annotation_path.write_text(json.dumps({"train": [3]}))
        self.resources["temporal_test_ids"].append("train")
        with self.assertRaisesRegex(ValueError, "overlap training or validation"):
            self.cohort()

    def test_reserved_term_in_active_loss_candidates_is_rejected(self):
        self.resources["active_go_ids"].append(3)
        with self.assertRaisesRegex(ValueError, "excluded from the recorded training loss"):
            self.cohort()

    def test_reserved_term_in_training_positives_is_rejected(self):
        self.resources["train_annotations"]["train"].append(3)
        with self.assertRaisesRegex(ValueError, "was a training positive"):
            self.cohort()

    def test_changed_bank_temporal_identity_is_rejected(self):
        self.bank["metadata"]["temporal_ids_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "bank metadata mismatch"):
            self.cohort()

    def test_expanded_general_bank_is_rejected(self):
        self.bank["go_ids"].append(7)
        with self.assertRaisesRegex(ValueError, "original, unexpanded"):
            self.cohort()

    def test_streamed_retrieval_matches_original_with_ties_and_large_k(self):
        scores = np.asarray([[.8, .8, .8, .3, .2, .7], [.9, .1, .4, .4, .4, .5]], dtype=np.float32)
        labels = np.asarray([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        cols = np.asarray([2, 4, 5], dtype=np.int64)
        ks = [1, 3, 5, 100]
        actual = temporal.retrieval_metrics(scores, labels, cols, ks)
        expected = original_retrieval_reference(scores, labels, cols, ks)
        self.assertEqual(actual, expected)

    def _check_cli_artifacts_for_branch(self, branch):
        # Keep a zero-support reserved term for every branch, not just MF.
        proc = self.data / "processed" / branch.lower()
        proc.mkdir(parents=True, exist_ok=True)
        (proc / f"pid_to_go_temporal_{branch.lower()}.json").write_text(json.dumps(self.labels))
        self.bundle["identity"]["branch"] = branch
        out = self.root / f"temporal_output_{branch.lower()}"

        def fixture_scores(bundle, resources, pids, bank, gids, device, progress):
            self.assertIs(bundle, self.bundle)
            self.assertIs(resources, self.resources)
            self.assertIs(bank, self.bank)
            self.assertEqual(gids, [1, 2, 3, 4, 5, 6])
            self.assertEqual(pids, ["temporal-a", "temporal-b"])
            self.assertEqual(resources["evaluation_annotations"], self.labels)
            self.assertEqual(device, "cpu")
            return self.raw.copy()
        args = ["--checkpoint", str(self.checkpoint), "--branch", branch, "--root", str(self.data),
                "--repo-root", str(self.root), "--mzsgo-repo", str(self.original_repo), "--device", "cpu",
                "--out-dir", str(out), "--ks", "1", "3", "6", "50", "100", "200", "500", "1000"]
        with redirect_stdout(io.StringIO()), patch.object(temporal, "inspect_checkpoint", return_value=self.bundle), \
             patch.object(temporal, "prepare_base_resources", return_value=self.resources), \
             patch.object(temporal, "score_test", side_effect=fixture_scores):
            temporal.main(args)
        result = json.loads((out / "metrics.json").read_text())
        self.assertEqual(result["mode"], "temporal")
        self.assertEqual(result["branch"], branch)
        self.assertEqual((result["n_candidates"], result["n_temporal_go"]), (6, 3))
        self.assertEqual(result["temporal_cohort_audit"]["n_temporal_go_with_test_positives"], 2)
        self.assertFalse(result["true_path_applied"])
        self.assertFalse(result["full_train_refit"])
        self.assertNotIn("H", result)
        self.assertEqual(result["full_universe_retrieval"]["temporal_recall@1"], 0.)
        self.assertEqual(result["full_universe_retrieval"]["temporal_recall@6"], 1.)
        with np.load(out / "evaluation_data.npz", allow_pickle=False) as artifact:
            np.testing.assert_array_equal(artifact["labels_temporal"], [[1, 0, 0], [0, 1, 0]])
            np.testing.assert_array_equal(artifact["temporal_columns"], [2, 4, 5])
            self.assertEqual(len(artifact["full_go_ids"]), 6)
            labels = artifact["labels_temporal"].copy()
        mapped = np.load(out / "scores_01.npy", allow_pickle=False)
        expected = mzsgo_metrics(labels, mapped[:, [2, 4, 5]])
        for key in ("Fmax", "AUPR", "precision", "recall", "threshold"):
            self.assertAlmostEqual(result["unseen_original_mzsgo"][key], expected[key], places=12)
        rows = (out / "temporal_rows.tex").read_text()
        self.assertIn(branch + r" & 2 & 1.0000 & 1.0000 & 1.0000 & 1.0000 & 1.0000 & 0.25000 & 0.25000 \\", rows)
        expected_benchmark = (r" & FA (ours)$^{\dagger}$ & " + " & ".join(
            f"{expected[key]:.4f}" for key in ("Fmax", "AUPR", "precision", "recall")) + r" \\")
        self.assertIn(expected_benchmark, rows)
        self.assertIn("% tab:temporal_retrieval:", rows)
        self.assertIn("% tab:mzsgo_temporal_benchmark:", rows)
        manifest = json.loads((out / "evaluation_manifest.json").read_text())
        self.assertEqual(manifest["format"], "mzsgo_fa_temporal_eval_v1")
        self.assertIn("temporal_rows.tex", manifest["files"])
        for name, digest in manifest["files"].items():
            self.assertEqual(sha256(out / name), digest)
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        with self.assertRaisesRegex(ValueError, "new or empty"):
            temporal.main(args)
        self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_cli_artifacts_use_temporal_metrics_and_full_universe_retrieval(self):
        self._check_cli_artifacts_for_branch("MF")

    def test_cc_cli_retains_reserved_columns_and_exports_paper_rows(self):
        self._check_cli_artifacts_for_branch("CC")

    def test_bp_cli_retains_reserved_columns_and_exports_paper_rows(self):
        self._check_cli_artifacts_for_branch("BP")

    def test_paper_rows_preserve_table_column_order_and_precision(self):
        result = {"branch": "CC", "n_proteins": 123,
                  "full_universe_retrieval": {
                      "temporal_recall@50": .125, "temporal_recall@100": .25,
                      "temporal_recall@200": .5, "temporal_recall@500": .75,
                      "temporal_recall@1000": .875, "temporal_positive_MRR": .01234,
                      "temporal_positive_mAP": .02345},
                  "unseen_original_mzsgo": {"Fmax": .1234, "AUPR": .2345,
                                             "precision": .3456, "recall": .4567}}
        rows = temporal.paper_rows(result)
        self.assertIn(r"CC & 123 & 0.1250 & 0.2500 & 0.5000 & 0.7500 & 0.8750 & 0.01234 & 0.02345 \\", rows)
        self.assertIn(r" & FA (ours)$^{\dagger}$ & 0.1234 & 0.2345 & 0.3456 & 0.4567 \\", rows)
        del result["full_universe_retrieval"]["temporal_recall@200"]
        rows = temporal.paper_rows(result)
        self.assertIn("% CC retrieval row unavailable: --ks omitted R@200.", rows)
        self.assertNotIn("CC & 123", rows)
        self.assertIn(r" & FA (ours)$^{\dagger}$ & 0.1234 & 0.2345 & 0.3456 & 0.4567 \\", rows)

    def test_non_primary_checkpoint_is_rejected_before_scoring(self):
        self.bundle["meta"]["selection"]["best_primary_epoch"] = 24
        args = ["--checkpoint", str(self.checkpoint), "--branch", "MF", "--root", str(self.data),
                "--repo-root", str(self.root), "--mzsgo-repo", str(self.original_repo), "--device", "cpu",
                "--out-dir", str(self.out)]
        with patch.object(temporal, "inspect_checkpoint", return_value=self.bundle), \
             patch.object(temporal, "prepare_base_resources", return_value=self.resources), \
             patch.object(temporal, "score_test") as scorer:
            with self.assertRaisesRegex(ValueError, "validation-selected"):
                temporal.main(args)
            scorer.assert_not_called()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
