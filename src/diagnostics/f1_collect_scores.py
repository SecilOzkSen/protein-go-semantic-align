"""Collect exact F1 cosine scores and labels from the existing evaluation path.
Run once with evaluation_split=valid, once with evaluation_split=test.
Never changes training, checkpoint, scores, or predictions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src import main as retriever_main
from src.training.trainer import OppTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = retriever_main.load_structured_cfg(a.config)
    if not cfg.eval_only or not cfg.warmstart_path or cfg.resume:
        raise ValueError("Requires eval_only=true, warmstart_path=F1 best_primary.pt, resume=null")
    if str(cfg.evaluation_split).lower() not in ("valid", "test"):
        raise ValueError("Use evaluation_split valid or test")
    out = Path(a.out);
    out.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out)
    cfg.wandb = False
    retriever_main.setup_logging(out)
    retriever_main.set_seed(cfg.seed)
    original = OppTrainer.eval_epoch

    def capture(self, loader, epoch_idx):
        original_score = self._score_full_go
        scores_all = [];
        labels_all = [];
        pids = []

        def intercepted(batch):
            scores, weights = original_score(batch)
            scores_all.append(scores.detach().float().cpu().numpy())
            labels_all.append(self._positive_mask(batch).detach().cpu().numpy().astype(np.uint8))
            pids.extend(map(str, batch["protein_ids"]))
            return scores, weights

        self._score_full_go = intercepted
        try:
            metrics = original(self, loader, epoch_idx)
        finally:
            self._score_full_go = original_score
        if not scores_all:
            raise RuntimeError("No evaluation batches captured")
        scores = np.concatenate(scores_all)
        labels = np.concatenate(labels_all)
        if scores.shape != labels.shape or scores.shape[0] != len(pids):
            raise RuntimeError("Shape or ID mismatch")
        if not np.isfinite(scores).all():
            raise RuntimeError("Non-finite scores")
        go_ids = np.asarray(self.eval_id_list, dtype=np.int64)
        np.savez_compressed(out / "scores.npz", scores=scores, labels=labels,
                            protein_ids=np.asarray(pids, dtype=str), go_ids=go_ids)
        with torch.no_grad():
            raw, weights = self._current_go_raw_bank()
            projected, _ = self._project_go_bank()
            segments = self._segment_embs_cpu.float()
            present = self._segment_present_cpu.bool()
            mean = (segments * present.unsqueeze(-1)).sum(1) / present.sum(1).clamp(min=1).unsqueeze(-1)
            np.savez_compressed(out / "go_geometry.npz",
                                go_ids=go_ids,
                                pretrained_segment_mean=mean.numpy(),
                                learned_segment_pool=raw.detach().float().cpu().numpy(),
                                learned_projected=projected.detach().float().cpu().numpy(),
                                segment_weights=weights.detach().float().cpu().numpy())
        metadata = {"split": str(cfg.evaluation_split), "checkpoint": str(cfg.warmstart_path),
                    "n_proteins": len(pids), "n_go": len(go_ids),
                    "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}}
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print("SAVED", out / "scores.npz", out / "go_geometry.npz")
        return metrics

    OppTrainer.eval_epoch = capture
    try:
        retriever_main.run_training(cfg)
    finally:
        OppTrainer.eval_epoch = original


if __name__ == "__main__":
    main()
