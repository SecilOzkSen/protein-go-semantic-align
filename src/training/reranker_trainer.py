from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


@dataclass
class RerankerTrainStats:
    loss: float
    pos_mean: float
    neg_mean: float


class RerankerTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        use_amp: bool = True,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.use_amp = bool(use_amp)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        self.global_step = 0

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    def train_step(self, batch: Dict[str, torch.Tensor]) -> RerankerTrainStats:
        batch = self._to_device(batch)
        self.model.train()

        H = batch["H"]                              # [B,T,Dh]
        valid_mask = batch["valid_mask"]            # [B,T]
        go_input_ids = batch["go_input_ids"]        # [B*K,L]
        go_attention_mask = batch["go_attention_mask"]
        labels = batch["labels"]                    # [B,K]
        cand_valid = batch["cand_valid"]            # [B,K] bool
        K = int(batch["K"].item())

        with autocast(enabled=self.use_amp):
            logits = self.model(
                H=H,
                valid_mask=valid_mask,
                go_input_ids=go_input_ids,
                go_attention_mask=go_attention_mask,
                K=K,
                return_alpha=False,
            )                                       # [B,K]

            # BCE per candidate, ignore invalid padded candidates
            loss_mat = self.bce(logits, labels)     # [B,K]
            loss_mat = loss_mat * cand_valid.to(loss_mat.dtype)
            denom = cand_valid.sum().clamp_min(1).to(loss_mat.dtype)
            loss = loss_mat.sum() / denom

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)

        # quick sanity stats
        with torch.no_grad():
            sig = torch.sigmoid(logits)
            pos = sig[(labels > 0.5) & cand_valid]
            neg = sig[(labels < 0.5) & cand_valid]
            pos_mean = float(pos.mean().item()) if pos.numel() else 0.0
            neg_mean = float(neg.mean().item()) if neg.numel() else 0.0

        self.global_step += 1
        return RerankerTrainStats(loss=float(loss.item()), pos_mean=pos_mean, neg_mean=neg_mean)

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        batch = self._to_device(batch)
        self.model.eval()

        H = batch["H"]
        valid_mask = batch["valid_mask"]
        go_input_ids = batch["go_input_ids"]
        go_attention_mask = batch["go_attention_mask"]
        labels = batch["labels"]
        cand_valid = batch["cand_valid"]
        K = int(batch["K"].item())

        logits = self.model(
            H=H,
            valid_mask=valid_mask,
            go_input_ids=go_input_ids,
            go_attention_mask=go_attention_mask,
            K=K,
            return_alpha=False,
        )  # [B,K]

        # hits@k proxy
        # For each protein: does any positive appear in top-1/top-5/top-10 after rerank?
        scores = logits.masked_fill(~cand_valid, float("-inf"))
        out: Dict[str, float] = {}

        for kk in (1, 5, 10):
            kk = min(kk, scores.size(1))
            top_idx = torch.topk(scores, k=kk, dim=1).indices  # [B,kk]
            hit = []
            for b in range(scores.size(0)):
                labs = labels[b]
                idx = top_idx[b]
                hit.append(float((labs[idx] > 0.5).any().item()))
            out[f"hits@{kk}"] = float(sum(hit) / max(1, len(hit)))

        # loss proxy
        loss_mat = self.bce(logits, labels) * cand_valid.to(labels.dtype)
        denom = cand_valid.sum().clamp_min(1).to(labels.dtype)
        out["bce"] = float((loss_mat.sum() / denom).item())

        return out