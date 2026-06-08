from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


@dataclass
class RerankerTrainStats:
    loss: float
    bce_loss: float
    dag_loss: float
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
        use_dag_loss: bool = False,
        lambda_dag: float = 0.1,
        dag_margin: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        pos_weight: float = 20.0,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.use_amp = bool(use_amp)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.use_dag_loss = bool(use_dag_loss)
        self.lambda_dag = float(lambda_dag)
        self.dag_margin = float(dag_margin)
        self.grad_clip_norm = grad_clip_norm

        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        self.pos_weight = torch.tensor(float(pos_weight), device=self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.pos_weight)

        self.global_step = 0

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    def _compute_bce_loss(
            self,
            logits: torch.Tensor,  # [B,K]
            labels: torch.Tensor,  # [B,K]
            cand_valid: torch.Tensor,  # [B,K] bool
    ) -> torch.Tensor:
        """
        BCE over valid candidate positions only.

        Important:
          - Do not use mask multiplication, because nan * 0 = nan.
          - Select valid logits explicitly.
          - Invalid candidates are excluded from the loss.
        """
        if cand_valid.dtype != torch.bool:
            valid = cand_valid != 0
        else:
            valid = cand_valid

        labels = labels.float()

        # Safety: avoid non-finite logits contaminating loss.
        # Valid non-finite logits should not normally happen, but this guard prevents
        # a whole run from becoming NaN.
        if not torch.isfinite(logits).all():
            bad_total = int((~torch.isfinite(logits)).sum().detach().cpu().item())
            bad_valid = int(((~torch.isfinite(logits)) & valid).sum().detach().cpu().item())
            logging.warning(
                "[nan-guard][bce] non-finite logits detected: total=%d valid=%d",
                bad_total,
                bad_valid,
            )

        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        )

        # Clamp for numerical safety. BCEWithLogits is stable, but this prevents
        # extreme sentinel values from union/filler candidates from causing trouble.
        logits = torch.clamp(logits, min=-30.0, max=30.0)

        # Invalid candidates should not contribute to loss.
        valid_logits = logits[valid]
        valid_labels = labels[valid]

        if valid_logits.numel() == 0:
            return logits.sum() * 0.0

        if self.pos_weight is not None:
            pos_weight = torch.as_tensor(
                float(self.pos_weight),
                device=logits.device,
                dtype=logits.dtype,
            )
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                valid_logits,
                valid_labels,
                pos_weight=pos_weight,
                reduction="mean",
            )
        else:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                valid_logits,
                valid_labels,
                reduction="mean",
            )

        if not torch.isfinite(bce):
            logging.warning("[nan-guard][bce] BCE became non-finite, returning zero loss")
            return logits.sum() * 0.0

        return bce

    def _compute_dag_loss(
            self,
            logits: torch.Tensor,  # [B,K]
            cand_valid: torch.Tensor,  # [B,K] bool
            dag_parent_mask: Optional[torch.Tensor],  # [B,K,K], child->parent mask
    ) -> torch.Tensor:
        """
        dag_parent_mask[b, i, j] = True means:
            candidate j is a parent of candidate i
            i = child, j = parent

        Penalize when child score exceeds parent score by more than margin:

            softplus(score_child - score_parent - margin)

        Only valid candidate pairs are used.
        """
        if (not self.use_dag_loss) or (dag_parent_mask is None):
            return logits.sum() * 0.0

        if cand_valid.dtype != torch.bool:
            valid = cand_valid != 0
        else:
            valid = cand_valid

        if dag_parent_mask.dtype != torch.bool:
            dag_parent_mask = dag_parent_mask != 0

        dag_parent_mask = dag_parent_mask.to(device=logits.device)
        valid = valid.to(device=logits.device)

        # Safety for non-finite logits.
        if not torch.isfinite(logits).all():
            bad_total = int((~torch.isfinite(logits)).sum().detach().cpu().item())
            bad_valid = int(((~torch.isfinite(logits)) & valid).sum().detach().cpu().item())
            logging.warning(
                "[nan-guard][dag] non-finite logits detected: total=%d valid=%d",
                bad_total,
                bad_valid,
            )

        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        )
        logits = torch.clamp(logits, min=-30.0, max=30.0)

        # valid pair mask:
        # child candidate must be valid and parent candidate must be valid.
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)  # [B,K,K]

        # candidate j is parent of candidate i
        edge_mask = dag_parent_mask & pair_valid

        if not edge_mask.any():
            return logits.sum() * 0.0

        # child_scores[b, i, j] = score of child i
        child_scores = logits.unsqueeze(2).expand_as(edge_mask)

        # parent_scores[b, i, j] = score of parent j
        parent_scores = logits.unsqueeze(1).expand_as(edge_mask)

        diffs = child_scores[edge_mask] - parent_scores[edge_mask] - float(self.dag_margin)

        # Extra safety in case extreme diffs appear.
        diffs = torch.nan_to_num(
            diffs,
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        )
        diffs = torch.clamp(diffs, min=-30.0, max=30.0)

        dag_loss = torch.nn.functional.softplus(diffs).mean()

        if not torch.isfinite(dag_loss):
            logging.warning("[nan-guard][dag] DAG loss became non-finite, returning zero loss")
            return logits.sum() * 0.0

        return dag_loss

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
        retriever_score = batch.get("retriever_score", None)
        rank_feature = batch.get("rank_feature", None)

        dag_parent_mask = batch.get("dag_parent_mask", None)  # [B,K,K], optional

        self.opt.zero_grad(set_to_none=True)

        with autocast(enabled=self.use_amp):
            logits = self.model(
                H=H,
                valid_mask=valid_mask,
                go_input_ids=go_input_ids,
                go_attention_mask=go_attention_mask,
                K=K,
                retriever_score=retriever_score,
                rank_feature=rank_feature,
                return_alpha=False,
            )                                       # [B,K]

            bce_loss = self._compute_bce_loss(
                logits=logits,
                labels=labels,
                cand_valid=cand_valid,
            )

            dag_loss = self._compute_dag_loss(
                logits=logits,
                cand_valid=cand_valid,
                dag_parent_mask=dag_parent_mask,
            )

            loss = bce_loss + (self.lambda_dag * dag_loss if self.use_dag_loss else 0.0)

        self.scaler.scale(loss).backward()

        if self.grad_clip_norm is not None:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.scaler.step(self.opt)
        self.scaler.update()

        with torch.no_grad():
            sig = torch.sigmoid(logits)
            pos = sig[(labels > 0.5) & cand_valid]
            neg = sig[(labels < 0.5) & cand_valid]
            pos_mean = float(pos.mean().item()) if pos.numel() else 0.0
            neg_mean = float(neg.mean().item()) if neg.numel() else 0.0

        self.global_step += 1
        return RerankerTrainStats(
            loss=float(loss.item()),
            bce_loss=float(bce_loss.item()),
            dag_loss=float(dag_loss.item()) if torch.is_tensor(dag_loss) else float(dag_loss),
            pos_mean=pos_mean,
            neg_mean=neg_mean,
        )

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
        retriever_score = batch.get("retriever_score", None)
        rank_feature = batch.get("rank_feature", None)

        dag_parent_mask = batch.get("dag_parent_mask", None)

        logits = self.model(
            H=H,
            valid_mask=valid_mask,
            go_input_ids=go_input_ids,
            go_attention_mask=go_attention_mask,
            K=K,
            retriever_score=retriever_score,
            rank_feature=rank_feature,
            return_alpha=False,
        )  # [B,K]

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

        bce_loss = self._compute_bce_loss(
            logits=logits,
            labels=labels,
            cand_valid=cand_valid,
        )
        dag_loss = self._compute_dag_loss(
            logits=logits,
            cand_valid=cand_valid,
            dag_parent_mask=dag_parent_mask,
        )

        total_loss = bce_loss + (self.lambda_dag * dag_loss if self.use_dag_loss else 0.0)

        out["bce"] = float(bce_loss.item())
        out["dag_loss"] = float(dag_loss.item()) if torch.is_tensor(dag_loss) else float(dag_loss)
        out["loss"] = float(total_loss.item())

        return out