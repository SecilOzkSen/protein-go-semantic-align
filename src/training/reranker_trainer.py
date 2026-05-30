from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

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
        logits: torch.Tensor,      # [B,K]
        labels: torch.Tensor,      # [B,K]
        cand_valid: torch.Tensor,  # [B,K] bool
    ) -> torch.Tensor:
        loss_mat = self.bce(logits, labels)                 # [B,K]
        loss_mat = loss_mat * cand_valid.to(loss_mat.dtype)
        denom = cand_valid.sum().clamp_min(1).to(loss_mat.dtype)
        return loss_mat.sum() / denom

    def _compute_dag_loss(
        self,
        logits: torch.Tensor,                     # [B,K]
        cand_valid: torch.Tensor,                 # [B,K] bool
        dag_parent_mask: Optional[torch.Tensor],  # [B,K,K], child->parent mask
    ) -> torch.Tensor:
        """
        dag_parent_mask[b, i, j] = 1 means:
            candidate j is a parent of candidate i
            i = child, j = parent

        Penalize when child score exceeds parent score by more than margin:
            relu(score_child - score_parent - margin)
        """
        if (not self.use_dag_loss) or (dag_parent_mask is None):
            return logits.new_zeros(())

        if dag_parent_mask.dtype != torch.bool:
            dag_parent_mask = dag_parent_mask != 0

        # valid pair mask: both child and parent candidates must be valid
        pair_valid = cand_valid.unsqueeze(2) & cand_valid.unsqueeze(1)   # [B,K,K]

        # keep only declared DAG edges among valid candidates
        edge_mask = dag_parent_mask & pair_valid                         # [B,K,K]

        if not edge_mask.any():
            return logits.new_zeros(())

        score_child = logits.unsqueeze(2)   # [B,K,1] -> child index on dim=1, broadcast over parent dim
        score_parent = logits.unsqueeze(1)  # [B,1,K] -> parent index on dim=2, broadcast over child dim

        viol = torch.relu(score_child - score_parent - self.dag_margin)  # [B,K,K]
        viol = viol * edge_mask.to(viol.dtype)

        denom = edge_mask.sum().clamp_min(1).to(viol.dtype)
        return viol.sum() / denom

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