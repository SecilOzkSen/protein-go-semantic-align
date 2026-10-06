from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.configs.data_classes import TrainerConfig
from src.metrics.cafa import compute_protein_centric_fmax, compute_term_aupr
from src.models.alignment_model import ProteinGoAligner


def multi_positive_full_go_infonce(scores, pos_mask, temperature):
    """Multi-positive InfoNCE over the complete active GO universe."""
    if scores.ndim != 2 or pos_mask.shape != scores.shape:
        raise ValueError("InfoNCE expects scores and pos_mask with shape [B,G].")
    tau = max(float(temperature), 1e-8)
    pos_mask = pos_mask.bool()
    counts = pos_mask.sum(1)
    if (counts == 0).any():
        raise RuntimeError("InfoNCE received a protein with zero positives.")
    logits = scores.float() / tau
    denom = torch.logsumexp(logits, dim=1)
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    num = torch.logsumexp(pos_logits, dim=1) - counts.float().log()
    loss = -(num - denom)
    if not torch.isfinite(loss).all():
        raise RuntimeError("Non-finite full-GO InfoNCE.")
    return loss.mean()


def positive_block_rank_loss(scores, pos_mask, margin, tau):
    """
    PBR: sigmoid((s_neg - s_pos + margin) / tau).
    Aggregation: negatives SUM -> positives MEAN -> proteins MEAN.
    """
    tau = max(float(tau), 1e-8)
    pos_mask = pos_mask.bool()
    rows, sum_stats, mean_stats = [], [], []
    hard_violation_stats = []
    for b in range(scores.size(0)):
        pos = scores[b][pos_mask[b]].float()
        neg = scores[b][~pos_mask[b]].float()
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        v = torch.sigmoid((neg[None, :] - pos[:, None] + float(margin)) / tau)
        per_pos = v.sum(dim=1)
        rows.append(per_pos.mean())
        sum_stats.append(per_pos.detach().mean())
        mean_stats.append(v.detach().mean())

        # Hard diagnostic only, separate from the smooth PBR objective.
        # Violation means the requested score margin is not yet satisfied:
        #     s_pos < s_neg + margin
        hard_violation = (
                neg[None, :] - pos[:, None] + float(margin) > 0.0
        ).float()
        hard_violation_stats.append(hard_violation.detach().mean())
    if not rows:
        z = scores.sum() * 0.0
        return z, {
            "pbr_violation_sum_per_positive": 0.0,
            "pbr_pair_violation_mean": 0.0,
            "pbr_margin_violation_fraction": 0.0,
            "pbr_margin_satisfied_fraction": 1.0,
        }
    loss = torch.stack(rows).mean()
    return loss, {
        "pbr_violation_sum_per_positive": float(torch.stack(sum_stats).mean().item()),
        "pbr_pair_violation_mean": float(torch.stack(mean_stats).mean().item()),
        "pbr_margin_violation_fraction": float(
            torch.stack(hard_violation_stats).mean().item()
        ),
        "pbr_margin_satisfied_fraction": float(
            1.0 - torch.stack(hard_violation_stats).mean().item()
        ),
    }


class OppTrainer:
    """Simplified full-GO Retriever v2 trainer."""

    def __init__(self, cfg: TrainerConfig, ctx, go_encoder, wandb_run=None):
        self.cfg = cfg
        self.ctx = ctx
        self.device = torch.device(cfg.device)
        self.wandb_run = wandb_run  # compatibility only; W&B logging is external
        self._global_step = 0

        if ctx.protein_pooling_strategy != "mean_attn_gate":
            raise ValueError("Retriever v2 requires mean_attn_gate protein pooling.")
        if ctx.go_encoder_output_mode != "segment_pooled":
            raise ValueError("Retriever v2 requires segment_pooled GO mode.")
        if cfg.use_lora:
            raise ValueError("Retriever v2 requires use_lora=False.")

        self.model = ProteinGoAligner(
            d_h=cfg.d_h,
            d_g=cfg.d_g,
            d_z=cfg.d_z,
            go_encoder=go_encoder,
            normalize=True,
            protein_pool_type=ctx.protein_pooling_strategy,
            go_pool_type=ctx.go_pool_type,
            go_segment_representation_mode=ctx.go_segment_representation_mode,
        ).to(self.device)

        if self.model.go_encoder is None:
            raise RuntimeError("GO encoder is required.")
        for p in self.model.go_encoder.parameters():
            p.requires_grad_(False)
        self.model.go_encoder.eval()

        if cfg.warmstart_path:
            self._load_warmstart(cfg.warmstart_path)

        self.eval_id_list = [int(x) for x in (ctx.eval_id_list or [])]
        if not self.eval_id_list:
            raise RuntimeError("Active GO universe is empty.")
        self._go_ids_cpu = torch.tensor(self.eval_id_list, dtype=torch.long)
        self._go_id2col = {g: i for i, g in enumerate(self.eval_id_list)}

        self._build_frozen_segment_bank()

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(
            params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
        )
        self._print_trainable_summary()

    def _print_trainable_summary(self):
        bad = [n for n, p in self.model.go_encoder.named_parameters() if p.requires_grad]
        if bad:
            raise RuntimeError(f"Frozen GO encoder has trainables: {bad[:10]}")
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print("[RETRIEVER-V2] active GO terms:", len(self.eval_id_list))
        print("[RETRIEVER-V2] trainable sample:", names[:40])

    def _load_warmstart(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt
        for _ in range(10):
            if isinstance(state, dict) and sum(torch.is_tensor(v) for v in state.values()) >= 5:
                break
            for key in ("model", "model_state_dict", "state_dict", "net", "module"):
                if isinstance(state, dict) and isinstance(state.get(key), dict):
                    state = state[key]
                    break
            else:
                raise RuntimeError("Cannot locate warmstart state_dict.")
        current = self.model.state_dict()
        loadable = {}
        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            clean = k
            changed = True
            while changed:
                changed = False
                for pref in ("trainer.model.", "module.", "model."):
                    if clean.startswith(pref):
                        clean = clean[len(pref):]
                        changed = True
            if clean in current and tuple(current[clean].shape) == tuple(v.shape):
                loadable[clean] = v
        if not loadable:
            raise RuntimeError("Warmstart loaded zero compatible tensors.")
        missing, unexpected = self.model.load_state_dict(loadable, strict=False)
        print(f"[WARMSTART] loaded={len(loadable)} missing={len(missing)} unexpected={len(unexpected)}")

    @torch.no_grad()
    def _build_frozen_segment_bank(self):
        """
        Encode each active GO segment once with frozen BioMedBERT.
        Segment mixing remains trainable and is applied every step.
        """
        toks = self.ctx.go_text_store.batch(self.eval_id_list)
        seg_ids = toks["seg_input_ids"]
        seg_mask = toks["seg_attention_mask"]
        present = toks["seg_present"].bool()
        if seg_ids.ndim != 3:
            raise RuntimeError(f"Expected [G,S,L], got {tuple(seg_ids.shape)}")
        G, S, L = seg_ids.shape
        if G != len(self.eval_id_list) or (present.sum(1) == 0).any():
            raise RuntimeError("Invalid segmented GO bank.")

        D = int(self.cfg.d_g)
        bank = torch.zeros(G, S, D, dtype=torch.float32)
        flat_ids = seg_ids.reshape(G * S, L)
        flat_mask = seg_mask.reshape(G * S, L)
        flat_present = present.reshape(G * S)
        idx_all = torch.nonzero(flat_present, as_tuple=False).flatten()
        flat_bank = bank.view(G * S, D)
        bs = max(1, int(self.cfg.eval_go_bs))

        self.model.go_encoder.eval()
        for st in range(0, idx_all.numel(), bs):
            idx = idx_all[st:st + bs]
            out = self.model.go_encoder(
                input_ids=flat_ids.index_select(0, idx).to(self.device),
                attention_mask=flat_mask.index_select(0, idx).to(self.device),
                output_mode="pooled",
            )
            if isinstance(out, tuple):
                out = out[0]
            elif isinstance(out, dict):
                out = out.get("pooled", out.get("pooler_output"))
            if not torch.is_tensor(out) or out.ndim != 2 or out.size(1) != D:
                raise RuntimeError("Unexpected frozen GO encoder output.")
            flat_bank.index_copy_(0, idx.cpu(), torch.nan_to_num(out).float().cpu())

        self._segment_embs_cpu = bank.contiguous()
        self._segment_present_cpu = present.cpu().contiguous()
        self._segment_names = list(toks.get("segment_names", []))
        print("[GO-SEGMENT-BANK]", tuple(bank.shape), self._segment_names)

    def _current_go_raw_bank(self):
        if self.ctx.go_segment_representation_mode != "segments_only":
            raise RuntimeError("Retriever v2 currently requires segments_only.")
        emb = self._segment_embs_cpu.to(self.device, non_blocking=True)
        present = self._segment_present_cpu.to(self.device, non_blocking=True)
        gate = getattr(self.model, "go_segment_gate", None)
        if gate is None:
            raise RuntimeError("ProteinGoAligner.go_segment_gate is missing.")
        logits = gate(emb).squeeze(-1).float().masked_fill(~present, -1e9)
        weights = torch.softmax(logits, dim=-1).to(emb.dtype)
        pooled = torch.einsum("gs,gsd->gd", weights, emb)
        return torch.nan_to_num(pooled), weights

    def _project_go_bank(self):
        raw, weights = self._current_go_raw_bank()
        z = self.model.go_ln(raw)
        z = self.model.proj_g(z)
        z = F.normalize(z.float(), dim=-1)
        return z, weights

    def _encode_protein(self, batch):
        H = batch["prot_emb_pad"].to(self.device, non_blocking=True).float()
        mask = batch["prot_attn_mask"].to(self.device, non_blocking=True).bool()
        z = self.model.encode_protein_for_scoring(H, mask=mask)
        if isinstance(z, tuple):
            z = z[0]
        if z.ndim != 2:
            raise RuntimeError(f"Protein encoder must return [B,D], got {tuple(z.shape)}")
        return F.normalize(z.float(), dim=-1)

    def _positive_mask(self, batch):
        B = len(batch["pos_go_global"])
        G = len(self.eval_id_list)
        mask = torch.zeros(B, G, dtype=torch.bool, device=self.device)
        for b, gids in enumerate(batch["pos_go_global"]):
            cols = []
            for gid in gids.detach().cpu().tolist():
                col = self._go_id2col.get(int(gid))
                if col is not None:
                    cols.append(col)
            if not cols:
                raise RuntimeError(
                    f"Protein {batch['protein_ids'][b]} has no positive in active GO universe."
                )
            mask[b, torch.tensor(cols, device=self.device, dtype=torch.long)] = True
        return mask

    def _score_full_go(self, batch):
        zp = self._encode_protein(batch)
        zg, seg_weights = self._project_go_bank()
        scores = zp @ zg.T
        return scores, seg_weights

    @torch.no_grad()
    def _score_diagnostics(self, scores, pos_mask):
        """
        Score-space diagnostics for Retriever v2.

        These are deliberately computed on the raw cosine-similarity matrix
        before InfoNCE temperature scaling. They answer a different question
        from the losses: are positives actually separating from negatives?
        """
        if scores.ndim != 2 or pos_mask.shape != scores.shape:
            raise ValueError(
                "Score diagnostics expect scores and pos_mask with shape [B,G]."
            )

        pos_mask = pos_mask.bool()
        neg_mask = ~pos_mask

        if not pos_mask.any():
            raise RuntimeError("Score diagnostics received no positive scores.")
        if not neg_mask.any():
            raise RuntimeError("Score diagnostics received no negative scores.")

        positive_mean = scores[pos_mask].float().mean()
        negative_mean = scores[neg_mask].float().mean()
        gap = positive_mean - negative_mean

        return {
            "positive_score_mean": float(positive_mean.item()),
            "negative_score_mean": float(negative_mean.item()),
            "pos_neg_score_gap": float(gap.item()),
        }

    @torch.no_grad()
    def _rank_diagnostics(self, scores, pos_mask):
        """Hard full-universe positive-rank diagnostics for a training batch."""
        ranks_all, worst, pbc_per_protein = [], [], []

        for b in range(scores.size(0)):
            order = torch.argsort(scores[b], descending=True)
            inv = torch.empty_like(order)
            inv[order] = torch.arange(order.numel(), device=order.device)
            pos_cols = torch.nonzero(pos_mask[b], as_tuple=False).flatten()
            if pos_cols.numel() == 0:
                continue

            ranks = inv.index_select(0, pos_cols).float() + 1.0
            ranks_all.append(ranks)
            worst.append(ranks.max())
            npos = int(pos_cols.numel())
            pbc_per_protein.append((ranks <= npos).float().mean())

        if not ranks_all:
            raise RuntimeError("Rank diagnostics received no positive GO terms.")

        allr = torch.cat(ranks_all)
        pbc = torch.stack(pbc_per_protein).mean()

        out = {
            "positive_rank_mean": float(allr.mean().item()),
            "positive_rank_median": float(torch.median(allr).item()),
            "positive_rank_worst_mean": float(torch.stack(worst).mean().item()),
            "positive_block_coverage": float(pbc.item()),
            "positive_block_violation": float((1.0 - pbc).item()),
        }
        for q in self.cfg.positive_rank_quantiles:
            out[f"positive_rank_q{int(round(100 * q))}"] = float(
                torch.quantile(allr, float(q)).item()
            )
        return out

    def step_losses(self, batch, epoch_idx, debug=False):
        self.model.train()
        self.model.go_encoder.eval()

        scores, seg_weights = self._score_full_go(batch)
        pos_mask = self._positive_mask(batch)

        l_con = multi_positive_full_go_infonce(
            scores, pos_mask, temperature=self.cfg.temperature
        )
        l_pbr, pbr_stats = positive_block_rank_loss(
            scores, pos_mask,
            margin=self.cfg.pbr_margin,
            tau=self.cfg.pbr_tau,
        )
        l_total = self.cfg.lambda_con * l_con + self.cfg.pbr_lambda * l_pbr

        if not torch.isfinite(l_total):
            raise RuntimeError("Non-finite Retriever v2 total loss.")

        score_diag = self._score_diagnostics(scores.detach(), pos_mask)
        rank_diag = self._rank_diagnostics(scores.detach(), pos_mask)
        self._global_step += 1

        out = {
            "total": l_total,
            "contrastive": l_con,
            "pbr": l_pbr,
            "pbr_weighted": float(self.cfg.pbr_lambda) * l_pbr.detach(),
            **pbr_stats,
            **score_diag,
            **rank_diag,
        }

        if self.cfg.log_go_segment_weights:
            means = seg_weights.detach().float().mean(0).cpu()
            names = self._segment_names or [f"segment_{i}" for i in range(means.numel())]
            for i, name in enumerate(names):
                out[f"go_segment_weight/{name}"] = float(means[i].item())

        return out

    @torch.no_grad()
    def _cardinality_bin(self, n):
        for lo, hi, label in (
                (1, 5, "1_5"), (6, 10, "6_10"), (11, 20, "11_20"),
                (21, 40, "21_40"), (41, 80, "41_80"), (81, 160, "81_160"),
                (161, 10 ** 9, "161plus"),
        ):
            if lo <= n <= hi:
                return label
        return "unknown"

    @torch.no_grad()
    def eval_epoch(self, loader, epoch_idx):
        """
        Exhaustive evaluation over the active GO universe.

        Returns scalar metrics. Raw diagnostics for the future
        RetrieverWandbLogger are stored in self.last_eval_diagnostics.
        """
        self.model.eval()
        self.model.go_encoder.eval()

        all_scores, all_true = [], []
        ks = tuple(sorted(set(int(k) for k in self.cfg.retrieval_eval_ks)))
        if not ks:
            raise RuntimeError("retrieval_eval_ks is empty.")

        recall_sum = {k: 0.0 for k in ks}
        cand = {
            k: {
                "coverage_sum": 0.0, "n": 0, "tp": 0.0, "fn": 0.0,
                "protein_f_sum": 0.0,
            }
            for k in ks
        }
        card_recall = {
            k: defaultdict(lambda: {"sum": 0.0, "n": 0})
            for k in ks
        }
        card_rank = defaultdict(
            lambda: {"protein_n": 0, "positive_ranks": [], "pbc_sum": 0.0}
        )

        rank_values, worst_rank_values, pbc_values = [], [], []
        reciprocal_rank_values, ndcg10_values = [], []

        G = len(self.eval_id_list)
        term_true = torch.zeros(G, dtype=torch.float64)
        term_hit = {k: torch.zeros(G, dtype=torch.float64) for k in ks}
        final_segment_weights = None

        for batch in loader:
            scores, segment_weights = self._score_full_go(batch)
            y = self._positive_mask(batch)
            final_segment_weights = segment_weights.detach().float().cpu()

            all_scores.append(scores.detach().cpu())
            all_true.append(y.float().cpu())

            B, Gcur = scores.shape
            maxk = min(max(ks), Gcur)
            top = torch.topk(scores, k=maxk, dim=1).indices
            true_counts = y.sum(dim=1)
            valid = true_counts > 0
            term_true += y.cpu().double().sum(dim=0)

            for b in range(B):
                if not valid[b]:
                    continue

                order = torch.argsort(scores[b], descending=True)
                inv = torch.empty_like(order)
                inv[order] = torch.arange(Gcur, device=self.device)

                pos_cols = torch.nonzero(y[b], as_tuple=False).flatten()
                ranks = inv.index_select(0, pos_cols).float() + 1.0
                npos = int(pos_cols.numel())

                rank_values.append(ranks.cpu())
                worst_rank_values.append(float(ranks.max().item()))

                pbc_i = float((ranks <= npos).float().mean().item())
                pbc_values.append(pbc_i)

                # Multi-positive MRR: reciprocal rank of the first relevant GO.
                reciprocal_rank_values.append(1.0 / float(ranks.min().item()))

                # Binary-relevance nDCG@10.
                k10 = min(10, Gcur)
                rel10 = y[b].index_select(0, order[:k10]).float()
                discounts = 1.0 / torch.log2(
                    torch.arange(2, k10 + 2, device=self.device, dtype=torch.float32)
                )
                dcg = float((rel10 * discounts).sum().item())
                ideal_hits = min(npos, k10)
                idcg = float(discounts[:ideal_hits].sum().item())
                ndcg10_values.append(dcg / idcg if idcg > 0.0 else 0.0)

                label = self._cardinality_bin(npos)
                card_rank[label]["protein_n"] += 1
                card_rank[label]["positive_ranks"].append(ranks.cpu())
                card_rank[label]["pbc_sum"] += pbc_i

            for k in ks:
                kk = min(k, Gcur)
                idx = top[:, :kk]
                hits = torch.gather(y.float(), 1, idx).sum(dim=1)
                cov = hits / true_counts.float().clamp_min(1)

                n_valid = int(valid.sum().item())
                if n_valid:
                    recall_sum[k] += float(cov[valid].sum().item())
                    cand[k]["coverage_sum"] += float(cov[valid].sum().item())
                    cand[k]["n"] += n_valid
                    cand[k]["tp"] += float(hits[valid].sum().item())
                    cand[k]["fn"] += float(
                        (true_counts[valid].float() - hits[valid]).sum().item()
                    )
                    protein_oracle_f = (
                            2.0 * hits[valid]
                            / (true_counts[valid].float() + hits[valid]).clamp_min(1e-12)
                    )
                    cand[k]["protein_f_sum"] += float(protein_oracle_f.sum().item())

                retrieved = torch.zeros_like(y)
                retrieved.scatter_(1, idx, True)
                term_hit[k] += (retrieved & y).cpu().double().sum(dim=0)

                for b in range(B):
                    if not valid[b]:
                        continue
                    label = self._cardinality_bin(int(true_counts[b].item()))
                    card_recall[k][label]["sum"] += float(cov[b].item())
                    card_recall[k][label]["n"] += 1

        if not all_scores:
            raise RuntimeError("Empty validation loader.")

        score_cpu = torch.cat(all_scores, dim=0).float()
        true_cpu = torch.cat(all_true, dim=0).float()
        score_np = score_cpu.numpy().astype(np.float32)
        true_np = true_cpu.numpy().astype(np.int32)

        logs = {}
        total_valid = int((true_cpu.sum(dim=1) > 0).sum().item())

        for k in ks:
            n = max(1, cand[k]["n"])
            tp, fn = cand[k]["tp"], cand[k]["fn"]

            logs[f"align_R@{k}"] = recall_sum[k] / max(1, total_valid)
            logs[f"cand_coverage@{k}"] = cand[k]["coverage_sum"] / n
            logs[f"oracle_microF@{k}"] = 2.0 * tp / max(1e-12, 2.0 * tp + fn)
            logs[f"oracle_proteinF@{k}"] = cand[k]["protein_f_sum"] / n

            present = term_true > 0
            per_term = torch.zeros_like(term_true)
            per_term[present] = term_hit[k][present] / term_true[present]
            logs[f"macro_term_recall@{k}"] = (
                float(per_term[present].mean().item()) if present.any() else 0.0
            )

            for label, st in card_recall[k].items():
                logs[f"R@{k}_card_{label}"] = st["sum"] / max(1, st["n"])

        if rank_values:
            ranks = torch.cat(rank_values)
            logs["positive_rank_mean"] = float(ranks.mean().item())
            logs["positive_rank_median"] = float(torch.median(ranks).item())
            logs["positive_rank_worst_mean"] = float(np.mean(worst_rank_values))

            pbc = float(np.mean(pbc_values))
            logs["positive_block_coverage"] = pbc
            logs["positive_block_violation"] = 1.0 - pbc

            for q in self.cfg.positive_rank_quantiles:
                logs[f"positive_rank_q{int(round(q * 100))}"] = float(
                    torch.quantile(ranks, float(q)).item()
                )

            logs["align_MRR"] = float(np.mean(reciprocal_rank_values))
            logs["align_nDCG@10"] = float(np.mean(ndcg10_values))

        cardinality_table = []
        ordered_bins = (
            "1_5", "6_10", "11_20", "21_40",
            "41_80", "81_160", "161plus",
        )

        for label in ordered_bins:
            st = card_rank.get(label)
            if not st or st["protein_n"] == 0:
                continue

            ranks_bin = torch.cat(st["positive_ranks"])
            pbc_bin = st["pbc_sum"] / st["protein_n"]
            pbv_bin = 1.0 - pbc_bin

            logs[f"N_card_{label}"] = int(st["protein_n"])
            logs[f"PBC_card_{label}"] = float(pbc_bin)
            logs[f"PBV_card_{label}"] = float(pbv_bin)
            logs[f"positive_rank_median_card_{label}"] = float(
                torch.median(ranks_bin).item()
            )
            logs[f"positive_rank_p90_card_{label}"] = float(
                torch.quantile(ranks_bin, 0.90).item()
            )

            row = {
                "cardinality_bin": label,
                "n_proteins": int(st["protein_n"]),
                "PBC": float(pbc_bin),
                "PBV": float(pbv_bin),
                "positive_rank_median": float(torch.median(ranks_bin).item()),
                "positive_rank_p90": float(torch.quantile(ranks_bin, 0.90).item()),
            }
            for k in ks:
                rec = card_recall[k].get(label)
                row[f"R@{k}"] = (
                    rec["sum"] / max(1, rec["n"])
                    if rec is not None else float("nan")
                )
            cardinality_table.append(row)

        # Downstream threshold metrics remain separate from retrieval metrics.
        tmin, tmax = float(score_np.min()), float(score_np.max())
        best_f = 0.0
        if np.isfinite(tmin) and np.isfinite(tmax) and tmin != tmax:
            for threshold in np.linspace(tmin, tmax, 101, dtype=np.float32):
                pred = (score_np >= threshold).astype(np.int32)
                tp = (pred & true_np).sum()
                fp = (pred & (1 - true_np)).sum()
                fn = ((1 - pred) & true_np).sum()
                precision = tp / (tp + fp + 1e-12)
                recall = tp / (tp + fn + 1e-12)
                best_f = max(
                    best_f,
                    float(2.0 * precision * recall / (precision + recall + 1e-12)),
                )

        logs["obs_fmax"] = best_f
        logs["obs_aupr"] = float(compute_term_aupr(true_np, score_np))
        protein_fmax, threshold = compute_protein_centric_fmax(
            true_np, score_np, num_thresholds=101
        )
        logs["obs_protein_fmax"] = float(protein_fmax)
        logs["obs_protein_fmax_threshold"] = float(threshold)

        # Raw diagnostics for RetrieverWandbLogger. Trainer does not construct
        # W&B tables/plots.
        self.last_eval_diagnostics = {
            "epoch": int(epoch_idx),
            "positive_ranks": (
                torch.cat(rank_values).numpy()
                if rank_values else np.asarray([], dtype=np.float32)
            ),
            "cardinality_table": cardinality_table,
            "go_ids": list(self.eval_id_list),
            "segment_names": list(self._segment_names),
            "go_segment_weights": (
                final_segment_weights.numpy()
                if final_segment_weights is not None else None
            ),
        }

        return logs