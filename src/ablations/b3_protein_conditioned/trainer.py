from __future__ import annotations

from collections import defaultdict
import json
import os
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from src.configs.data_classes import TrainerConfig
from src.metrics.validated_metrics import compute_protein_centric_fmax, compute_term_aupr
from src.metrics.validated_metrics import compute_retrieval_metrics
from .aligner import ProteinGoAligner
from src.metrics.validated_metrics import protein_centric_f_at_threshold


def multi_positive_full_go_infonce(scores, pos_mask, temperature):
    """Experiment F1: per-positive full-GO InfoNCE, protein-balanced.

    Each annotated GO contributes its own log-softmax loss. No IC weights.
    The function name is retained for compatibility with the existing trainer.
    """
    if scores.ndim != 2 or pos_mask.shape != scores.shape:
        raise ValueError("InfoNCE expects scores and pos_mask with shape [B,G].")
    tau = max(float(temperature), 1e-8)
    pos_mask = pos_mask.bool()
    counts = pos_mask.sum(dim=1)
    if (counts == 0).any():
        raise RuntimeError("InfoNCE received a protein with zero positives.")
    logits = scores.float() / tau
    log_probs = F.log_softmax(logits, dim=1)
    per_protein = -(log_probs * pos_mask.float()).sum(dim=1) / counts.float()
    if not torch.isfinite(per_protein).all():
        raise RuntimeError("Non-finite per-positive full-GO InfoNCE.")
    return per_protein.mean()


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


def positive_conditioned_local_pbr(
        scores: torch.Tensor,
        positive_mask: torch.Tensor,
        k: int = 128,
        tau: float = 0.05,
        go_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Local sigmoid PBR with zero margin and optional positive GO weights."""

    if go_weights is not None:
        if go_weights.ndim != 1 or go_weights.numel() != scores.shape[1]:
            raise ValueError("go_weights must have shape [G]")
        if not torch.isfinite(go_weights).all() or (go_weights <= 0).any():
            raise ValueError("go_weights must be finite and strictly positive")

    per_protein = []

    for row_scores, row_mask in zip(scores, positive_mask):
        positives = row_scores[row_mask].float()
        negatives = row_scores[~row_mask].float()
        positive_weights = go_weights[row_mask].float() if go_weights is not None else None

        per_positive = []

        for pos in positives:
            # Select negatives ranked above or tied with this positive.
            with torch.no_grad():
                eligible = torch.where(
                    negatives.detach() >= pos.detach()
                )[0]

                if eligible.numel() > 0:
                    closest = torch.topk(
                        negatives.detach()[eligible],
                        k=min(k, eligible.numel()),
                        largest=False,
                    ).indices
                    selected = eligible[closest]

            if eligible.numel() == 0:
                per_positive.append(pos * 0.0)
            else:
                violations = (negatives[selected] - pos) / tau
                per_positive.append(
                    torch.sigmoid(violations).mean()
                )

        values = torch.stack(per_positive)
        if positive_weights is None:
            per_protein.append(values.mean())
        else:
            per_protein.append((values * positive_weights).sum() / positive_weights.sum())

    return torch.stack(per_protein).mean()


class OppTrainer:
    """Branch-independent full-GO training with real score-level aggregation.

    The GO cache may also contain an ``is_a`` text segment, but the scored
    aggregation channels are name and definition. Individual parent names
    remain inputs to the separate parent consistency correction.
    """

    scoring_segment_names = ["name", "definition"]

    def __init__(self, cfg: TrainerConfig, ctx, go_encoder, wandb_run=None):
        self.cfg = cfg
        self.go_representation = str(getattr(cfg, "go_representation", "segments"))
        if self.go_representation not in {"single_text", "segments"}:
            raise ValueError("go_representation must be single_text or segments")
        # B0 has one complete text representation, not a synthetic name channel.
        self.scoring_segment_names = [] if self.go_representation == "single_text" else ["name", "definition"]
        self.pbr_mode = str(getattr(cfg, "pbr_mode", "pure")).lower()
        if self.pbr_mode not in {"pure", "local"}:
            raise ValueError(f"Unknown pbr_mode={self.pbr_mode!r}; expected pure or local")
        self.pbr_local_k = int(getattr(cfg, "pbr_local_k", 128))
        self.frequency_aware_local = bool(getattr(cfg, "frequency_aware_local", False))
        self._frequency_weights_gpu = None
        if self.pbr_local_k < 1:
            raise ValueError("pbr_local_k must be >= 1")
        self.ctx = ctx
        self.device = torch.device(cfg.device)
        self.wandb_run = wandb_run  # compatibility only; W&B logging is external
        self._global_step = 0
        self.last_eval_diagnostics = {}

        if self.go_representation == "segments" and ctx.go_encoder_output_mode != "segment_pooled":
            raise ValueError("Retriever v2 requires segment_pooled GO mode.")
        if cfg.use_lora:
            raise ValueError("Retriever v2 requires use_lora=False.")

        self.model = ProteinGoAligner(
            d_h=cfg.d_h,
            d_g=cfg.d_g,
            d_z=cfg.d_z,
            go_encoder=go_encoder,
            normalize=True,
            # Legacy constructor label retained only for state-dict compatibility.
            # The actual scoring representation is selected by protein_mode.
            protein_pool_type="mean_attn_gate",
            go_pool_type=ctx.go_pool_type,
            go_segment_representation_mode=ctx.go_segment_representation_mode,
            go_representation=self.go_representation,
            protein_mode=str(getattr(cfg, "protein_mode", "local")),
            segment_aggregation=str(getattr(cfg, "segment_aggregation", "learned")),
            window_length=int(getattr(cfg, "window_length", 128)),
            window_stride=int(getattr(cfg, "window_stride", 64)),
            mil_tau=float(getattr(cfg, "mil_tau", 0.1)),
            parent_strength=float(getattr(cfg, "parent_strength", 0.15)),
            go_chunk_size=int(getattr(cfg, "go_chunk_size", 512)),
            protein_conditioning=getattr(cfg, "protein_conditioning", True),
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
        if len(self.eval_id_list) != len(set(self.eval_id_list)):
            raise RuntimeError("Active GO universe contains duplicate GO IDs.")
        self._go_ids_cpu = torch.tensor(self.eval_id_list, dtype=torch.long)
        self._go_id2col = {g: i for i, g in enumerate(self.eval_id_list)}

        # Optional frozen bank cache. Training uses only ctx.eval_id_list (seen GO),
        # even when the cache contains the full seen + temporal universe.
        # Explicit configuration wins over a shell environment left by another run.
        if self.go_representation == "single_text":
            cache_path = getattr(cfg, "single_text_bank_path", None)
            if cache_path is None:
                cache_path = getattr(getattr(ctx, "go_cache", None), "cache_path", None)
            if cache_path is None:
                raise RuntimeError("single_text_bank_path is required; prepare_data creates the B0 bank")
            self._load_frozen_single_text_bank(cache_path)
            self._text_embs_gpu = self._text_embs_cpu.to(self.device, non_blocking=True)
            self._scoring_indices = []
            self._scoring_present_cpu = None
        else:
            cache_path = getattr(cfg, "mzsgo_frozen_bank", None)
            if cache_path is None:
                cache_path = os.environ.get("MZSGO_FROZEN_GO_BANK", "").strip()
            if cache_path:
                self._load_frozen_go_bank(cache_path)
            else:
                self._build_frozen_segment_bank()
                self._build_parent_bank()

            # Frozen bank is immutable, while the shared projection is trainable.
            self._segment_embs_gpu = self._segment_embs_cpu.to(self.device, non_blocking=True)
            self._segment_present_gpu = self._segment_present_cpu.to(self.device, non_blocking=True)
            self._parent_bank_gpu = self._parent_bank_cpu.to(self.device, non_blocking=True)
            self._parent_present_gpu = self._parent_present_cpu.to(self.device, non_blocking=True)

            missing_channels = set(self.scoring_segment_names) - set(self._segment_names)
            if missing_channels:
                raise RuntimeError(f"Frozen bank lacks scoring channels: {sorted(missing_channels)}")
            self._scoring_indices = [self._segment_names.index(n) for n in self.scoring_segment_names]
            self._scoring_present_cpu = self._segment_present_cpu[:, self._scoring_indices]
            if not self._scoring_present_cpu[:, 0].all():
                missing_name_ids = self._go_ids_cpu[~self._scoring_present_cpu[:, 0]].tolist()
                raise RuntimeError(f"Active GO terms without a name segment: {missing_name_ids[:20]}")

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
        print("[AGGREGATION] architecture:", self.model.architecture_metadata())
        print("[AGGREGATION] active GO terms:", len(self.eval_id_list))
        print("[AGGREGATION] trainable sample:", names[:40])

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

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @torch.no_grad()
    def _load_frozen_single_text_bank(self, cache_path):
        """Select active rows from the independent, validated B0 text bank."""
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"Single-text GO bank not found; run prepare_data first: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "mzsgo_frozen_single_text_v1":
            raise RuntimeError("B0 requires a true single-text GO bank, not a segmented bank")
        metadata = payload.get("metadata", {})
        jsonl = getattr(self.cfg, "canonical_go_jsonl", None)
        if not jsonl or not os.path.isfile(jsonl):
            raise FileNotFoundError("canonical_go_jsonl is required for the single-text bank")
        if metadata.get("canonical_jsonl_sha256") != self._file_sha256(jsonl):
            raise RuntimeError("Single-text GO source SHA256 mismatch")
        if (metadata.get("representation") != "single_text"
                or metadata.get("text_fields") != ["name", "definition"]
                or metadata.get("parent_text_included") is not False
                or metadata.get("pooling") != "mean" or metadata.get("normalized") is not False):
            raise RuntimeError("Single-text GO cache representation/pooling policy mismatch")
        expected_length = int(getattr(self.cfg, "go_single_text_max_len", 128))
        if metadata.get("go_single_text_max_len") != expected_length:
            raise RuntimeError("Single-text GO token length differs from cache")
        encoder_config = getattr(getattr(self.model.go_encoder, "model", None), "config", None)
        expected_model = (getattr(self.cfg, "encoder_name", None)
                          or getattr(self.model.go_encoder, "model_name", None)
                          or getattr(encoder_config, "_name_or_path", None))
        if expected_model is not None and str(metadata.get("model")) != str(expected_model):
            raise RuntimeError("BioMedBERT checkpoint differs from the single-text cache")
        text_store = getattr(self.ctx, "go_text_store", None)
        if bool(getattr(text_store, "is_go_segmented", False)):
            raise RuntimeError("B0 must use an unsegmented GO text store")
        if text_store is not None and int(text_store.max_len) != expected_length:
            raise RuntimeError("B0 text-store length differs from its cached bank")
        # prepare_data verifies strings/tokens as well as metadata. Require that
        # verified identity when a prepared context carries it.
        prepared_metadata = getattr(getattr(self.ctx, "go_cache", None), "metadata", None)
        if prepared_metadata is not None and prepared_metadata != metadata:
            raise RuntimeError("Single-text bank differs from the prepared data identity")
        full_ids = [int(g) for g in payload.get("go_ids", [])]
        if not full_ids or len(full_ids) != len(set(full_ids)):
            raise RuntimeError("Single-text cache GO IDs must be nonempty and unique")
        vectors = payload.get("text_embs")
        if not torch.is_tensor(vectors) or vectors.shape != (len(full_ids), int(self.cfg.d_g)):
            raise RuntimeError("Single-text bank must contain exactly one d_g vector per GO")
        if not torch.isfinite(vectors).all():
            raise RuntimeError("Nonfinite single-text GO embeddings")
        lookup = {g: i for i, g in enumerate(full_ids)}
        missing = sorted(set(self.eval_id_list) - set(lookup))
        if missing:
            raise RuntimeError(f"Active GO IDs missing from single-text bank: {missing[:20]}")
        index = torch.tensor([lookup[g] for g in self.eval_id_list], dtype=torch.long)
        self._text_embs_cpu = vectors.index_select(0, index).float().contiguous()
        self._single_text_metadata = metadata
        print(f"[GO-SINGLE-TEXT-CACHE] loaded={cache_path} active={len(index)}/{len(full_ids)} "
              f"vectors={tuple(self._text_embs_cpu.shape)}")

    @torch.no_grad()
    def _load_frozen_go_bank(self, cache_path):
        """Load immutable BioMedBERT banks, selecting active GO IDs in exact order.

        The full cache may contain temporal GO terms. They are not used during
        training unless the caller explicitly includes them in eval_id_list.
        """
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"Frozen GO bank not found: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "mzsgo_frozen_segment_parent_v1":
            raise RuntimeError("Unexpected frozen GO bank format")

        metadata = payload.get("metadata", {})
        jsonl = getattr(self.cfg, "canonical_go_jsonl", None)
        if jsonl is None:
            jsonl = os.environ.get("EXPH_GO_JSONL", "").strip()
        if not jsonl:
            raise RuntimeError("canonical_go_jsonl is required when using the frozen bank")
        if not os.path.isfile(jsonl):
            raise FileNotFoundError(jsonl)
        if self._file_sha256(jsonl) != metadata.get("canonical_jsonl_sha256"):
            raise RuntimeError("GO text SHA256 mismatch: cache is stale or wrong ontology")
        expected_model = getattr(self.model.go_encoder, "model_name", None)
        if expected_model is not None and str(expected_model) != str(metadata.get("model")):
            raise RuntimeError("BioMedBERT checkpoint differs from cached checkpoint")
        expected_len = int(self.ctx.go_text_store.segment_max_len)
        if int(metadata.get("segment_max_len", -1)) != expected_len:
            raise RuntimeError("GO segment token length differs from cache")
        if metadata.get("normalized") is not False:
            raise RuntimeError("Expected unnormalized frozen GO embeddings")

        full_ids = [int(x) for x in payload["go_ids"]]
        if len(full_ids) != len(set(full_ids)):
            raise RuntimeError("Duplicate GO IDs in cache")
        lookup = {g: i for i, g in enumerate(full_ids)}
        missing = sorted(set(self.eval_id_list) - set(lookup))
        if missing:
            raise RuntimeError(f"Active GO IDs missing from frozen bank: {missing[:20]}")
        idx = torch.tensor([lookup[g] for g in self.eval_id_list], dtype=torch.long)
        names = list(payload["segment_names"])
        expected_names = list(self.ctx.go_text_store.segment_names)
        if names != expected_names:
            raise RuntimeError(f"GO segment order mismatch: {names} vs {expected_names}")

        seg = payload["segment_embs"]
        seg_present = payload["segment_present"]
        par = payload["parent_embs"]
        par_present = payload["parent_present"]
        n = len(full_ids)
        d = int(self.cfg.d_g)
        if seg.ndim != 3 or seg.shape != (n, len(names), d):
            raise RuntimeError(f"Invalid segment bank shape: {tuple(seg.shape)}")
        if seg_present.shape != seg.shape[:2]:
            raise RuntimeError("Invalid segment presence mask")
        if par.ndim != 3 or par.shape[0] != n or par.shape[2] != d:
            raise RuntimeError(f"Invalid parent bank shape: {tuple(par.shape)}")
        if par_present.shape != par.shape[:2]:
            raise RuntimeError("Invalid parent presence mask")
        if not torch.isfinite(seg).all() or not torch.isfinite(par).all():
            raise RuntimeError("Nonfinite GO embeddings")
        self._segment_embs_cpu = seg.index_select(0, idx).float().contiguous()
        self._segment_present_cpu = seg_present.index_select(0, idx).bool().contiguous()
        self._parent_bank_cpu = par.index_select(0, idx).float().contiguous()
        self._parent_present_cpu = par_present.index_select(0, idx).bool().contiguous()
        self._segment_names = names
        if not self._segment_present_cpu.any(dim=1).all():
            raise RuntimeError("Active GO term with no present segments")
        print(f"[GO-CACHE] loaded={cache_path} active={len(idx)}/{n} "
              f"segments={tuple(self._segment_embs_cpu.shape)} "
              f"parents={tuple(self._parent_bank_cpu.shape)}")

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
        if seg_mask.shape != seg_ids.shape or present.shape != (G, S):
            raise RuntimeError("Segment token/presence masks do not match the GO token bank.")
        if G != len(self.eval_id_list) or (present.sum(1) == 0).any():
            raise RuntimeError("Invalid segmented GO bank.")
        segment_names = list(toks.get("segment_names", []))
        if len(segment_names) != S or len(set(segment_names)) != S:
            raise RuntimeError("Segment names must uniquely identify every GO bank channel.")

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
            if not torch.is_tensor(out) or out.shape != (idx.numel(), D):
                raise RuntimeError("Unexpected frozen GO encoder output.")
            if not torch.isfinite(out).all():
                raise RuntimeError("Nonfinite frozen GO encoder output.")
            flat_bank.index_copy_(0, idx.cpu(), out.float().cpu())

        self._segment_embs_cpu = bank.contiguous()
        self._segment_present_cpu = present.cpu().contiguous()
        self._segment_names = segment_names
        print("[GO-SEGMENT-BANK]", tuple(bank.shape), self._segment_names)

    @torch.no_grad()
    def _build_parent_bank(self):
        """Resolve parent IDs from canonical JSONL; encode parent names with frozen encoder."""
        path = getattr(self.cfg, "canonical_go_jsonl", None)
        if path is None:
            path = os.environ.get("EXPH_GO_JSONL", "").strip()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Canonical GO JSONL is required to build the parent bank: {path}")
        records = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                records[row["go_id"]] = row

        def key(g):
            s = str(g)
            return s if s.startswith("GO:") else f"GO:{int(g):07d}"

        parents = []
        for gid in self.eval_id_list:
            row = records.get(key(gid))
            if row is None:
                raise RuntimeError(f"Missing canonical record: {gid}")
            ids = row.get("is_a_parent_ids", [])
            # Never infer ID/name alignment from is_a_parents order.
            parents.append([pid for pid in ids if pid in records and records[pid].get("name")])
        P = max(1, max(map(len, parents)))
        unique = sorted({pid for ids in parents for pid in ids})
        vectors = {}
        if unique:
            tokenizer = self.ctx.go_text_store.tokenizer
            bs = max(1, int(self.cfg.eval_go_bs))
            self.model.go_encoder.eval()
            for st in range(0, len(unique), bs):
                ids = unique[st:st + bs]
                enc = tokenizer([records[x]["name"] for x in ids], truncation=True,
                                max_length=self.ctx.go_text_store.segment_max_len,
                                padding="max_length", return_tensors="pt")
                out = self.model.go_encoder(input_ids=enc["input_ids"].to(self.device),
                                            attention_mask=enc["attention_mask"].to(self.device),
                                            output_mode="pooled")
                if isinstance(out, tuple):
                    out = out[0]
                elif isinstance(out, dict):
                    out = out.get("pooled", out.get("pooler_output"))
                if not torch.is_tensor(out) or out.shape != (len(ids), self.cfg.d_g):
                    raise RuntimeError("Unexpected parent encoder output shape")
                if not torch.isfinite(out).all():
                    raise RuntimeError("Nonfinite parent encoder output")
                for pid, vec in zip(ids, out.float().cpu()): vectors[pid] = vec
        bank = torch.zeros(len(parents), P, self.cfg.d_g, dtype=torch.float32)
        present = torch.zeros(len(parents), P, dtype=torch.bool)
        for i, ids in enumerate(parents):
            for j, pid in enumerate(ids):
                bank[i, j] = vectors[pid]
                present[i, j] = True
        self._parent_bank_cpu = bank
        self._parent_present_cpu = present
        print(f"[EXPERIMENT-H] parent bank={tuple(bank.shape)} unique_parents={len(unique)} "
              f"GO_with_parents={int(present.any(1).sum())}")

    def _positive_mask(self, batch):
        B = len(batch["pos_go_global"])
        G = len(self.eval_id_list)
        mask = torch.zeros(B, G, dtype=torch.bool, device=self.device)
        for b, gids in enumerate(batch["pos_go_global"]):
            cols = []
            for gid in gids.detach().cpu().tolist():
                col = self._go_id2col.get(int(gid))
                if col is None:
                    raise RuntimeError(
                        f"Protein {batch['protein_ids'][b]} has positive GO:{int(gid):07d} "
                        "outside the active GO universe. Refusing to silently drop a label."
                    )
                cols.append(col)
            if not cols:
                raise RuntimeError(
                    f"Protein {batch['protein_ids'][b]} has no positive in active GO universe."
                )
            mask[b, torch.tensor(cols, device=self.device, dtype=torch.long)] = True
        return mask

    def _aggregation_weights(self, *, protein_summary=None, go_gate_context=None):
        """Live, differentiable weights for the exact scored text channels."""
        if self.go_representation == "single_text":
            return None
        ni, di = self._scoring_indices
        return self.model.aggregation_weights(
            self._segment_embs_gpu[:, ni],
            self._segment_embs_gpu[:, di],
            self._segment_present_gpu[:, di],
            protein_summary=protein_summary,
            go_gate_context=go_gate_context,
        )

    def _prepare_go_gate_context(self):
        ni, di = self._scoring_indices
        return self.model.prepare_go_gate_context(
            self._segment_embs_gpu[:, ni], self._segment_embs_gpu[:, di],
            self._segment_present_gpu[:, di],
        )

    def _score_full_go(self, batch, segment_weights=None, go_gate_context=None):
        H = batch["prot_emb_pad"].to(self.device, non_blocking=True)
        mask = batch["prot_attn_mask"].to(self.device, non_blocking=True).bool()
        if self.go_representation == "single_text":
            if segment_weights is not None:
                raise ValueError("B0 single-text scoring has no segment weights")
            return self.model.score_single_text(H.float(), mask, self._text_embs_gpu), None
        name_idx, definition_idx = self._scoring_indices
        seg = self._segment_embs_gpu
        present = self._segment_present_gpu
        weights = segment_weights
        if not self.model.protein_conditioning and weights is None:
            weights = self._aggregation_weights()
        # These same live weights control the scorer and the diagnostics.
        scores = self.model.score_fine_grained(
            H.float(), mask,
            seg[:, name_idx], seg[:, definition_idx],
            present[:, definition_idx],
            self._parent_bank_gpu,
            self._parent_present_gpu,
            None,  # Legacy global_go_raw argument is not used by this scorer.
            segment_weights=weights,
            go_gate_context=go_gate_context,
        )
        if self.model.protein_conditioning:
            # Scores already carry the live gate graph. The returned pair weights
            # are an exact detached diagnostic copy, never used in the loss.
            weights = self.model.last_go_segment_info["segment_weights"]
        return scores, weights

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
        # Select the original all-negative PBR or positive-conditioned local PBR.
        # Pure PBR is SUM over negatives, local PBR is MEAN over selected
        # negatives. Their coefficients are deliberately NOT interchangeable.
        if float(self.cfg.pbr_lambda) == 0.0:
            # Do not mine negatives or build a pairwise graph for a disabled loss.
            l_pbr = scores.new_zeros(())
            pbr_stats = {"pbr_enabled": 0.0, "pbr_weighted": l_pbr}
        elif self.pbr_mode == "pure":
            l_pbr, pbr_diagnostics = positive_block_rank_loss(
                scores, pos_mask,
                margin=float(self.cfg.pbr_margin),
                tau=float(self.cfg.pbr_tau),
            )
            pbr_stats = {
                "pbr_enabled": 1.0,
                "pbr": l_pbr.detach(),
                "pbr_weighted": (float(self.cfg.pbr_lambda) * l_pbr).detach(),
                **pbr_diagnostics,
            }
        else:
            if abs(float(self.cfg.pbr_margin)) > 1e-12:
                raise ValueError("Local PBR requires loss.pbr_margin=0.0")
            if self.frequency_aware_local and self._frequency_weights_gpu is None:
                raise RuntimeError("Frequency-aware Local PBR enabled but weights not initialized")
            l_pbr = positive_conditioned_local_pbr(
                scores, pos_mask,
                k=self.pbr_local_k,
                tau=float(self.cfg.pbr_tau),
                go_weights=self._frequency_weights_gpu if self.frequency_aware_local else None,
            )
            pbr_stats = {
                "pbr_enabled": 1.0,
                "pbr": l_pbr.detach(),
                "local_pbr": l_pbr.detach(),
                "local_pbr_weighted": (float(self.cfg.pbr_lambda) * l_pbr).detach(),
            }

        l_total = (
                float(self.cfg.lambda_con) * l_con
                + float(self.cfg.pbr_lambda) * l_pbr
        )

        if not torch.isfinite(l_total):
            raise RuntimeError("Non-finite Retriever v2 total loss.")

        # Expensive full-GO ranking diagnostics are logging-only, not part of the loss.
        # Retain the logging cadence without synchronizing full gate arrays each batch.
        score_diag = dict(self.model.last_h_diagnostics)
        rank_diag = {}
        diagnostic_every = max(1, int(getattr(self.cfg, "diagnostics_log_every_steps", 500)))
        log_diagnostics = (self._global_step + 1) % diagnostic_every == 0
        if log_diagnostics:
            with torch.no_grad():
                score_diag.update(self._score_diagnostics(scores.detach(), pos_mask))
                rank_diag = self._rank_diagnostics(scores.detach(), pos_mask)
        self._global_step += 1

        out = {
            "total": l_total,
            "contrastive": l_con,
            "ranking_loss": l_pbr.detach(),
            # Retain legacy logging keys for existing main.py/W&B integration.
            # BCE is not computed or optimized in this experiment.
            "bce": l_total.detach() * 0.0,
            "bce_weighted": l_total.detach() * 0.0,
            **pbr_stats,
            **score_diag,
            **rank_diag,
        }

        if (seg_weights is not None and self.cfg.log_go_segment_weights
                and (log_diagnostics or self._global_step == 1)):
            pair_weights = self.model.protein_conditioning
            means = seg_weights.detach().float().mean((0, 1) if pair_weights else 0).cpu()
            names = self.scoring_segment_names
            for i, name in enumerate(names):
                prefix = "protein_go_segment_weight" if pair_weights else "go_segment_weight"
                out[f"{prefix}/{name}"] = float(means[i].item())

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
        # Cache only GO features for B3-PC. Pair weights are recomputed from each
        # original protein batch, under the caller's unchanged autocast scope.
        pair_conditioned = self.model.protein_conditioning
        evaluation_context = self._prepare_go_gate_context() if pair_conditioned else None
        evaluation_weights = None if pair_conditioned else self._aggregation_weights()
        final_segment_weights = (evaluation_weights.detach().float().cpu()
                                 if evaluation_weights is not None else None)
        pair_weight_rows = []
        pair_row_limit = int(getattr(self.cfg, "diagnostics_max_table_rows", 200))
        collect_pair_weights = pair_conditioned and bool(self.cfg.log_go_segment_weights)
        pair_weight_sum = torch.zeros((G, 2), dtype=torch.float64) if collect_pair_weights else None
        pair_weight_square_sum = torch.zeros((G, 2), dtype=torch.float64) if collect_pair_weights else None
        pair_weight_protein_count = 0

        parent_adjustment_sum = 0.0
        parent_adjustment_proteins = 0

        for batch in loader:
            scores, segment_weights = self._score_full_go(
                batch, segment_weights=evaluation_weights, go_gate_context=evaluation_context,
            )
            parent_adjustment_sum += scores.shape[0] * self.model.last_h_diagnostics[
                "h/parent_adjustment_mean_abs"
            ]
            parent_adjustment_proteins += scores.shape[0]
            y = self._positive_mask(batch)
            if collect_pair_weights:
                pair_weight_sum += segment_weights.double().sum(0).cpu()
                pair_weight_square_sum += segment_weights.double().square().sum(0).cpu()
                pair_weight_protein_count += scores.shape[0]
                # Deterministic genuine protein/GO pairs, not averaged GO weights.
                remaining = pair_row_limit - len(pair_weight_rows)
                if remaining > 0:
                    sample_go_count = min(8, G)
                    sample_protein_count = min(scores.shape[0], (remaining + sample_go_count - 1) // sample_go_count)
                    sample_weights = segment_weights[:sample_protein_count, :sample_go_count].float().cpu()
                    for bi in range(sample_protein_count):
                        for gi in range(sample_go_count):
                            if len(pair_weight_rows) >= pair_row_limit:
                                break
                            pair_weight_rows.append({
                                "protein_id": str(batch["protein_ids"][bi]),
                                "go_id": f"GO:{int(self.eval_id_list[gi]):07d}",
                                "name_weight": float(sample_weights[bi, gi, 0]),
                                "definition_weight": float(sample_weights[bi, gi, 1]),
                                "definition_present": bool(self._scoring_present_cpu[gi, 1]),
                            })

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

        logs = compute_retrieval_metrics(score_np, true_np)
        total_valid = int((true_cpu.sum(dim=1) > 0).sum().item())

        for k in ks:
            n = max(1, cand[k]["n"])
            tp, fn = cand[k]["tp"], cand[k]["fn"]

            logs[f"align_R@{k}"] = recall_sum[k] / max(1, total_valid)
            logs[f"micro_recall@{k}"] = tp / max(1.0, tp + fn)
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
        # Exact raw-score micro Fmax, with ties grouped.
        flat_scores = score_np.ravel()
        flat_true = true_np.ravel()
        sort_idx = np.argsort(-flat_scores, kind="stable")
        ordered_scores = flat_scores[sort_idx]
        ordered_true = flat_true[sort_idx]
        boundaries = np.r_[0, np.flatnonzero(
            ordered_scores[1:] != ordered_scores[:-1]
        ) + 1]
        ends = np.r_[boundaries[1:], len(ordered_scores)]
        cumulative_tp = np.cumsum(ordered_true, dtype=np.int64)
        tp_at = cumulative_tp[ends - 1].astype(np.float64)
        pred_at = ends.astype(np.float64)
        total_pos = float(flat_true.sum())
        micro_f = 2.0 * tp_at / np.maximum(pred_at + total_pos, 1.0)
        best_f = float(micro_f.max()) if len(micro_f) else 0.0

        logs["obs_fmax"] = best_f
        logs["obs_macro_term_ap"] = float(compute_term_aupr(true_np, score_np))  # Explicit macro term AP
        logs["obs_aupr"] = logs["obs_macro_term_ap"]  # Backward-compatible alias; NOT trapezoidal AUPR
        protein_fmax, threshold = compute_protein_centric_fmax(
            true_np, score_np
        )
        logs["obs_protein_fmax"] = float(protein_fmax)  # Exact raw-score validation Fmax
        logs["obs_protein_fmax_threshold"] = float(threshold)  # RAW score threshold
        # An operating threshold must be explicitly supplied for this run.
        # Never reuse an unexplained constant across ontology branches/models.
        fixed_t = getattr(self.cfg, "validation_fixed_threshold", None)
        if fixed_t is not None:
            fixed_t = float(fixed_t)
            if not np.isfinite(fixed_t):
                raise ValueError("validation_fixed_threshold must be finite")
            logs["fixed_threshold"] = fixed_t
            logs["protein_F_at_valid_threshold"] = protein_centric_f_at_threshold(
                true_np, score_np, fixed_t
            )
        logs["h/parent_adjustment_mean_abs"] = (
                parent_adjustment_sum / max(1, parent_adjustment_proteins)
        )

        # Raw diagnostics for RetrieverWandbLogger. Trainer does not construct
        # W&B tables/plots.
        frequencies = getattr(self.cfg, "train_frequency_by_go", None)
        if frequencies is not None:
            frequencies = {int(g): int(n) for g, n in frequencies.items()}
        go_term_table = []
        pair_weight_mean = pair_weight_std = None
        if collect_pair_weights and pair_weight_protein_count:
            pair_weight_mean = pair_weight_sum / pair_weight_protein_count
            pair_weight_variance = pair_weight_square_sum / pair_weight_protein_count - pair_weight_mean.square()
            pair_weight_std = pair_weight_variance.clamp_min(0).sqrt()
        if bool(getattr(self.cfg, "log_go_term_metrics", True)):
            for col, gid in enumerate(self.eval_id_list):
                n_true = int(term_true[col].item())
                record = {
                    "go_id": f"GO:{int(gid):07d}",
                    "train_frequency": frequencies.get(int(gid), 0) if frequencies is not None else None,
                    "validation_positive_count": n_true,
                    "has_validation_positive": n_true > 0,
                }
                if final_segment_weights is not None:
                    record.update({
                        "name_weight": float(final_segment_weights[col, 0]),
                        "definition_weight": float(final_segment_weights[col, 1]),
                        "definition_present": bool(self._scoring_present_cpu[col, 1]),
                    })
                elif pair_conditioned:
                    record.update(go_representation="segments", gate_conditioning="protein_and_go_text",
                                  definition_present=bool(self._scoring_present_cpu[col, 1]))
                    if pair_weight_mean is not None:
                        for si, segment_name in enumerate(self.scoring_segment_names):
                            record[f"{segment_name}_weight_mean_across_proteins"] = float(pair_weight_mean[col, si])
                            record[f"{segment_name}_weight_std_across_proteins"] = float(pair_weight_std[col, si])
                else:
                    record["go_representation"] = "single_text"
                for k in ks:
                    record[f"recall@{k}"] = float(term_hit[k][col] / n_true) if n_true else None
                go_term_table.append(record)

        self.last_eval_diagnostics = {
            "epoch": int(epoch_idx),
            "positive_ranks": (
                torch.cat(rank_values).numpy()
                if rank_values and bool(getattr(self.cfg, "log_positive_rank_cdf", True))
                else np.asarray([], dtype=np.float32)
            ),
            "cardinality_table": cardinality_table if bool(getattr(self.cfg, "log_cardinality_metrics", True)) else [],
            "go_ids": list(self.eval_id_list),
            "segment_names": list(self.scoring_segment_names),
            "go_segment_present": (self._scoring_present_cpu.numpy()
                                   if self._scoring_present_cpu is not None else None),
            "go_segment_weight_source": ("scoring" if final_segment_weights is not None
                                         else "not_applicable_protein_conditioned" if pair_conditioned
                                         else "not_applicable_single_text"),
            "protein_go_segment_table": pair_weight_rows,
            "protein_go_segment_weight_source": "actual_scoring_pairs" if pair_conditioned else "not_applicable",
            "protein_go_segment_weight_axes": ["protein", "go", "segment"] if pair_conditioned else None,
            "protein_go_segment_summary_protein_count": pair_weight_protein_count,
            "gate_conditioning": self.model.architecture_metadata()["gate_conditioning"],
            "segment_aggregation": self.model.segment_aggregation,
            "architecture": self.model.architecture_metadata(),
            "go_term_table": go_term_table,
            "go_segment_weights": (
                final_segment_weights.numpy()
                if final_segment_weights is not None else None
            ),
        }

        return logs
