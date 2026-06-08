from __future__ import annotations

import os
import json
import logging
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import types

# Retriever
from src.models.alignment_model import ProteinGoAligner
from src.encoders.go_encoder import BioMedBERTEncoder, LoRAParameters

# Reranker
from src.training.reranker_trainer import RerankerTrainer
from src.models.reranker_model import RerankerMeanPoolConcatMLP, ResidueGoCrossAttentionReranker
from src.datasets.go_text_store import GoTextStore

# dataset + collator
from src.training.collate import ContrastiveEmbCollator
from src.configs.paths import SRC_DIR, go_index_paths
from src.utils.helpers import load_go_texts_by_phase
from src.main import build_datasets, build_stores, build_go_cache

RETRIEVER_YAML_PATH = SRC_DIR / "reranker.yaml"

class CandidateDumpLookup:
    def __init__(self, dump_dir: str | Path, topk: int):
        self.dump_dir = Path(dump_dir)
        self.topk = int(topk)

        self.eval_go_ids = np.load(self.dump_dir / "eval_go_ids.npy", mmap_mode="r").astype(np.int64)
        self.top_cols = np.load(self.dump_dir / "top_go_cols.int32.npy", mmap_mode="r")
        self.top_scores = np.load(self.dump_dir / "top_scores.float32.npy", mmap_mode="r")
        self.top_labels = np.load(self.dump_dir / "top_labels.int8.npy", mmap_mode="r")
        self.true_go_ids = np.load(self.dump_dir / "true_go_ids.npy", mmap_mode="r")

        valid_path = self.dump_dir / "top_valid.int8.npy"
        self.top_valid = np.load(valid_path, mmap_mode="r") if valid_path.exists() else None

        pids_path = self.dump_dir / "protein_ids.json"
        with pids_path.open("r", encoding="utf-8") as f:
            self.protein_ids = [str(x) for x in json.load(f)]

        self.pid_to_row = {pid: i for i, pid in enumerate(self.protein_ids)}

        if self.topk > self.top_cols.shape[1]:
            raise ValueError(
                f"Requested topk={self.topk}, but dump has only {self.top_cols.shape[1]}"
            )

    def get_batch(self, protein_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = []
        missing = []
        for pid in protein_ids:
            pid = str(pid)
            if pid not in self.pid_to_row:
                missing.append(pid)
            else:
                rows.append(self.pid_to_row[pid])

        if missing:
            raise KeyError(f"Missing protein IDs in candidate dump: {missing[:10]}")

        rows_np = np.asarray(rows, dtype=np.int64)

        cols = np.asarray(self.top_cols[rows_np, : self.topk], dtype=np.int64)
        cand_ids = self.eval_go_ids[cols].astype(np.int64)

        scores = np.asarray(self.top_scores[rows_np, : self.topk], dtype=np.float32)
        labels = np.asarray(self.top_labels[rows_np, : self.topk], dtype=np.float32)

        true_ids = np.asarray(self.true_go_ids[rows_np], dtype=np.int64)

        if self.top_valid is not None:
            valid = np.asarray(self.top_valid[rows_np, : self.topk], dtype=np.int8)
        else:
            valid = np.ones_like(labels, dtype=np.int8)

        # Safety for union/no-fill dumps.
        # Invalid/filler candidates often have sentinel scores such as -1e6.
        # These can become -inf under fp16/AMP and later produce NaNs.
        scores = np.nan_to_num(
            scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        scores = np.clip(scores, -20.0, 20.0).astype(np.float32)

        # Invalid candidates should be neutral inputs and never positive labels.
        scores[valid == 0] = 0.0
        labels[valid == 0] = 0.0

        return (
            torch.from_numpy(cand_ids.copy()).long(),
            torch.from_numpy(scores.copy()).float(),
            torch.from_numpy(labels.copy()).float(),
            torch.from_numpy(true_ids.copy()).long(),
            torch.from_numpy(valid.copy()).bool(),
        )

        # Helpers for checkpoint loading
# -----------------------------
def safe_torch_load(path: str, map_location="cpu"):
    """
    PyTorch 2.6+ changed weights_only default.
    Since this is your own checkpoint, load with weights_only=False.
    """
    try:
        torch.serialization.add_safe_globals([PosixPath])
    except Exception:
        pass

    return torch.load(path, map_location=map_location, weights_only=False)


def extract_sub_state(state: dict, prefix: str) -> dict:
    out = {}
    p = prefix if prefix.endswith(".") else (prefix + ".")
    for k, v in state.items():
        if k.startswith(p):
            out[k[len(p):]] = v
    return out

def strip_prefix_from_state(sd: dict, prefix: str) -> dict:
    p = prefix if prefix.endswith(".") else prefix + "."
    out = {}
    for k, v in sd.items():
        if k.startswith(p):
            out[k[len(p):]] = v
        else:
            out[k] = v
    return out

def _go_str_to_int(x) -> int:
    """
    Converts:
      GO:0008150 -> 8150
      "8150" -> 8150
      8150 -> 8150
    """
    if isinstance(x, int):
        return x

    s = str(x).strip()
    if s.upper().startswith("GO:"):
        s = s.split(":", 1)[1]
    return int(s)


def load_child_to_parents_json(path: str) -> Dict[int, List[int]]:
    """
    Expected JSON format:
    {
      "GO:0000001": [["GO:0048308", "is_a"], ["GO:0048311", "is_a"]],
      "GO:0000002": [["GO:0007005", "is_a"]]
    }

    Returns:
      {child_go_int: [parent_go_int, ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    child_to_parents: Dict[int, List[int]] = {}

    for child, parents in raw.items():
        child_id = _go_str_to_int(child)

        if not parents:
            child_to_parents[child_id] = []
            continue

        parent_ids = []
        for item in parents:
            if isinstance(item, (list, tuple)):
                parent_go = item[0]
            else:
                parent_go = item
            parent_ids.append(_go_str_to_int(parent_go))

        child_to_parents[child_id] = parent_ids

    return child_to_parents


def build_go_encoder_from_retriever_ckpt(
    ckpt_path: str,
    *,
    model_name: str,
    device: str,
    max_length: int,
    enable_lora: bool,
    use_special_tokens: bool,
    lora_parameters: LoRAParameters | None,
    gradient_checkpointing: bool = False,
):
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = state.get("model", state)

    go_sd = extract_sub_state(state, "go_encoder")
    go_sd = strip_prefix_from_state(go_sd, "model")

    if not go_sd:
        raise RuntimeError("Checkpoint içinde go_encoder.* bulunamadı.")

    enc = BioMedBERTEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        enable_lora=enable_lora,
        use_special_tokens=use_special_tokens,
        lora_parameters=lora_parameters,
        gradient_checkpointing=gradient_checkpointing,
    )
    enc.eval()

    missing, unexpected = enc.model.load_state_dict(go_sd, strict=False)

    print(f"[go_encoder load] missing={len(missing)} unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print("[go_encoder load] unexpected sample:", unexpected[:20])
    if len(missing) > 0:
        print("[go_encoder load] missing sample:", missing[:20])

    return enc


# -----------------------------
# OOM-safe retriever scoring
# -----------------------------
@torch.no_grad()
def retriever_topk_ids_chunked(
    retriever: ProteinGoAligner,
    H: torch.Tensor,
    valid_mask: torch.Tensor,
    G_once_cpu: torch.Tensor,
    eval_go_ids: List[int],
    topk: int,
    device: torch.device,
    chunk_k: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      cand_ids: [B,K] global GO ids on CPU
      cand_scores: [B,K] retriever scores on CPU, aligned with cand_ids
    """
    B = H.size(0)
    eval_ids_t = torch.as_tensor(eval_go_ids, dtype=torch.long)

    K = int(min(topk, len(eval_go_ids)))
    best_scores = torch.full((B, K), -1e9, device=device)
    best_idx = torch.full((B, K), -1, dtype=torch.long, device=device)

    Geval = G_once_cpu.size(0)
    retriever.eval()

    for s in range(0, Geval, chunk_k):
        e = min(Geval, s + chunk_k)
        G_chunk = G_once_cpu[s:e].to(device, non_blocking=True)
        C = G_chunk.size(0)
        G_eval = G_chunk.unsqueeze(0).expand(B, C, G_chunk.size(1)).contiguous()

        sc = retriever(H=H, G=G_eval, mask=valid_mask, return_alpha=False)
        if isinstance(sc, tuple):
            sc = sc[0]

        cur_scores, cur_rel = torch.topk(sc, k=min(K, C), dim=1)
        cur_idx = cur_rel + s

        merged_scores = torch.cat([best_scores, cur_scores], dim=1)
        merged_idx = torch.cat([best_idx, cur_idx], dim=1)

        new_scores, new_pos = torch.topk(merged_scores, k=K, dim=1)
        new_idx = merged_idx.gather(1, new_pos)

        best_scores, best_idx = new_scores, new_idx

    cand_ids = eval_ids_t.index_select(0, best_idx.reshape(-1).cpu()).view(B, K)
    return cand_ids, best_scores.detach().cpu()


# -----------------------------
# Metrics
# -----------------------------
def compute_global_fmax_aupr_from_items(items: List[Tuple[float, int]], n_true_total: int) -> Dict[str, float]:
    if n_true_total <= 0 or len(items) == 0:
        return {"fmax": 0.0, "aupr": 0.0}

    items.sort(key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    best_f = 0.0

    aupr = 0.0
    prev_rec = 0.0

    for score, is_true in items:
        if is_true:
            tp += 1
        else:
            fp += 1

        prec = tp / max(1, tp + fp)
        rec = tp / n_true_total

        if prec + rec > 0:
            f = 2 * prec * rec / (prec + rec)
            if f > best_f:
                best_f = f

        dr = rec - prev_rec
        if dr > 0:
            aupr += prec * dr
            prev_rec = rec

    return {"fmax": float(best_f), "aupr": float(aupr)}


def count_true_total(pos_go_global: List[torch.Tensor]) -> int:
    n = 0
    for t in pos_go_global:
        if t is None:
            continue
        n += int(t.numel())
    return n


# -----------------------------
# Checkpoint save helper
# -----------------------------
def _get_optimizer(rr_trainer):
    return getattr(rr_trainer, "opt", None) or getattr(rr_trainer, "optimizer", None)


def _get_scaler(rr_trainer):
    return getattr(rr_trainer, "scaler", None)


def save_checkpoint(
    out_dir: str,
    step: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metrics: Dict[str, float],
    tag: str,
    scaler=None,
    args=None,
    path_override: str | Path | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    if path_override is None:
        path = Path(out_dir) / f"ckpt_{tag}_step{step}_epoch{epoch}.pt"
    else:
        path = Path(path_override)
        path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": int(step),
        "global_step": int(step),
        "epoch": int(epoch),
        "metrics": metrics,
        "model": model.state_dict(),
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    if scaler is not None:
        try:
            payload["scaler"] = scaler.state_dict()
        except Exception:
            pass

    if args is not None and hasattr(args, "__dict__"):
        payload["config"] = dict(vars(args))

    torch.save(payload, str(path))
    return str(path)

def _looks_like_state_dict(x) -> bool:
    if not isinstance(x, dict):
        return False
    n = 0
    for v in x.values():
        if torch.is_tensor(v):
            n += 1
            if n >= 3:
                return True
    return False


def _unwrap_model_state(ckpt: dict) -> dict:
    if _looks_like_state_dict(ckpt):
        return ckpt

    for key in ["model", "state_dict", "model_state_dict", "module", "net"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            if _looks_like_state_dict(state):
                return state

            for key2 in ["model", "state_dict", "model_state_dict", "module", "net"]:
                if key2 in state and isinstance(state[key2], dict) and _looks_like_state_dict(state[key2]):
                    return state[key2]

    raise RuntimeError(f"Could not find model state_dict in checkpoint. keys={list(ckpt.keys())[:30]}")


def _clean_state_keys(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue

        kk = k
        changed = True
        while changed:
            changed = False
            for pref in ["module.", "model.", "rr_trainer.model."]:
                if kk.startswith(pref):
                    kk = kk[len(pref):]
                    changed = True

        out[kk] = v

    return out


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def load_reranker_resume(
    *,
    ckpt_path: str | Path,
    rr_trainer: RerankerTrainer,
    device: torch.device,
    resume_optimizer: bool = True,
    reset_optimizer_lr: bool = True,
    resume_lr: float | None = None,
) -> Dict[str, object]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    logging.info("[resume] loading checkpoint: %s", str(ckpt_path))

    ckpt = safe_torch_load(str(ckpt_path), map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint must be dict, got {type(ckpt)}")

    state = _unwrap_model_state(ckpt)
    state = _clean_state_keys(state)

    missing, unexpected = rr_trainer.model.load_state_dict(state, strict=False)

    logging.info(
        "[resume] model loaded: missing=%d unexpected=%d",
        len(missing),
        len(unexpected),
    )

    if missing:
        logging.warning("[resume] missing sample: %s", missing[:20])
    if unexpected:
        logging.warning("[resume] unexpected sample: %s", unexpected[:20])

    opt = _get_optimizer(rr_trainer)
    loaded_optimizer = False

    if resume_optimizer and opt is not None and "optimizer" in ckpt:
        try:
            opt.load_state_dict(ckpt["optimizer"])
            loaded_optimizer = True
            logging.info("[resume] optimizer state loaded")
        except Exception as e:
            logging.warning("[resume] optimizer load failed, continuing model-only: %s", repr(e))

    scaler = _get_scaler(rr_trainer)
    loaded_scaler = False

    if scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
            loaded_scaler = True
            logging.info("[resume] GradScaler state loaded")
        except Exception as e:
            logging.warning("[resume] scaler load failed: %s", repr(e))

    if reset_optimizer_lr and opt is not None and resume_lr is not None:
        set_optimizer_lr(opt, float(resume_lr))
        logging.info("[resume] optimizer lr reset to %.8g", float(resume_lr))

    epoch = int(ckpt.get("epoch", 0))
    step = int(ckpt.get("global_step", ckpt.get("step", 0)))
    metrics = ckpt.get("metrics", {})

    logging.info(
        "[resume] epoch=%d step=%d loaded_optimizer=%s loaded_scaler=%s metrics=%s",
        epoch,
        step,
        loaded_optimizer,
        loaded_scaler,
        metrics,
    )

    return {
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "loaded_optimizer": loaded_optimizer,
        "loaded_scaler": loaded_scaler,
    }


# -----------------------------
# Helpers
# -----------------------------
def load_id_list(path: str) -> List[int]:
    if path.endswith(".json"):
        with open(path, "r") as f:
            xs = json.load(f)
        return [int(x) for x in xs]

    out = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(int(s))
    return out


def build_reranker_dataloaders(datasets, args, eval_go_ids: List[int], go_text_store) -> Tuple[DataLoader, DataLoader]:
    logger = logging.getLogger("build_dataloaders")

    train_ds = datasets["train"]
    val_ds = datasets["val"]

    sample0 = train_ds[0]
    print("train_ds[0] keys:", sample0.keys() if isinstance(sample0, dict) else type(sample0))
    print("train_ds[0].get('pos_go_global'):",
          sample0.get("pos_go_global", None) if isinstance(sample0, dict) else None)

    collate = ContrastiveEmbCollator(
        zs_mask_vec=torch.ones(len(eval_go_ids), dtype=torch.bool),
        go_text_store=go_text_store,
        bidirectional=False,
        neg_k=0,
        device=torch.device("cpu"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )
    logger.info("Dataloaders ready. batch_size=%d", args.batch_size)
    return train_loader, val_loader


@torch.no_grad()
def build_eval_G_once(
    eval_go_ids: List[int],
    go_text_store: GoTextStore,
    go_encoder: nn.Module,
    device: torch.device,
    chunk: int = 256,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Returns:
      G_once: [Geval, Dg] on CPU
      id2col: global_go_id -> column index
    """
    id2col = {int(g): i for i, g in enumerate(eval_go_ids)}
    toks = go_text_store.batch(eval_go_ids)
    input_ids = toks["input_ids"]
    attn = toks["attention_mask"]

    out_cpu = []
    go_encoder.eval()
    for s in range(0, input_ids.size(0), chunk):
        e = min(input_ids.size(0), s + chunk)
        embs = go_encoder(
            input_ids=input_ids[s:e].to(device, non_blocking=True),
            attention_mask=attn[s:e].to(device, non_blocking=True),
        )
        if isinstance(embs, tuple):
            embs = embs[0]
        if isinstance(embs, dict):
            if "pooler_output" in embs:
                embs = embs["pooler_output"]
            else:
                embs = embs["last_hidden_state"][:, 0]
        if embs.dim() == 3:
            embs = embs[:, 0]
        out_cpu.append(torch.nan_to_num(embs).float().cpu().contiguous())

    G_once = torch.cat(out_cpu, dim=0).contiguous()
    return G_once, id2col


def valid_mask_from_attn(attn: torch.Tensor) -> torch.Tensor:
    if attn.dtype != torch.bool:
        attn = attn != 0
    return attn


def make_labels_for_candidates(
    cand_ids: torch.Tensor,
    pos_go_global: List[torch.Tensor],
) -> torch.Tensor:
    """
    labels: [B,K] float32, 1 if cand_id in positives
    """
    B, K = cand_ids.shape
    labels = torch.zeros((B, K), dtype=torch.float32)
    cand_cpu = cand_ids.detach().cpu()

    for b in range(B):
        pos = pos_go_global[b]
        if pos is None or pos.numel() == 0:
            continue
        pos_set = set(int(x) for x in pos.detach().cpu().tolist())
        for j in range(K):
            if int(cand_cpu[b, j].item()) in pos_set:
                labels[b, j] = 1.0
    return labels


def make_dag_parent_mask_for_candidates(
    cand_ids: torch.Tensor,
    go_child_to_parents: Dict[int, List[int]],
) -> torch.Tensor:
    """
    Returns:
      dag_parent_mask: [B,K,K] bool

    Semantics:
      dag_parent_mask[b, i, j] = True
      if candidate j is a parent of candidate i
      i = child, j = parent
    """
    cand_cpu = cand_ids.detach().cpu()
    B, K = cand_cpu.shape
    out = torch.zeros((B, K, K), dtype=torch.bool)

    for b in range(B):
        row = [int(x) for x in cand_cpu[b].tolist()]
        row_set = set(row)

        for i, child_go in enumerate(row):
            parents = go_child_to_parents.get(child_go, [])
            if not parents:
                continue

            parent_set = row_set.intersection(int(p) for p in parents)
            if not parent_set:
                continue

            for j, parent_go in enumerate(row):
                if parent_go in parent_set:
                    out[b, i, j] = True

    return out


def tokenize_candidates_flat(
    go_text_store: GoTextStore,
    cand_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Returns:
      input_ids: [B*K,L]
      attention_mask: [B*K,L]
    """
    flat = cand_ids.reshape(-1).detach().cpu().tolist()
    toks = go_text_store.batch(flat)
    return toks

def make_rank_feature(B: int, K: int, device: torch.device) -> torch.Tensor:
    """Log-normalized 1-based rank feature, shape [B,K]."""
    r = torch.arange(1, K + 1, dtype=torch.float32, device=device)
    r = torch.log1p(r) / torch.log1p(torch.tensor(float(K), dtype=torch.float32, device=device))
    return r.unsqueeze(0).expand(B, K).contiguous()



# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def evaluate_reranker(
    rr_trainer: RerankerTrainer,
    retriever: ProteinGoAligner,
    val_loader: DataLoader,
    G_once_cpu: torch.Tensor,
    eval_go_ids: List[int],
    go_text_store: GoTextStore,
    go_child_to_parents: Dict[int, List[int]],
    device: torch.device,
    topk: int,
    max_batches: int = 0,
    candidate_lookup: CandidateDumpLookup | None = None,
) -> Dict[str, float]:
    """
    Safe evaluator for dump-backed or live-retriever candidate evaluation.

    Key fixes:
      1. Invalid candidates are replaced with a safe GO id before tokenization.
      2. Retriever score / rank features are sanitized and zeroed for invalid candidates.
      3. Logits are nan/inf sanitized and invalid logits are forced to -1e9.
      4. Metrics are computed only over valid candidates.
      5. Full-space denominator keeps unretrieved true labels as false negatives.
      6. Candidate-space denominator uses true labels that actually appear among valid candidates.
    """
    rr_trainer.model.eval()
    retriever.eval()

    eval_set = set(int(x) for x in eval_go_ids)

    # Candidate-space scored items only: (score, label)
    items_candidate: List[Tuple[float, int]] = []

    # Denominators
    n_true_full = 0
    n_true_candidate = 0

    # Protein-level hit metrics
    hit1 = 0
    hit5 = 0
    hit10 = 0
    n_prot = 0

    # Optional safe loss tracking, computed from the same forward pass
    bce_vals: List[float] = []
    dag_vals: List[float] = []
    loss_vals: List[float] = []

    # Optional loss params
    pos_weight_value = getattr(rr_trainer, "pos_weight", None)
    use_dag_loss = bool(getattr(rr_trainer, "use_dag_loss", False))
    lambda_dag = float(getattr(rr_trainer, "lambda_dag", 0.0))
    dag_margin = float(getattr(rr_trainer, "dag_margin", 0.0))

    for vb, vbatch in enumerate(val_loader):
        if max_batches and vb >= max_batches:
            break

        H2 = vbatch["prot_emb_pad"].to(device, non_blocking=True)
        vm = valid_mask_from_attn(vbatch["prot_attn_mask"].to(device, non_blocking=True))
        pos2 = vbatch["pos_go_global"]

        # ------------------------------------------------------------------
        # Candidate source
        # ------------------------------------------------------------------
        if candidate_lookup is not None:
            cand2, cand_scores2, lab2, true2, valid2 = candidate_lookup.get_batch(
                vbatch["protein_ids"]
            )
        else:
            cand2, cand_scores2 = retriever_topk_ids_chunked(
                retriever=retriever,
                H=H2,
                valid_mask=vm,
                G_once_cpu=G_once_cpu,
                eval_go_ids=eval_go_ids,
                topk=int(topk),
                device=device,
                chunk_k=2048,
            )
            lab2 = make_labels_for_candidates(cand2, pos2)
            true2 = None
            valid2 = torch.ones_like(lab2, dtype=torch.bool)

        B2, K2 = cand2.shape
        cand_valid = valid2.to(device, non_blocking=True).bool()

        # ------------------------------------------------------------------
        # IMPORTANT: invalid candidate IDs should not be tokenized as real GO.
        # Replace invalid ids with a safe valid GO id before tokenization.
        # The logits for invalid positions are later forced to -1e9.
        # ------------------------------------------------------------------
        cand2_safe = cand2.clone()
        if candidate_lookup is not None:
            safe_gid = int(candidate_lookup.eval_go_ids[0])
        else:
            safe_gid = int(eval_go_ids[0])

        valid2_cpu = valid2.detach().cpu().bool()
        cand2_safe[~valid2_cpu] = safe_gid

        toks2 = tokenize_candidates_flat(go_text_store, cand2_safe)
        go_input_ids = toks2["input_ids"].to(device, non_blocking=True)
        go_attention_mask = toks2["attention_mask"].to(device, non_blocking=True)

        # ------------------------------------------------------------------
        # DAG mask, then restrict it to valid candidates only.
        # dag_parent_mask[b, i, j] = candidate j is parent of candidate i.
        # ------------------------------------------------------------------
        dag_parent_mask2 = make_dag_parent_mask_for_candidates(
            cand_ids=cand2,
            go_child_to_parents=go_child_to_parents,
        ).to(device, non_blocking=True).bool()

        valid_pair = cand_valid.unsqueeze(2) & cand_valid.unsqueeze(1)
        dag_parent_mask2 = dag_parent_mask2 & valid_pair

        # ------------------------------------------------------------------
        # Retriever score / rank feature sanitization.
        # This is crucial for union/no-fill dumps where invalid/filler scores
        # may contain NaN, +/-inf, or large sentinel values.
        # ------------------------------------------------------------------
        retriever_score2 = cand_scores2.to(device, non_blocking=True)
        rank_feature2 = make_rank_feature(B2, K2, device)

        retriever_score2 = torch.nan_to_num(
            retriever_score2,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        retriever_score2 = torch.clamp(retriever_score2, min=-20.0, max=20.0)

        rank_feature2 = torch.nan_to_num(
            rank_feature2,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        retriever_score2 = torch.where(
            cand_valid,
            retriever_score2,
            torch.zeros_like(retriever_score2),
        )
        rank_feature2 = torch.where(
            cand_valid,
            rank_feature2,
            torch.zeros_like(rank_feature2),
        )

        # ------------------------------------------------------------------
        # Single forward pass.
        # Do not call rr_trainer.eval_step here, because that causes a second
        # forward and may report NaN loss from invalid candidates.
        # ------------------------------------------------------------------
        logits = rr_trainer.model(
            H=H2,
            K=K2,
            valid_mask=vm,
            go_input_ids=go_input_ids,
            go_attention_mask=go_attention_mask,
            retriever_score=retriever_score2,
            rank_feature=rank_feature2,
        )

        # ------------------------------------------------------------------
        # Logit safety.
        # Invalid logits cannot affect ranking or metrics.
        # ------------------------------------------------------------------
        if not torch.isfinite(logits).all():
            bad_total = int((~torch.isfinite(logits)).sum().detach().cpu().item())
            bad_valid = int(((~torch.isfinite(logits)) & cand_valid).sum().detach().cpu().item())
            logging.warning(
                "[eval-nan-guard] non-finite logits detected: total=%d valid=%d",
                bad_total,
                bad_valid,
            )

        logits = torch.nan_to_num(
            logits,
            nan=-1e9,
            posinf=1e9,
            neginf=-1e9,
        )

        logits = torch.where(
            cand_valid,
            logits,
            torch.full_like(logits, -1e9),
        )

        # ------------------------------------------------------------------
        # Optional safe BCE / DAG loss tracking from the same sanitized logits.
        # ------------------------------------------------------------------
        labels_dev = lab2.to(device, non_blocking=True).float()
        labels_dev = torch.where(cand_valid, labels_dev, torch.zeros_like(labels_dev))

        valid_logits = logits[cand_valid]
        valid_labels = labels_dev[cand_valid]

        if valid_logits.numel() > 0:
            if pos_weight_value is not None:
                pos_weight_t = torch.as_tensor(
                    float(pos_weight_value),
                    device=device,
                    dtype=valid_logits.dtype,
                )
                bce_eval = torch.nn.functional.binary_cross_entropy_with_logits(
                    valid_logits,
                    valid_labels,
                    pos_weight=pos_weight_t,
                    reduction="mean",
                )
            else:
                bce_eval = torch.nn.functional.binary_cross_entropy_with_logits(
                    valid_logits,
                    valid_labels,
                    reduction="mean",
                )
        else:
            bce_eval = logits.sum() * 0.0

        if use_dag_loss and dag_parent_mask2.any():
            child_scores = logits.unsqueeze(2).expand(B2, K2, K2)
            parent_scores = logits.unsqueeze(1).expand(B2, K2, K2)
            diffs = child_scores[dag_parent_mask2] - parent_scores[dag_parent_mask2] - dag_margin
            dag_eval = torch.nn.functional.softplus(diffs).mean()
            if not torch.isfinite(dag_eval):
                logging.warning("[eval-nan-guard] non-finite DAG loss, setting to zero")
                dag_eval = logits.sum() * 0.0
        else:
            dag_eval = logits.sum() * 0.0

        loss_eval = bce_eval + lambda_dag * dag_eval

        if torch.isfinite(bce_eval):
            bce_vals.append(float(bce_eval.detach().cpu().item()))
        if torch.isfinite(dag_eval):
            dag_vals.append(float(dag_eval.detach().cpu().item()))
        if torch.isfinite(loss_eval):
            loss_vals.append(float(loss_eval.detach().cpu().item()))

        # ------------------------------------------------------------------
        # Convert to numpy for metrics.
        # ------------------------------------------------------------------
        sc_np = logits.detach().float().cpu().numpy()
        lab_np = lab2.detach().cpu().numpy()
        cand_np = cand2.detach().cpu().numpy()
        valid_np = cand_valid.detach().cpu().numpy().astype(bool)

        for i in range(B2):
            # Full denominator
            if true2 is not None:
                pos_set = set(
                    int(x)
                    for x in true2[i].detach().cpu().tolist()
                    if int(x) >= 0 and int(x) in eval_set
                )
            else:
                row = pos2[i].detach().cpu().tolist() if pos2[i] is not None else []
                pos_set = set(
                    int(x)
                    for x in row
                    if int(x) >= 0 and int(x) in eval_set
                )

            n_true_full += len(pos_set)

            # Candidate denominator: true labels that are actually valid scored candidates.
            n_true_candidate += int(((lab_np[i] > 0) & valid_np[i]).sum())

            # Candidate-space scored items
            for j in range(K2):
                if not valid_np[i, j]:
                    continue

                score_ij = float(sc_np[i, j])
                if not np.isfinite(score_ij):
                    score_ij = -1e9

                items_candidate.append((score_ij, int(lab_np[i, j])))

            # Protein-level hits, valid candidates only
            order = np.argsort(-sc_np[i])
            order = [j for j in order if valid_np[i, j]]

            top1 = [int(cand_np[i, order[0]])] if len(order) > 0 else []
            top5 = [int(cand_np[i, j]) for j in order[: min(5, len(order))]]
            top10 = [int(cand_np[i, j]) for j in order[: min(10, len(order))]]

            hit1 += 1 if any(g in pos_set for g in top1) else 0
            hit5 += 1 if any(g in pos_set for g in top5) else 0
            hit10 += 1 if any(g in pos_set for g in top10) else 0
            n_prot += 1

    # ----------------------------------------------------------------------
    # Candidate-space reranker quality
    # ----------------------------------------------------------------------
    pr_topk = compute_global_fmax_aupr_from_items(items_candidate, n_true_candidate)

    # ----------------------------------------------------------------------
    # Full pipeline metric:
    # candidate-outside GT labels remain in denominator as false negatives.
    # ----------------------------------------------------------------------
    pr_full = compute_global_fmax_aupr_from_items(items_candidate, n_true_full)

    out = {
        "fmax_topk": pr_topk["fmax"],
        "aupr_topk": pr_topk["aupr"],
        "fmax_full": pr_full["fmax"],
        "aupr_full": pr_full["aupr"],
        "hits@1": float(hit1) / max(1, n_prot),
        "hits@5": float(hit5) / max(1, n_prot),
        "hits@10": float(hit10) / max(1, n_prot),
        "n_prot": float(n_prot),
        "n_true_candidate": float(n_true_candidate),
        "n_true_full": float(n_true_full),
        "n_pairs_scored": float(len(items_candidate)),
    }

    if bce_vals:
        out["bce"] = float(sum(bce_vals) / len(bce_vals))
    if dag_vals:
        out["dag_loss"] = float(sum(dag_vals) / len(dag_vals))
    if loss_vals:
        out["loss"] = float(sum(loss_vals) / len(loss_vals))

    out["retrieval_recall@K"] = float(n_true_candidate) / max(1.0, float(n_true_full))
    out["oracle_microF@K"] = float(
        (2.0 * n_true_candidate)
        / max(1e-12, 2.0 * n_true_candidate + (n_true_full - n_true_candidate))
    )

    return out

def inject_true_positives_into_candidates(
    cand_ids: torch.Tensor,                 # [B,K] CPU
    pos_go_global: List[torch.Tensor],      # list of [Pi]
    inject_n: int = 3,
) -> torch.Tensor:
    """
    Train-time only:
    Add up to inject_n missing true GO ids into candidate set for each protein.
    Keeps output shape [B,K] by replacing tail entries.
    """
    cand_cpu = cand_ids.detach().cpu().clone()
    B, K = cand_cpu.shape

    for b in range(B):
        pos = pos_go_global[b]
        if pos is None or pos.numel() == 0:
            continue

        orig_row = [int(x) for x in cand_cpu[b].tolist()]
        row_set = set(orig_row)

        pos_list = [int(x) for x in pos.detach().cpu().tolist()]
        pos_in = [g for g in pos_list if g in row_set]
        missing_pos = [g for g in pos_list if g not in row_set]

        if len(missing_pos) == 0:
            continue

        # dynamic injection rule
        if len(pos_in) == 0:
            inject = missing_pos[:inject_n]
        elif len(pos_in) == 1:
            inject = missing_pos[:max(0, inject_n - 1)]
        else:
            inject = []

        if len(inject) == 0:
            continue

        # Replace tail positions
        row = orig_row[:]
        replace_positions = list(range(K - len(inject), K))
        for j, g in zip(replace_positions, inject):
            row[j] = int(g)

        # Deduplicate while preserving order
        new_row = []
        seen = set()
        for g in row:
            if g not in seen:
                new_row.append(g)
                seen.add(g)

        # Refill from original row if dedup made it shorter
        for g in orig_row:
            if len(new_row) >= K:
                break
            if g not in seen:
                new_row.append(g)
                seen.add(g)

        # Final safety, should almost never trigger
        if len(new_row) < K:
            for g in pos_list:
                if len(new_row) >= K:
                    break
                if g not in seen:
                    new_row.append(g)
                    seen.add(g)

        cand_cpu[b] = torch.tensor(new_row[:K], dtype=torch.long)

    return cand_cpu

def feature_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """
    x: [B, K]
    Drops an entire scalar feature channel per protein row.
    Since features are normalized, zero is neutral.
    """
    if (not training) or p <= 0:
        return x

    B = x.size(0)
    keep = (torch.rand(B, 1, device=x.device) > p).to(x.dtype)
    return x * keep


# -----------------------------
# Config
# -----------------------------
def load_structured_cfg(path: str = RETRIEVER_YAML_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model = cfg.get("model", {})
    training = cfg.get("training", {})
    stores = cfg.get("stores", {})
    data = cfg.get("data", {})

    args = types.SimpleNamespace(
        # model
        model_kind=str(model.get("model_kind", "meanpool_mlp")),
        text_model_name=model.get(
            "text_model_name",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        ),
        protein_dim=int(model.get("protein_dim", 1280)),
        reranker_hidden_dim=int(model.get("reranker_hidden_dim", 512)),
        reranker_dropout=float(model.get("reranker_dropout", 0.1)),
        freeze_text_encoder=bool(model.get("freeze_text_encoder", True)),
        use_protein_ln=bool(model.get("use_protein_ln", True)),
        use_go_ln=bool(model.get("use_go_ln", True)),
        use_go_token_align_pooler=bool(model.get("use_go_token_align_pooler", True)),
        go_align_attn_dim=model.get("go_align_attn_dim", None),
        go_align_dropout=float(model.get("go_align_dropout", 0.1)),
        use_go_residual=bool(model.get("use_go_residual", True)),
        cross_dim=int(model.get("cross_dim", 256)),
        cross_heads=int(model.get("cross_heads", 4)),
        cross_dropout=float(model.get("cross_dropout", 0.1)),
        candidate_chunk_size=int(model.get("candidate_chunk_size", 16)),
        use_retriever_features=bool(model.get("use_retriever_features", True)),
        use_cls_residual=bool(model.get("use_cls_residual", True)),

        # training
        retriever_ckpt=str(training.get("retriever_ckpt", "")),
        device=str(training.get("device", "cuda")),
        fp16=bool(training.get("fp16", True)),
        topk=int(training.get("topk", 200)),
        batch_size=int(training.get("batch_size", 4)),
        lr=float(training.get("lr", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        epochs=int(training.get("epochs", 2)),
        log_every=int(training.get("log_every", 50)),
        eval_every=int(training.get("eval_every", 500)),
        save_metric=str(training.get("save_metric", "fmax")),
        out_dir=str(training.get("out_dir", "./reranker_out")),
        use_dag_loss=bool(training.get("use_dag_loss", False)),
        lambda_dag=float(training.get("lambda_dag", 0.1)),
        dag_margin=float(training.get("dag_margin", 0.0)),
        grad_clip_norm=training.get("grad_clip_norm", None),
        pos_weight=float(training.get("pos_weight", 20.0)),
        inject_true_positives=int(training.get("inject_true_positives", 0)),
        use_candidate_dump=bool(training.get("use_candidate_dump", False)),
        train_candidate_dump=str(training.get("train_candidate_dump", "")),
        val_candidate_dump=str(training.get("val_candidate_dump", "")),
        eval_every_steps=int(training.get("eval_every_steps", 0)),
        score_feature_dropout=float(training.get("score_feature_dropout", 0.0)),
        rank_feature_dropout = float(training.get("rank_feature_dropout", 0.0)),
        save_mid_checkpoints = bool(training.get("save_mid_checkpoints", True)),
        mid_checkpoint_metric = str(training.get("mid_checkpoint_metric", "fmax_full")),
        # resume
        resume=str(training.get("resume", "")),
        resume_optimizer=bool(training.get("resume_optimizer", True)),
        reset_optimizer_lr=bool(training.get("reset_optimizer_lr", True)),
        resume_lr=float(training.get("resume_lr", training.get("lr", 5e-5))),
        resume_start_next_epoch=bool(training.get("resume_start_next_epoch", True)),
        max_eval_batches=int(training.get("max_eval_batches", 0)),

        # stores
        train_ids_path=Path(stores.get("train_ids_path", "")),
        pid2pos=Path(stores.get("pid2pos_path", "")),
        val_ids_path=Path(stores.get("val_ids_path", "")),
        embed_dir_res=Path(stores.get("embed_dir_res", "")),
        embed_dir_fused=Path(stores.get("embed_dir_fused", "")),
        seq_len_lookup_dir=Path(stores.get("seq_len_lookup_dir", "")),
        go_text_folder=Path(stores.get("go_text_folder", "")),
        dag_parents_path=Path(stores.get("dag_parents_path", "")),
        go_cache_path=Path(stores.get("go_cache_path", "")),
        go_basic_json=Path(stores.get("go_basic_json", "")),
        zero_shot_path=Path(stores.get("zero_shot_path", "")),
        few_shot_path=Path(stores.get("few_shot_path", "")),
        go_path_seen=Path(stores.get("go_path_seen", "")),
        go_path_observed=Path(stores.get("go_path_observed", "")),

        # data
        max_len=int(data.get("protein_max_len", 1024)),
        overlap=int(data.get("overlap", 128)),
        fs_target_ratio=float(data.get("fs_target_ratio", 0.1)),
    )

    if args.go_align_attn_dim is not None:
        args.go_align_attn_dim = int(args.go_align_attn_dim)

    if args.grad_clip_norm is not None:
        args.grad_clip_norm = float(args.grad_clip_norm)

    return args


# -----------------------------
# Main loop
# -----------------------------
def main(phase_id: int = -2):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = load_structured_cfg()
    device = torch.device(args.device)

    go_id_to_text: Dict[int, Dict[int, str]] = {}
    go_id_to_text[phase_id] = load_go_texts_by_phase(args.go_text_folder, phase=phase_id)

    lora_params = LoRAParameters(adapter_name="go_encoder")
    go_enc_wrap = build_go_encoder_from_retriever_ckpt(
        args.retriever_ckpt,
        model_name=args.text_model_name,
        device=str(device),
        max_length=512,
        enable_lora=True,
        use_special_tokens=False,
        lora_parameters=lora_params,
        gradient_checkpointing=False,
    )
    go_encoder = go_enc_wrap.model.to(device)
    go_text_store = GoTextStore(
        full_id2text=go_id_to_text,
        tokenizer=go_enc_wrap.tokenizer,
        phase=phase_id,
    )

    retriever = ProteinGoAligner(
        d_h=args.protein_dim,
        d_g=None,
        d_z=768,
        go_encoder=go_encoder,
        normalize=True,
        protein_pool_type="attn"
    ).to(device)

    ckpt = safe_torch_load(args.retriever_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = state.get("model", state)
    state = {k: v for k, v in state.items() if not k.startswith("go_encoder.")}

    missing, unexpected = retriever.load_state_dict(state, strict=False)
    print(f"[main] retriever load: missing={len(missing)} unexpected={len(unexpected)}")

    if getattr(retriever, "go_encoder", None) is None:
        raise RuntimeError("retriever.go_encoder is None. Provide GO encoder or change build_eval_G_once logic.")

    res_store = build_stores(args)
    go_cache = build_go_cache(str(args.go_cache_path))

    if getattr(args, "dag_parents_path", ""):
        go_child_to_parents = load_child_to_parents_json(args.dag_parents_path)
        print(f"[main] loaded dag_parents: {len(go_child_to_parents)} children")
    else:
        go_child_to_parents = {}
        logging.warning("[main] dag_parents_path not set, DAG loss will stay inactive.")

    datasets = build_datasets(args, res_store, go_text_store)

    text_ids = set(int(x) for x in go_id_to_text[phase_id].keys())
    cache_ids = [int(x) for x in go_cache.row2id]
    eval_go_ids = [g for g in cache_ids if g in text_ids]

    eval_go_id_set = set(int(x) for x in eval_go_ids)

    sample_total_pos = 0
    sample_pos_in_eval = 0

    for ds_name in ["train", "val"]:
        ds = datasets[ds_name]
        upper = min(len(ds), 500)

        for i in range(upper):
            item = ds[i]
            pos = item.get("pos_go_global", None)

            if pos is None:
                continue

            if torch.is_tensor(pos):
                pos_list = [int(x) for x in pos.tolist()]
            else:
                pos_list = [int(x) for x in pos]

            sample_total_pos += len(pos_list)
            sample_pos_in_eval += sum(1 for x in pos_list if x in eval_go_id_set)

    logging.info(
        "[debug-coverage] total_pos=%d pos_in_eval=%d frac=%.4f",
        sample_total_pos,
        sample_pos_in_eval,
        sample_pos_in_eval / max(1, sample_total_pos),
    )

    print(f"[main] cache_ids={len(cache_ids)} text_ids={len(text_ids)} eval_go_ids(intersect)={len(eval_go_ids)}")
    assert len(eval_go_ids) > 0

    G_once_cpu, id2col = build_eval_G_once(
        eval_go_ids=eval_go_ids,
        go_text_store=go_text_store,
        go_encoder=go_encoder,
        device=device,
        chunk=256,
    )
    print(f"[main] G_once_cpu={tuple(G_once_cpu.shape)}")

    train_loader, val_loader = build_reranker_dataloaders(datasets=datasets, args=args, eval_go_ids=eval_go_ids, go_text_store=go_text_store)

    train_candidate_lookup = None
    val_candidate_lookup = None

    if getattr(args, "use_candidate_dump", False):
        train_candidate_lookup = CandidateDumpLookup(args.train_candidate_dump, topk=args.topk)
        val_candidate_lookup = CandidateDumpLookup(args.val_candidate_dump, topk=args.topk)

        logging.info("[candidate-dump] train=%s", args.train_candidate_dump)
        logging.info("[candidate-dump] val=%s", args.val_candidate_dump)

    if args.model_kind in {"hiercross", "residue_go_cross_attention", "p3a_hiercross"}:
        reranker_model = ResidueGoCrossAttentionReranker(
            text_model_name=args.text_model_name,
            d_h=args.protein_dim,
            hidden_dim=args.reranker_hidden_dim,
            freeze_text_encoder=args.freeze_text_encoder,
            dropout=args.reranker_dropout,
            use_protein_ln=args.use_protein_ln,
            use_go_ln=args.use_go_ln,
            cross_dim=args.cross_dim,
            cross_heads=args.cross_heads,
            cross_dropout=args.cross_dropout,
            candidate_chunk_size=args.candidate_chunk_size,
            use_retriever_features=args.use_retriever_features,
            use_cls_residual=args.use_cls_residual,
        ).to(device)
    else:
        reranker_model = RerankerMeanPoolConcatMLP(
            text_model_name=args.text_model_name,
            d_h=args.protein_dim,
            hidden_dim=args.reranker_hidden_dim,
            freeze_text_encoder=args.freeze_text_encoder,
            dropout=args.reranker_dropout,
            use_protein_ln=args.use_protein_ln,
            use_go_ln=args.use_go_ln,
            use_go_token_align_pooler=args.use_go_token_align_pooler,
            go_align_attn_dim=args.go_align_attn_dim,
            go_align_dropout=args.go_align_dropout,
            use_go_residual=args.use_go_residual,
        ).to(device)

    rr_trainer = RerankerTrainer(
        model=reranker_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=bool(args.fp16),
        device=str(device),
        use_dag_loss=args.use_dag_loss,
        lambda_dag=args.lambda_dag,
        dag_margin=args.dag_margin,
        grad_clip_norm=args.grad_clip_norm,
        pos_weight=args.pos_weight,
    )

    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    best = {
        "fmax_full": -float("inf"),
        "fmax_topk": -float("inf"),
        "aupr_full": -float("inf"),
        "aupr_topk": -float("inf"),
    }

    step = 0
    start_epoch = 0
    best_mid_metric = -float("inf")

    resume_path = str(getattr(args, "resume", "") or "")
    if resume_path:
        resume_info = load_reranker_resume(
            ckpt_path=resume_path,
            rr_trainer=rr_trainer,
            device=device,
            resume_optimizer=bool(getattr(args, "resume_optimizer", True)),
            reset_optimizer_lr=bool(getattr(args, "reset_optimizer_lr", True)),
            resume_lr=float(getattr(args, "resume_lr", getattr(args, "lr", 5e-5))),
        )

        loaded_epoch = int(resume_info.get("epoch", 0))
        loaded_step = int(resume_info.get("step", 0))
        loaded_metrics = resume_info.get("metrics", {}) or {}

        step = loaded_step

        if bool(getattr(args, "resume_start_next_epoch", True)):
            start_epoch = loaded_epoch + 1
        else:
            start_epoch = loaded_epoch

        metric_name = str(getattr(args, "mid_checkpoint_metric", "fmax_full"))
        if isinstance(loaded_metrics, dict) and metric_name in loaded_metrics:
            best_mid_metric = float(loaded_metrics[metric_name])

        if isinstance(loaded_metrics, dict):
            for k in best:
                if k in loaded_metrics:
                    best[k] = float(loaded_metrics[k])

        logging.info(
            "[resume] start_epoch=%d step=%d best_mid_metric=%.6f",
            start_epoch,
            step,
            best_mid_metric,
        )

    if start_epoch >= int(args.epochs):
        logging.warning(
            "[resume] start_epoch=%d >= epochs=%d. Increase epochs in YAML if you want to continue training.",
            start_epoch,
            int(args.epochs),
        )

    for epoch in range(start_epoch, int(args.epochs)):
        print(f"\n[main] epoch={epoch}")
        rr_trainer.model.train()

        for batch in train_loader:
            H = batch["prot_emb_pad"].to(device, non_blocking=True)
            valid_mask = valid_mask_from_attn(batch["prot_attn_mask"].to(device, non_blocking=True))
            pos_go_global = batch["pos_go_global"]

            if train_candidate_lookup is not None:
                cand_ids, cand_scores, labels, true_ids_batch, cand_valid_cpu = train_candidate_lookup.get_batch(
                    batch["protein_ids"]
                )
            else:
                cand_ids, cand_scores = retriever_topk_ids_chunked(
                    retriever=retriever,
                    H=H,
                    valid_mask=valid_mask,
                    G_once_cpu=G_once_cpu,
                    eval_go_ids=eval_go_ids,
                    topk=int(args.topk),
                    device=device,
                    chunk_k=2048,
                )
                labels = make_labels_for_candidates(cand_ids, pos_go_global)

            # Optional train-time positive injection. Default is 0 for strict direct retrieval.
            # If you enable this later, retriever scores for injected tail candidates are not exact.
            if train_candidate_lookup is None and int(args.inject_true_positives) > 0:
                cand_ids = inject_true_positives_into_candidates(
                    cand_ids=cand_ids,
                    pos_go_global=pos_go_global,
                    inject_n=int(args.inject_true_positives),
                )

            B, K = cand_ids.shape

            if train_candidate_lookup is not None:
                cand_valid = cand_valid_cpu.to(device, non_blocking=True).bool()
            else:
                cand_valid = torch.ones((B, K), dtype=torch.bool, device=device)

            # Replace invalid candidate IDs before tokenization.
            # The logits for these positions will be masked out anyway.
            cand_ids_safe = cand_ids.clone()
            if train_candidate_lookup is not None:
                safe_gid = int(train_candidate_lookup.eval_go_ids[0])
            else:
                safe_gid = int(eval_go_ids[0])

            cand_ids_safe[~cand_valid.detach().cpu().bool()] = safe_gid

            dag_parent_mask = make_dag_parent_mask_for_candidates(
                cand_ids=cand_ids,
                go_child_to_parents=go_child_to_parents,
            )

            toks = tokenize_candidates_flat(go_text_store, cand_ids_safe)
            go_input_ids = toks["input_ids"].to(device, non_blocking=True)
            go_attention_mask = toks["attention_mask"].to(device, non_blocking=True)

            if step == 0:
                logging.info("[debug-batch-keys] %s", list(batch.keys()))
                logging.info("[debug-pos-type] %s", type(batch.get("pos_go_global", None)))
                logging.info("[debug-pos-value] %s", batch.get("pos_go_global", None))

            if step < 5 or step % 500 == 0:
                pos_counts = []
                for b in range(len(pos_go_global)):
                    pos_set = set(int(x) for x in
                                  (pos_go_global[b].detach().cpu().tolist() if pos_go_global[b] is not None else []))
                    cand_set = set(int(x) for x in cand_ids[b].detach().cpu().tolist())
                    pos_counts.append(len(pos_set & cand_set))

                logging.info(
                    "[debug] step=%d label_sum=%.1f avg_pos_in_topk=%.3f max_pos_in_topk=%d cand_valid=%d",
                    step,
                    float(labels.sum().item()),
                    float(sum(pos_counts) / max(1, len(pos_counts))),
                    int(max(pos_counts) if pos_counts else 0),
                    int(cand_valid.sum().item()),
                )

                if len(pos_go_global) > 0:
                    ex_pos = pos_go_global[0].detach().cpu().tolist() if pos_go_global[0] is not None else []
                    ex_cand = cand_ids[0].detach().cpu().tolist()
                    logging.info("[debug] first pos sample=%s", ex_pos[:10])
                    logging.info("[debug] first cand sample=%s", ex_cand[:10])

            retriever_score = cand_scores.to(device, non_blocking=True)
            rank_feature = make_rank_feature(B, K, device)
            cand_valid = cand_valid.bool()

            retriever_score = torch.nan_to_num(
                retriever_score,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # Large sentinel scores from invalid/filler candidates can break fp16.
            retriever_score = torch.clamp(retriever_score, min=-20.0, max=20.0)

            rank_feature = torch.nan_to_num(
                rank_feature,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # Invalid candidates should be neutral model inputs.
            retriever_score = torch.where(cand_valid, retriever_score, torch.zeros_like(retriever_score))
            rank_feature = torch.where(cand_valid, rank_feature, torch.zeros_like(rank_feature))
            score_p = float(getattr(args, "score_feature_dropout", 0.0))
            rank_p = float(getattr(args, "rank_feature_dropout", 0.0))

            retriever_score_in = feature_dropout(
                retriever_score,
                p=score_p,
                training=rr_trainer.model.training,
            )

            rank_feature_in = feature_dropout(
                rank_feature,
                p=rank_p,
                training=rr_trainer.model.training,
            )

            rr_batch = dict(
                H=H,
                valid_mask=valid_mask,
                go_input_ids=go_input_ids,
                go_attention_mask=go_attention_mask,
                labels=labels.to(device, non_blocking=True),
                cand_valid=cand_valid,
                dag_parent_mask=dag_parent_mask.to(device, non_blocking=True),
                retriever_score=retriever_score_in,
                rank_feature=rank_feature_in,
                K=torch.tensor(K, dtype=torch.long, device=device),
            )

            stats = rr_trainer.train_step(rr_batch)
            step += 1

            if step % int(args.log_every) == 0:
                logging.info(
                    "[train] epoch=%d step=%d loss=%.4f bce=%.4f dag=%.4f pos_mean=%.4f neg_mean=%.4f",
                    epoch,
                    step,
                    stats.loss,
                    stats.bce_loss,
                    stats.dag_loss,
                    stats.pos_mean,
                    stats.neg_mean,
                )
            if int(getattr(args, "eval_every_steps", 0)) > 0:
                if step > 0 and step % int(args.eval_every_steps) == 0:
                    logging.info("[mid-epoch-eval] step=%d", step)

                    val_logs = evaluate_reranker(
                        rr_trainer=rr_trainer,
                        retriever=retriever,
                        val_loader=val_loader,
                        G_once_cpu=G_once_cpu,
                        eval_go_ids=eval_go_ids,
                        go_text_store=go_text_store,
                        go_child_to_parents=go_child_to_parents,
                        device=device,
                        topk=int(args.topk),
                        max_batches=int(getattr(args, "max_eval_batches", 0)),
                        candidate_lookup=val_candidate_lookup,
                    )

                    logging.info("[val@step%d] %s", step, val_logs)

                    metric_name = str(getattr(args, "mid_checkpoint_metric", "fmax_full"))
                    metric_value = float(val_logs.get(metric_name, -float("inf")))

                    if bool(getattr(args, "save_mid_checkpoints", True)) and metric_value > best_mid_metric:
                        best_mid_metric = metric_value

                        opt = _get_optimizer(rr_trainer)
                        scaler = _get_scaler(rr_trainer)

                        best_mid_path = save_checkpoint(
                            out_dir=out_dir,
                            step=step,
                            epoch=epoch,
                            optimizer=opt,
                            model=rr_trainer.model,
                            metrics=val_logs,
                            tag=f"mid_best_{metric_name}",
                            scaler=scaler,
                            args=args,
                        )

                        stable_path = save_checkpoint(
                            out_dir=out_dir,
                            step=step,
                            epoch=epoch,
                            optimizer=opt,
                            model=rr_trainer.model,
                            metrics=val_logs,
                            tag=f"best_{metric_name}",
                            scaler=scaler,
                            args=args,
                            path_override=Path(out_dir) / f"ckpt_best_{metric_name}.pt",
                        )

                        logging.info(
                            "[checkpoint] saved mid best %s=%.4f -> %s",
                            metric_name,
                            metric_value,
                            best_mid_path,
                        )
                        logging.info(
                            "[checkpoint] updated stable best -> %s",
                            stable_path,
                        )

        metrics = evaluate_reranker(
            rr_trainer=rr_trainer,
            retriever=retriever,
            val_loader=val_loader,
            G_once_cpu=G_once_cpu,
            eval_go_ids=eval_go_ids,
            go_text_store=go_text_store,
            go_child_to_parents=go_child_to_parents,
            device=device,
            topk=int(args.topk),
            max_batches=0,
            candidate_lookup=val_candidate_lookup
        )

        logging.info(
            "[val] epoch %d :: Fmax@200=%.4f AUPR@200=%.4f Fmax@Full=%.4f AUPR@Full=%.4f R@K=%.4f hits@1=%.4f hits@5=%.4f hits@10=%.4f",
            epoch,
            metrics["fmax_topk"],
            metrics["aupr_topk"],
            metrics["fmax_full"],
            metrics["aupr_full"],
            metrics["retrieval_recall@K"],
            metrics["hits@1"],
            metrics["hits@5"],
            metrics["hits@10"],
        )

        save_metric_name = str(args.save_metric).lower()
        if save_metric_name == "fmax_full":
            key = "fmax_full"
        elif save_metric_name == "aupr_full":
            key = "aupr_full"
        elif save_metric_name == "aupr_topk":
            key = "aupr_topk"
        else:
            key = "fmax_topk"

        if metrics[key] > best[key]:
            best[key] = metrics[key]
            opt = _get_optimizer(rr_trainer)
            scaler = _get_scaler(rr_trainer)
            best_path = save_checkpoint(
                out_dir=out_dir,
                step=step,
                epoch=epoch,
                model=rr_trainer.model,
                optimizer=opt,
                metrics=metrics,
                tag=f"best_{key}",
                scaler=scaler,
                args=args,
            )
            logging.info("[checkpoint] saved best_%s -> %s", key, best_path)

    print("[main] done")


if __name__ == "__main__":
    main()