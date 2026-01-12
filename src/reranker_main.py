from __future__ import annotations

import os
import math
import time
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import yaml, types
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import PosixPath

# Retriever
from src.models.alignment_model import ProteinGoAligner
from src.encoders.go_encoder import BioMedBERTEncoder, LoRAParameters
from src.configs.parameters import GO_SPECIAL_TOKENS

# Reranker
from src.training.reranker_trainer import RerankerTrainer
from src.models.reranker_model import RerankerMeanPoolConcatMLP  # Base Model
from src.datasets.go_text_store import GoTextStore
# dataset + collator
from src.training.collate import ContrastiveEmbCollator
from src.configs.paths import SRC_DIR, go_index_paths
from src.utils.helpers import load_go_texts_by_phase
from src.main import build_datasets, build_stores, build_go_cache

RETRIEVER_YAML_PATH = SRC_DIR / "reranker.yaml"

# Helpers for checkpoint loading

def safe_torch_load(path: str, map_location="cpu"):
    """
    PyTorch 2.6+ weights_only default değişti.
    Kendi checkpoint'in olduğu için weights_only=False ile yükle.
    PosixPath allowlist de ekliyoruz.
    """
    try:
        # bazı env'lerde gerekli
        torch.serialization.add_safe_globals([PosixPath])
    except Exception:
        pass

    return torch.load(path, map_location=map_location, weights_only=False)


def extract_sub_state(state: dict, prefix: str) -> dict:
    """
    state_dict içinden prefix ile başlayanları kırpıp döndürür.
    """
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

def build_go_encoder_from_retriever_ckpt(
    ckpt_path: str,
    *,
    model_name: str,
    device: str,
    max_length: int,
    enable_lora: bool,
    use_special_tokens: bool,
    lora_parameters: LoRAParameters | None,
    gradient_checkpointing: bool = False,  # eval için kapatmak daha temiz
):
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = state.get("model", state)

    go_sd = extract_sub_state(state, "go_encoder")
    go_sd = strip_prefix_from_state(go_sd, "model")

    if not go_sd:
        raise RuntimeError("Checkpoint içinde go_encoder.* bulunamadı.")

    # Encoder’ı checkpoint ile aynı şekilde kur
    enc = BioMedBERTEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        enable_lora=enable_lora,
        use_special_tokens=use_special_tokens,
        lora_parameters=lora_parameters,
        gradient_checkpointing=gradient_checkpointing,
        # pooling head vs kullanıyorsan burada da aynı parametreleri ver
        # use_attention_pool=..., attn_hidden=..., attn_dropout=...,
        # special_token_weights=... (eğitimde varsa)
    )
    enc.eval()

    # Not: BioMedBERTEncoder içindeki self.model, PEFT'li model
    missing, unexpected = enc.model.load_state_dict(go_sd, strict=False)

    print(f"[go_encoder load] missing={len(missing)} unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print("[go_encoder load] unexpected sample:", unexpected[:20])
    if len(missing) > 0:
        print("[go_encoder load] missing sample:", missing[:20])

    return enc

# -----------------------------
# OOM-safe retriever scoring: chunk over Geval
# -----------------------------
@torch.no_grad()
def retriever_topk_ids_chunked(
    retriever: ProteinGoAligner,
    H: torch.Tensor,                     # [B,T,Dh] on device
    valid_mask: torch.Tensor,            # [B,T] bool on device
    G_once_cpu: torch.Tensor,            # [Geval,Dg] CPU float32
    eval_go_ids: List[int],
    topk: int,
    device: torch.device,
    chunk_k: int = 2048,
) -> torch.Tensor:
    """
    Returns cand_ids: [B,K] global GO ids (CPU)
    Avoids building [B,Geval,Dg].
    """
    B = H.size(0)
    eval_ids_t = torch.as_tensor(eval_go_ids, dtype=torch.long)

    # Keep running topk
    K = int(min(topk, len(eval_go_ids)))
    best_scores = torch.full((B, K), -1e9, device=device)
    best_idx = torch.full((B, K), -1, dtype=torch.long, device=device)

    # iterate over chunks of GO embeddings
    Geval = G_once_cpu.size(0)
    retriever.eval()

    for s in range(0, Geval, chunk_k):
        e = min(Geval, s + chunk_k)
        G_chunk = G_once_cpu[s:e].to(device, non_blocking=True)  # [C,Dg]
        C = G_chunk.size(0)
        G_eval = G_chunk.unsqueeze(0).expand(B, C, G_chunk.size(1)).contiguous()

        sc = retriever(H=H, G=G_eval, mask=valid_mask, return_alpha=False)
        if isinstance(sc, tuple):
            sc = sc[0]
        # sc: [B,C]

        # merge topk between (best_scores,best_idx) and this chunk
        cur_scores, cur_rel = torch.topk(sc, k=min(K, C), dim=1)  # [B,k’]
        cur_idx = cur_rel + s

        merged_scores = torch.cat([best_scores, cur_scores], dim=1)  # [B, K+k’]
        merged_idx = torch.cat([best_idx, cur_idx], dim=1)

        new_scores, new_pos = torch.topk(merged_scores, k=K, dim=1)
        new_idx = merged_idx.gather(1, new_pos)

        best_scores, best_idx = new_scores, new_idx

    cand_ids = eval_ids_t.index_select(0, best_idx.reshape(-1).cpu()).view(B, K)
    return cand_ids


# -----------------------------
# Metrics: Fmax + AUPR from candidate scores
# -----------------------------
def _flat_truth_set(pos_go_global: List[torch.Tensor]) -> Tuple[set, int]:
    """
    Returns:
      truth_pairs: set of (i, go_id) for all positives
      n_true_total: total positives count
    """
    truth_pairs = set()
    n_true = 0
    for i, t in enumerate(pos_go_global):
        if t is None or t.numel() == 0:
            continue
        ids = [int(x) for x in t.detach().cpu().tolist()]
        for g in ids:
            truth_pairs.add((i, g))
        n_true += len(ids)
    return truth_pairs, n_true


def compute_fmax_aupr_from_scores(
    cand_ids: torch.Tensor,          # [B,K] CPU
    scores: torch.Tensor,            # [B,K] CPU float
    pos_go_global: List[torch.Tensor],
) -> Dict[str, float]:
    """
    Candidate-space metrics.
    - Fmax computed by sweeping thresholds over predicted scores.
    - AUPR computed over predicted candidate pairs only (approx), treating missing as 0.
      This is fine for model selection in reranker training.
    """
    B, K = cand_ids.shape
    cand_np = cand_ids.detach().cpu().numpy()
    sc_np = scores.detach().cpu().numpy()

    truth_pairs, n_true_total = _flat_truth_set(pos_go_global)
    if n_true_total == 0:
        return {"fmax": 0.0, "aupr": 0.0}

    # Build list of (score, is_true)
    items = []
    for i in range(B):
        for j in range(K):
            go = int(cand_np[i, j])
            s = float(sc_np[i, j])
            is_true = 1 if (i, go) in truth_pairs else 0
            items.append((s, is_true))

    # Sort by score desc
    items.sort(key=lambda x: x[0], reverse=True)

    # Sweep: at each cut, consider predicted = all items with score >= current
    tp = 0
    fp = 0
    best_f = 0.0

    # For AUPR: precision-recall curve points
    precisions = []
    recalls = []

    for idx, (s, is_true) in enumerate(items, start=1):
        if is_true:
            tp += 1
        else:
            fp += 1

        prec = tp / max(1, tp + fp)
        rec = tp / n_true_total

        precisions.append(prec)
        recalls.append(rec)

        if prec + rec > 0:
            f = 2 * prec * rec / (prec + rec)
            if f > best_f:
                best_f = f

    # Approx AUPR via trapezoidal integration over recall
    # Ensure monotonic recall, it is monotonic by construction
    aupr = 0.0
    prev_r = 0.0
    prev_p = 1.0
    for p, r in zip(precisions, recalls):
        dr = r - prev_r
        # use current precision as step function (common for PR integration)
        if dr > 0:
            aupr += p * dr
        prev_r = r
        prev_p = p

    return {"fmax": float(best_f), "aupr": float(aupr)}

def compute_global_fmax_aupr_from_items(items: List[Tuple[float, int]], n_true_total: int) -> Dict[str, float]:
    """
    items: list of (score, is_true) over ALL proteins and ALL candidate pairs
    n_true_total: total number of positives across ALL proteins (within the eval universe definition)
    """
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

        # Fmax
        if prec + rec > 0:
            f = 2 * prec * rec / (prec + rec)
            if f > best_f:
                best_f = f

        # AUPR (step integration in recall)
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
def save_checkpoint(
    out_dir: str,
    step: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: Dict[str, float],
    tag: str,
):
    os.makedirs(out_dir, exist_ok=True)
    path = Path(out_dir) / f"ckpt_{tag}_step{step}_epoch{epoch}.pt"
    payload = {
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, str(path))
    return str(path)


# -----------------------------
# Reranker eval: produce scores for candidates and compute fmax/aupr
# -----------------------------
@torch.no_grad()
def evaluate_reranker(
    rr_trainer: RerankerTrainer,
    retriever: ProteinGoAligner,
    val_loader: DataLoader,
    G_once_cpu: torch.Tensor,
    eval_go_ids: List[int],
    go_text_store: GoTextStore,
    device: torch.device,
    topk: int,
    max_batches: int = 0,   # 0 => all
) -> Dict[str, float]:
    rr_trainer.model.eval()
    retriever.eval()

    # Global accumulators
    items: List[Tuple[float, int]] = []
    n_true_total = 0

    # hits@k global (protein-level)
    hit1 = 0
    hit5 = 0
    hit10 = 0
    n_prot = 0

    for vb, vbatch in enumerate(val_loader):
        if max_batches and vb >= max_batches:
            break

        H2 = vbatch["prot_emb_pad"].to(device, non_blocking=True)
        vm = valid_mask_from_attn(vbatch["prot_attn_mask"].to(device, non_blocking=True))
        pos2 = vbatch["pos_go_global"]

        # total positives (global)
        n_true_total += count_true_total(pos2)

        cand2 = retriever_topk_ids_chunked(
            retriever=retriever,
            H=H2,
            valid_mask=vm,
            G_once_cpu=G_once_cpu,
            eval_go_ids=eval_go_ids,
            topk=int(topk),
            device=device,
            chunk_k=2048,
        )  # [B,K] CPU

        toks2 = tokenize_candidates_flat(go_text_store, cand2)
        go_input_ids = toks2["input_ids"].to(device, non_blocking=True)
        go_attention_mask = toks2["attention_mask"].to(device, non_blocking=True)

        B2, K2 = cand2.shape
        cand_valid = torch.ones((B2, K2), dtype=torch.bool, device=device)

        # labels for determining is_true, keep CPU for set logic
        lab2 = make_labels_for_candidates(cand2, pos2)  # CPU [B,K]

        rr_batch2 = dict(
            H=H2,
            valid_mask=vm,
            go_input_ids=go_input_ids,
            go_attention_mask=go_attention_mask,
            labels=lab2.to(device, non_blocking=True),
            cand_valid=cand_valid,
            K=torch.tensor(K2, dtype=torch.long, device=device)
        )

        out = rr_trainer.eval_step(rr_batch2)

        if "scores" in out:
            sc = out["scores"].detach().float().cpu()   # [B,K]
        elif "logits" in out:
            sc = out["logits"].detach().float().cpu()
        else:
            sc = rr_trainer.model(
                H=H2,
                valid_mask=vm,
                go_input_ids=go_input_ids,
                go_attention_mask=go_attention_mask,
                cand_valid=cand_valid,
            ).detach().float().cpu()

        # --- Global items accumulation for Fmax/AUPR ---
        sc_np = sc.numpy()
        lab_np = lab2.numpy()
        for i in range(B2):
            for j in range(K2):
                items.append((float(sc_np[i, j]), int(lab_np[i, j])))

        # --- hits@k (protein-level) ---
        cand_np = cand2.numpy()
        for i in range(B2):
            pos_set = set(int(x) for x in (pos2[i].detach().cpu().tolist() if pos2[i] is not None else []))
            order = np.argsort(-sc_np[i])  # desc

            top1 = [int(cand_np[i, order[0]])] if K2 > 0 else []
            top5 = [int(cand_np[i, j]) for j in order[: min(5, K2)]]
            top10 = [int(cand_np[i, j]) for j in order[: min(10, K2)]]

            hit1 += 1 if any(g in pos_set for g in top1) else 0
            hit5 += 1 if any(g in pos_set for g in top5) else 0
            hit10 += 1 if any(g in pos_set for g in top10) else 0
            n_prot += 1

    pr = compute_global_fmax_aupr_from_items(items, n_true_total)

    return {
        "fmax": pr["fmax"],
        "aupr": pr["aupr"],
        "hits@1": float(hit1) / max(1, n_prot),
        "hits@5": float(hit5) / max(1, n_prot),
        "hits@10": float(hit10) / max(1, n_prot),
        "n_prot": float(n_prot),
        "n_true_total": float(n_true_total),
        "n_pairs": float(len(items)),
    }
# -----------------------------
# Helpers
# -----------------------------
def load_id_list(path: str) -> List[int]:
    # supports json list or txt (one id per line)
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

def build_reranker_dataloaders(datasets, args, eval_go_ids: List[int]) -> Tuple[DataLoader, DataLoader]:
    logger = logging.getLogger("build_dataloaders")

    train_ds = datasets["train"]
    val_ds = datasets["val"]
    zs_mask_np = getattr(train_ds, "zs_mask", None)

    collate = ContrastiveEmbCollator(
        go_lookup=lambda ids: torch.zeros(len(ids), 1),  # unused here
        zs_mask_vec=torch.ones(len(eval_go_ids), dtype=torch.bool),
        bidirectional=False,
        go_text_store=None,
        faiss_miner=None,
        neg_k=0,
        device=torch.device("cpu"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
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
      G_once: [Geval, Dg] on CPU (float32)
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
        # unwrap
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
    # ensure bool [B,T]
    if attn.dtype != torch.bool:
        attn = attn != 0
    return attn


@torch.no_grad()
def retriever_topk_ids(
    retriever: ProteinGoAligner,
    H: torch.Tensor,                     # [B,T,Dh]
    valid_mask: torch.Tensor,            # [B,T] bool
    G_once_cpu: torch.Tensor,            # [Geval,Dg] CPU
    eval_go_ids: List[int],
    topk: int,
    device: torch.device,
    cand_chunk_k: int = 64,
) -> torch.Tensor:
    """
    Returns:
      cand_ids: [B, K] global GO ids (long) on CPU
    """
    B = H.size(0)
    G_once = G_once_cpu.to(device, non_blocking=True)           # [Geval,Dg]
    Geval, Dg = G_once.size()
    G_eval = G_once.unsqueeze(0).expand(B, Geval, Dg).contiguous()

    scores = retriever(H=H, G=G_eval, mask=valid_mask, return_alpha=False)  # [B,Geval]
    # note: if you have logit_scale, apply here. keep consistent with retrieval training if needed.

    K = min(int(topk), int(scores.size(1)))
    idx = torch.topk(scores, k=K, dim=1).indices                # [B,K]
    eval_ids_t = torch.as_tensor(eval_go_ids, dtype=torch.long, device=idx.device)
    cand_ids = eval_ids_t.index_select(0, idx.reshape(-1)).view(B, K).detach().cpu()
    return cand_ids


def make_labels_for_candidates(
    cand_ids: torch.Tensor,             # [B,K] global ids (CPU or GPU ok)
    pos_go_global: List[torch.Tensor],  # list of [Pi] global ids
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


def tokenize_candidates_flat(
    go_text_store: GoTextStore,
    cand_ids: torch.Tensor,   # [B,K] CPU preferred
) -> Dict[str, torch.Tensor]:
    """
    Returns tokens for [B*K] candidates:
      input_ids: [B*K,L]
      attention_mask: [B*K,L]
    """
    flat = cand_ids.reshape(-1).detach().cpu().tolist()
    toks = go_text_store.batch(flat)
    return toks
def load_structured_cfg(path: str = RETRIEVER_YAML_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model = cfg.get("model", {})
    training = cfg.get("training", {})
    stores = cfg.get("stores", {})
    data = cfg.get("data", {})

    args = types.SimpleNamespace(
        #Model
        text_model_name=model.get("text_model_name", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
        protein_dim=int(model.get("protein_dim", 1280)),
        #Training
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
        #stores
        out_dir=str(training.get("out_dir", "./reranker_out")),
        train_ids_path=str(stores.get("train_ids_path", "")),
        pid2pos_path=str(stores.get("pid2pos_path", "")),
        val_ids_path=str(stores.get("val_ids_path", "")),
        embed_dir_res_path=str(stores.get("embed_dir_res_path", "")),
        embed_dir_fused_path=str(stores.get("embed_dir_fused_path", "")),
        seq_len_lookup_dir_path=str(stores.get("seq_len_lookup_dir_path", "")),
        protein_manifest_file_path=str(stores.get("protein_manifest_file_path", "")),
        go_text_folder=str(stores.get("go_text_folder", "")),
        # data
        max_len=int(data.get("protein_max_len", 1024)),
        overlap=int(data.get("overlap", 128)),
    )
    return args

# -----------------------------
# Main loop
# -----------------------------
def main(phase_id = -2):
    args = load_structured_cfg()
    device = torch.device(args.device)

    # 2) Build GoTextStore + GO encoder (same BiomedBERT backbone used in retriever, or standalone)
    go_id_to_text: Dict[int, Dict[int, str]] = {}
    go_id_to_text[phase_id] = load_go_texts_by_phase(args.go_text_folder, phase=phase_id)

    lora_params = LoRAParameters(adapter_name="go_encoder")
    # go_encoder'ı ayrı yükle
    go_enc_wrap = build_go_encoder_from_retriever_ckpt(
        args.retriever_ckpt,
        model_name=args.text_model_name,
        device=str(device),
        max_length=512,
        enable_lora=True,
        use_special_tokens=True,  # retriever train’de eklediysen True
        lora_parameters=lora_params,  # yukarıda oluşturduğun params
        gradient_checkpointing=False,
    )
    go_encoder = go_enc_wrap.model  # build_eval_G_once bunu çağırıyor
    go_encoder = go_encoder.to(device)
    go_text_store = GoTextStore(full_id2text=go_id_to_text, tokenizer=go_enc_wrap.tokenizer, phase=phase_id)

    # 2) Retriever'ı go_encoder ile oluştur
    retriever = ProteinGoAligner(
        d_h=args.protein_dim,
        d_g=None,  # senin modelin içinde go_ln vs varsa genelde d_g otomatik, değilse doğru d_g ver
        d_z=768,
        go_encoder=go_encoder,  # kritik
        normalize=True,
        mean_pool=False,
    ).to(device)

    ckpt = safe_torch_load(args.retriever_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = state.get("model", state)

    state = {k: v for k, v in state.items() if not k.startswith("go_encoder.")}

    # retriever yükle (topk üretmek için)
    missing, unexpected = retriever.load_state_dict(state, strict=False)
    print(f"[main] retriever load: missing={len(missing)} unexpected={len(unexpected)}")


    # If your retriever has go_encoder inside, use it to build eval_G_once.
    # Otherwise, you must load a GO encoder separately.
    if getattr(retriever, "go_encoder", None) is None:
        raise RuntimeError("retriever.go_encoder is None. Provide GO encoder or change build_eval_G_once logic.")

    res_store, fused_store = build_stores(args)
    go_cache = build_go_cache(go_index_paths(phase_id)["TEXT_EMB"])
    datasets = build_datasets(args, res_store, fused_store, go_cache)

    text_ids = set(int(x) for x in go_id_to_text[phase_id].keys())
    cache_ids = [int(x) for x in go_cache.row2id]
    eval_go_ids = [g for g in cache_ids if g in text_ids]

    print(f"[main] cache_ids={len(cache_ids)} text_ids={len(text_ids)} eval_go_ids(intersect)={len(eval_go_ids)}")
    assert len(eval_go_ids) > 0

    # 5) Build datasets/loaders
    # GO universe = retriever training'de kullanılan GO universe
    # 4) Precompute global GO embedding matrix once (CPU)
    G_once_cpu, id2col = build_eval_G_once(
        eval_go_ids=eval_go_ids,
        go_text_store=go_text_store,
        go_encoder=go_encoder,
        device=device,
        chunk=256,
    )
    print(f"[main] G_once_cpu={tuple(G_once_cpu.shape)}")

    train_loader, val_loader = build_reranker_dataloaders(datasets, args, eval_go_ids)

    # 6) Build reranker model + trainer
    reranker_model = RerankerMeanPoolConcatMLP(text_model_name=args.text_model_name, d_h=args.protein_dim).to(device)
     # fill args in your project
    rr_trainer = RerankerTrainer(
        model=reranker_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=bool(args.fp16),
        device=str(device),
    )

    # init logging
    logging.basicConfig(level=logging.INFO)
    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    best = {"fmax": -1.0, "aupr": -1.0}
    best_path = None

    # 7) Train
    step = 0
    for epoch in range(int(args.epochs)):
        print(f"\n[main] epoch={epoch}")
        for batch in train_loader:
            # batch must include:
            # prot_emb_pad: [B,T,Dh], prot_attn_mask: [B,T], pos_go_global: List[Tensor]
            H = batch["prot_emb_pad"].to(device, non_blocking=True)
            valid_mask = valid_mask_from_attn(batch["prot_attn_mask"].to(device, non_blocking=True))
            pos_go_global = batch["pos_go_global"]

            cand_ids = retriever_topk_ids_chunked(
                retriever=retriever,
                H=H,
                valid_mask=valid_mask,
                G_once_cpu=G_once_cpu,
                eval_go_ids=eval_go_ids,
                topk=int(args.topk),
                device=device,
                chunk_k=2048,
            )  # [B,K] CPU

            labels = make_labels_for_candidates(cand_ids, pos_go_global)  # [B,K] CPU float

            toks = tokenize_candidates_flat(go_text_store, cand_ids)
            go_input_ids = toks["input_ids"].to(device, non_blocking=True)
            go_attention_mask = toks["attention_mask"].to(device, non_blocking=True)

            B, K = cand_ids.shape
            cand_valid = torch.ones((B, K), dtype=torch.bool).to(device, non_blocking=True)

            rr_batch = dict(
                H=H,
                valid_mask=valid_mask,
                go_input_ids=go_input_ids,
                go_attention_mask=go_attention_mask,
                labels=labels.to(device, non_blocking=True),
                cand_valid=cand_valid,
                K=torch.tensor(K, dtype=torch.long, device=device),
            )

            stats = rr_trainer.train_step(rr_batch)
            if step % int(args.log_every) == 0:
                print(f"[train] step={step} loss={stats.loss:.4f} pos_mean={stats.pos_mean:.4f} neg_mean={stats.neg_mean:.4f}")

            if step > 0 and (step % int(args.eval_every) == 0):
                metrics = evaluate_reranker(
                    rr_trainer=rr_trainer,
                    retriever=retriever,
                    val_loader=val_loader,
                    G_once_cpu=G_once_cpu,
                    eval_go_ids=eval_go_ids,
                    go_text_store=go_text_store,
                    device=device,
                    topk=int(args.topk),
                    max_batches=0,
                )
                print(
                    f"[val] step={step} "
                    f"fmax={metrics['fmax']:.4f} aupr={metrics['aupr']:.4f} "
                    f"hits@1={metrics['hits@1']:.4f} hits@5={metrics['hits@5']:.4f} hits@10={metrics['hits@10']:.4f}"
                )

                key = "fmax" if str(args.save_metric).lower() == "fmax" else "aupr"
                if metrics[key] > best[key]:
                    best[key] = metrics[key]
                    # you need access to optimizer inside trainer, assuming rr_trainer.optimizer exists
                    opt = getattr(rr_trainer, "opt", None)  or getattr(rr_trainer, "optimizer", None)
                    if opt is None:
                        raise RuntimeError("rr_trainer.optimizer not found. Expose optimizer for checkpointing.")
                    best_path = save_checkpoint(
                        out_dir=out_dir,
                        step=step,
                        epoch=epoch,
                        model=rr_trainer.model,
                        optimizer=opt,
                        metrics=metrics,
                        tag=f"best_{key}",
                    )
                    print(f"[checkpoint] saved best_{key} -> {best_path}")

            step += 1

    print("[main] done")


if __name__ == "__main__":
    main()