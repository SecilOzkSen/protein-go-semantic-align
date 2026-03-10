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
from src.models.reranker_model import RerankerMeanPoolConcatMLP
from src.datasets.go_text_store import GoTextStore
# dataset + collator
from src.training.collate import ContrastiveEmbCollator
from src.configs.paths import SRC_DIR, go_index_paths
from src.utils.helpers import load_go_texts_by_phase
from src.main import build_datasets, build_stores, build_go_cache

RETRIEVER_YAML_PATH = SRC_DIR / "reranker.yaml"

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
) -> torch.Tensor:
    """
    Returns:
      cand_ids: [B,K] global GO ids on CPU
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
    return cand_ids


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


def build_reranker_dataloaders(datasets, args, eval_go_ids: List[int]) -> Tuple[DataLoader, DataLoader]:
    logger = logging.getLogger("build_dataloaders")

    train_ds = datasets["train"]
    val_ds = datasets["val"]

    collate = ContrastiveEmbCollator(
        go_lookup=lambda ids: torch.zeros(len(ids), 1),
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
) -> Dict[str, float]:
    rr_trainer.model.eval()
    retriever.eval()

    # Only candidate-space scored items are stored
    items_candidate: List[Tuple[float, int]] = []

    # Denominators
    n_true_full = 0
    n_true_candidate = 0

    # Protein-level hits
    hit1 = 0
    hit5 = 0
    hit10 = 0
    n_prot = 0

    # Optional loss tracking from eval_step
    bce_vals: List[float] = []
    dag_vals: List[float] = []
    loss_vals: List[float] = []

    for vb, vbatch in enumerate(val_loader):
        if max_batches and vb >= max_batches:
            break

        H2 = vbatch["prot_emb_pad"].to(device, non_blocking=True)
        vm = valid_mask_from_attn(vbatch["prot_attn_mask"].to(device, non_blocking=True))
        pos2 = vbatch["pos_go_global"]

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

        lab2 = make_labels_for_candidates(cand2, pos2)  # CPU [B,K]
        dag_parent_mask2 = make_dag_parent_mask_for_candidates(
            cand_ids=cand2,
            go_child_to_parents=go_child_to_parents,
        )  # [B,K,K] CPU bool

        rr_batch2 = dict(
            H=H2,
            valid_mask=vm,
            go_input_ids=go_input_ids,
            go_attention_mask=go_attention_mask,
            labels=lab2.to(device, non_blocking=True),
            cand_valid=cand_valid,
            dag_parent_mask=dag_parent_mask2.to(device, non_blocking=True),
            K=torch.tensor(K2, dtype=torch.long, device=device),
        )

        eval_out = rr_trainer.eval_step(rr_batch2)
        if "bce" in eval_out:
            bce_vals.append(float(eval_out["bce"]))
        if "dag_loss" in eval_out:
            dag_vals.append(float(eval_out["dag_loss"]))
        if "loss" in eval_out:
            loss_vals.append(float(eval_out["loss"]))

        logits = rr_trainer.model(
            H=H2,
            K=K2,
            valid_mask=vm,
            go_input_ids=go_input_ids,
            go_attention_mask=go_attention_mask,
        ).detach().float().cpu()  # [B,K]

        sc_np = logits.numpy()
        lab_np = lab2.numpy()
        cand_np = cand2.numpy()

        for i in range(B2):
            pos_set = set(int(x) for x in (
                pos2[i].detach().cpu().tolist() if pos2[i] is not None else []
            ))
            cand_row = [int(x) for x in cand_np[i].tolist()]
            cand_set = set(cand_row)

            # Full denominator: all true labels
            n_true_full += len(pos_set)

            # TopK denominator: only true labels that survived retrieval
            n_true_candidate += len(pos_set & cand_set)

            # Candidate-space scored items only
            for j in range(K2):
                items_candidate.append((float(sc_np[i, j]), int(lab_np[i, j])))

            # Protein-level hits
            order = np.argsort(-sc_np[i])

            top1 = [int(cand_np[i, order[0]])] if K2 > 0 else []
            top5 = [int(cand_np[i, j]) for j in order[: min(5, K2)]]
            top10 = [int(cand_np[i, j]) for j in order[: min(10, K2)]]

            hit1 += 1 if any(g in pos_set for g in top1) else 0
            hit5 += 1 if any(g in pos_set for g in top5) else 0
            hit10 += 1 if any(g in pos_set for g in top10) else 0
            n_prot += 1

    # Candidate-space reranker quality
    pr_topk = compute_global_fmax_aupr_from_items(items_candidate, n_true_candidate)

    # Full pipeline metric, candidate dışı GT'ler denominator'da kalır
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

    # Useful retrieval ceiling diagnostic
    out["retrieval_recall@K"] = float(n_true_candidate) / max(1.0, float(n_true_full))

    return out


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

        # stores
        train_ids_path=str(stores.get("train_ids_path", "")),
        pid2pos=str(stores.get("pid2pos_path", "")),
        val_ids_path=str(stores.get("val_ids_path", "")),
        embed_dir_res=str(stores.get("embed_dir_res", "")),
        embed_dir_fused=str(stores.get("embed_dir_fused", "")),
        seq_len_lookup_dir=str(stores.get("seq_len_lookup_dir", "")),
        go_text_folder=str(stores.get("go_text_folder", "")),
        dag_parents_path=str(stores.get("dag_parents_path", "")),

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

    train_loader, val_loader = build_reranker_dataloaders(datasets, args, eval_go_ids)

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
    )

    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    best = {"fmax": -1.0, "aupr": -1.0}
    step = 0

    for epoch in range(int(args.epochs)):
        print(f"\n[main] epoch={epoch}")
        rr_trainer.model.train()

        for batch in train_loader:
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
            )

            labels = make_labels_for_candidates(cand_ids, pos_go_global)
            dag_parent_mask = make_dag_parent_mask_for_candidates(
                cand_ids=cand_ids,
                go_child_to_parents=go_child_to_parents,
            )

            toks = tokenize_candidates_flat(go_text_store, cand_ids)
            go_input_ids = toks["input_ids"].to(device, non_blocking=True)
            go_attention_mask = toks["attention_mask"].to(device, non_blocking=True)

            B, K = cand_ids.shape
            cand_valid = torch.ones((B, K), dtype=torch.bool, device=device)

            rr_batch = dict(
                H=H,
                valid_mask=valid_mask,
                go_input_ids=go_input_ids,
                go_attention_mask=go_attention_mask,
                labels=labels.to(device, non_blocking=True),
                cand_valid=cand_valid,
                dag_parent_mask=dag_parent_mask.to(device, non_blocking=True),
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
            opt = getattr(rr_trainer, "opt", None) or getattr(rr_trainer, "optimizer", None)
            best_path = save_checkpoint(
                out_dir=out_dir,
                step=step,
                epoch=epoch,
                model=rr_trainer.model,
                optimizer=opt,
                metrics=metrics,
                tag=f"best_{key}",
            )
            logging.info("[checkpoint] saved best_%s -> %s", key, best_path)

    print("[main] done")


if __name__ == "__main__":
    main()