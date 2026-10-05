from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import gc
import json
import math
import logging
import pickle
from pathlib import Path, PosixPath
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.utils.helpers import go_str_to_int_any, load_go_texts_by_phase
from src.configs.paths import SRC_DIR
from src.main import build_datasets, build_stores, build_go_cache
from src.training.collate import ContrastiveEmbCollator
from src.models.alignment_model import ProteinGoAligner
from src.models.reranker_model import RerankerMeanPoolConcatMLP
from src.encoders.go_encoder import BioMedBERTEncoder, LoRAParameters
from src.datasets.go_text_store import GoTextStore


# =========================================================
# SAFE DEFAULTS
# =========================================================
DEFAULT_YAML_PATH = SRC_DIR / "reranker.yaml"
PHASE_ID = -2

TOPK = 200
BATCH_SIZE = 1
GO_EMB_CHUNK = 16
SCORE_CHUNK = 64

OUTPUT_DIR = Path("outputs/retriever_vs_reranker")

# override istersen path ver
RERANKER_CKPT_OVERRIDE: Optional[str] = None

# hızlı test için sayı ver, full val için None
VAL_SUBSET_SIZE: Optional[int] = 100

FORCE_GC = True


# =========================================================
# Utils
# =========================================================
def safe_path(x: Any) -> Optional[Path]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    return Path(s)


def maybe_empty_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def maybe_gc():
    if FORCE_GC:
        gc.collect()


def safe_torch_load(path: str | Path, map_location="cpu"):
    try:
        torch.serialization.add_safe_globals([PosixPath])
    except Exception:
        pass
    return torch.load(str(path), map_location=map_location, weights_only=False)


def extract_sub_state(state: dict, prefix: str) -> dict:
    out = {}
    p = prefix if prefix.endswith(".") else prefix + "."
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


def load_id_list(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(f"ID list path not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            xs = json.load(f)
        return [go_str_to_int_any(x) for x in xs]

    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            xs = pickle.load(f)
        return [go_str_to_int_any(x) for x in xs]

    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(go_str_to_int_any(s))
    return out


def valid_mask_from_attn(attn: torch.Tensor) -> torch.Tensor:
    if attn.dtype != torch.bool:
        attn = attn != 0
    return attn


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# =========================================================
# Metrics
# =========================================================
def compute_global_fmax_aupr_from_items(items: List[Tuple[float, int]], n_true_total: int) -> Dict[str, float]:
    if n_true_total <= 0 or len(items) == 0:
        return {"fmax": 0.0, "aupr": 0.0}

    items = sorted(items, key=lambda x: x[0], reverse=True)

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


def reciprocal_rank(labels_sorted: List[int]) -> float:
    for idx, y in enumerate(labels_sorted, start=1):
        if y == 1:
            return 1.0 / idx
    return 0.0


def dcg_at_k(labels_sorted: List[int], k: int) -> float:
    out = 0.0
    for i, rel in enumerate(labels_sorted[:k], start=1):
        if rel > 0:
            out += float(rel) / math.log2(i + 1)
    return out


def ndcg_at_k(labels_sorted: List[int], k: int) -> float:
    dcg = dcg_at_k(labels_sorted, k)
    ideal = sorted(labels_sorted, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


# =========================================================
# Config
# =========================================================
def load_structured_cfg(path: str | Path = DEFAULT_YAML_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model = cfg.get("model", {})
    training = cfg.get("training", {})
    stores = cfg.get("stores", {})

    args = SimpleNamespace(
        text_model_name=model.get(
            "text_model_name",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        ),
        protein_dim=int(model.get("protein_dim", 1280)),
        protein_pool_type=str(model.get("protein_pool_type", "go_align")),
        device=str(training.get("device", "cuda")),
        fp16=bool(training.get("fp16", True)),
        retriever_ckpt=str(training.get("retriever_ckpt", "")),
        batch_size=int(training.get("batch_size", BATCH_SIZE)),
        topk=int(training.get("topk", TOPK)),

        go_cache_path=safe_path(stores.get("go_cache_path")),
        go_text_folder=safe_path(stores.get("go_text_folder")),
        go_path_observed=safe_path(stores.get("go_path_observed")),

        reranker_hidden_dim=int(model.get("reranker_hidden_dim", 512)),
        reranker_dropout=float(model.get("reranker_dropout", 0.1)),
        freeze_text_encoder=bool(model.get("freeze_text_encoder", True)),
        use_protein_ln=bool(model.get("use_protein_ln", True)),
        use_go_ln=bool(model.get("use_go_ln", True)),
        use_go_token_align_pooler=bool(model.get("use_go_token_align_pooler", True)),
        go_align_attn_dim=model.get("go_align_attn_dim", None),
        go_align_dropout=float(model.get("go_align_dropout", 0.1)),
        use_go_residual=bool(model.get("use_go_residual", True)),
    )
    return args


# =========================================================
# GO encoder restore
# =========================================================
def build_go_encoder_from_retriever_ckpt(
    ckpt_path: str | Path,
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
    logging.info("[go_encoder load] missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        logging.info("[go_encoder load] missing sample: %s", missing[:10])
    if unexpected:
        logging.info("[go_encoder load] unexpected sample: %s", unexpected[:10])

    return enc


# =========================================================
# GO embeddings, CPU-safe
# =========================================================
def build_eval_G_once(
    eval_go_ids: List[int],
    go_text_store: GoTextStore,
    go_encoder: torch.nn.Module,
    chunk: int = GO_EMB_CHUNK,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    id2col = {int(g): i for i, g in enumerate(eval_go_ids)}
    out_cpu = []

    cpu_device = torch.device("cpu")
    go_encoder = go_encoder.to(cpu_device)
    go_encoder.eval()

    for s in range(0, len(eval_go_ids), chunk):
        e = min(len(eval_go_ids), s + chunk)
        go_ids_chunk = eval_go_ids[s:e]

        toks = go_text_store.batch(go_ids_chunk)
        input_ids = toks["input_ids"].to(cpu_device)
        attn = toks["attention_mask"].to(cpu_device)

        with torch.no_grad():
            embs = go_encoder(
                input_ids=input_ids,
                attention_mask=attn,
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

        del toks, input_ids, attn, embs
        maybe_gc()

        if ((s // chunk) + 1) % 20 == 0:
            logging.info("GO embedding chunks processed: %d / %d", e, len(eval_go_ids))

    G_once = torch.cat(out_cpu, dim=0).contiguous()
    del out_cpu
    maybe_gc()

    return G_once, id2col


# =========================================================
# DataLoader
# =========================================================
def build_val_loader(val_ds, go_text_store, eval_go_ids: List[int], batch_size: int) -> DataLoader:
    collate = ContrastiveEmbCollator(
        zs_mask_vec=torch.ones(len(eval_go_ids), dtype=torch.bool),
        go_text_store=go_text_store,
        bidirectional=False,
        neg_k=0,
        device=torch.device("cpu"),
    )
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=False,
    )
    return loader


# =========================================================
# Retriever restore
# =========================================================
def load_retriever_model(
    args,
    device: torch.device,
    d_g: int = 768,
):
    retriever = ProteinGoAligner(
        d_h=args.protein_dim,
        d_g=d_g,
        d_z=768,
        go_encoder=None,
        normalize=True,
        protein_pool_type=args.protein_pool_type,
    ).to(device)

    ckpt = safe_torch_load(args.retriever_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = state.get("model", state)
    state = {k: v for k, v in state.items() if not k.startswith("go_encoder.")}

    missing, unexpected = retriever.load_state_dict(state, strict=False)
    logging.info("[retriever load] missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        logging.info("[retriever load] missing sample: %s", missing[:10])
    if unexpected:
        logging.info("[retriever load] unexpected sample: %s", unexpected[:10])

    retriever.eval()
    return retriever


# =========================================================
# Reranker restore
# =========================================================
def extract_model_state_from_ckpt(ckpt: dict) -> dict:
    state = ckpt.get("model", ckpt)
    state = state.get("model", state)
    return state


def load_reranker_model(
    args,
    device: torch.device,
    reranker_ckpt_path: str | Path,
):
    model = RerankerMeanPoolConcatMLP(
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

    ckpt = safe_torch_load(reranker_ckpt_path, map_location="cpu")
    state = extract_model_state_from_ckpt(ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info("[reranker load] missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        logging.info("[reranker load] missing sample: %s", missing[:10])
    if unexpected:
        logging.info("[reranker load] unexpected sample: %s", unexpected[:10])

    model.eval()
    return model


# =========================================================
# Candidate tokenization
# =========================================================
def tokenize_candidates_flat(
    go_text_store: GoTextStore,
    cand_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    flat = cand_ids.reshape(-1).detach().cpu().tolist()
    toks = go_text_store.batch(flat)
    return toks


def make_labels_for_candidates(
    cand_ids: torch.Tensor,
    pos_go_global: List[torch.Tensor],
) -> torch.Tensor:
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


# =========================================================
# Retriever top-k
# =========================================================
@torch.no_grad()
def retriever_topk_ids_chunked(
    retriever: ProteinGoAligner,
    H: torch.Tensor,
    valid_mask: torch.Tensor,
    G_once_cpu: torch.Tensor,
    eval_go_ids: List[int],
    topk: int,
    device: torch.device,
    chunk_k: int = SCORE_CHUNK,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            sc = retriever(H=H, G=G_eval, mask=valid_mask, return_alpha=False)
            if isinstance(sc, tuple):
                sc = sc[0]

        sc = sc.float()

        cur_scores, cur_rel = torch.topk(sc, k=min(K, C), dim=1)
        cur_idx = cur_rel + s

        merged_scores = torch.cat([best_scores, cur_scores], dim=1)
        merged_idx = torch.cat([best_idx, cur_idx], dim=1)

        new_scores, new_pos = torch.topk(merged_scores, k=K, dim=1)
        new_idx = merged_idx.gather(1, new_pos)

        best_scores, best_idx = new_scores, new_idx

        del G_chunk, G_eval, sc, cur_scores, cur_rel, cur_idx
        del merged_scores, merged_idx, new_scores, new_pos, new_idx
        maybe_empty_cache(device)

    cand_ids = eval_ids_t.index_select(0, best_idx.reshape(-1).cpu()).view(B, K)
    cand_scores = best_scores.cpu()

    del best_scores, best_idx
    maybe_empty_cache(device)
    maybe_gc()

    return cand_ids, cand_scores


# =========================================================
# Evaluation core
# =========================================================
@torch.no_grad()
def evaluate_method_on_fixed_candidates(
    method_name: str,
    val_loader: DataLoader,
    retriever: ProteinGoAligner,
    reranker: Optional[RerankerMeanPoolConcatMLP],
    G_once_cpu: torch.Tensor,
    eval_go_ids: List[int],
    go_text_store: GoTextStore,
    device: torch.device,
    topk: int = 200,
) -> Tuple[Dict[str, float], List[dict]]:
    """
    method_name:
      - "retriever"
      - "retriever_reranker"
    """
    items_candidate: List[Tuple[float, int]] = []

    n_true_candidate = 0
    hit1 = 0
    hit5 = 0
    hit10 = 0
    sum_mrr = 0.0
    sum_ndcg10 = 0.0
    n_prot = 0

    rows = []

    retriever.eval()
    if reranker is not None:
        reranker.eval()

    for batch_idx, batch in enumerate(val_loader):
        H = batch["prot_emb_pad"].to(device, non_blocking=True)
        valid_mask = valid_mask_from_attn(batch["prot_attn_mask"].to(device, non_blocking=True))
        pos_go_global = batch["pos_go_global"]
        protein_ids = batch["protein_ids"]

        # fixed candidate set from retriever
        cand_ids_cpu, retr_scores_cpu = retriever_topk_ids_chunked(
            retriever=retriever,
            H=H,
            valid_mask=valid_mask,
            G_once_cpu=G_once_cpu,
            eval_go_ids=eval_go_ids,
            topk=topk,
            device=device,
            chunk_k=SCORE_CHUNK,
        )

        labels_cpu = make_labels_for_candidates(cand_ids_cpu, pos_go_global)
        B, K = cand_ids_cpu.shape

        if method_name == "retriever":
            score_cpu = retr_scores_cpu.float()

        elif method_name == "retriever_reranker":
            toks = tokenize_candidates_flat(go_text_store, cand_ids_cpu)
            go_input_ids = toks["input_ids"].to(device, non_blocking=True)
            go_attention_mask = toks["attention_mask"].to(device, non_blocking=True)

            logits = reranker(
                H=H,
                valid_mask=valid_mask,
                go_input_ids=go_input_ids,
                go_attention_mask=go_attention_mask,
                K=K,
                return_alpha=False,
            )
            score_cpu = logits.detach().float().cpu()

            del toks, go_input_ids, go_attention_mask, logits
            maybe_empty_cache(device)

        else:
            raise ValueError(f"Unsupported method_name: {method_name}")

        sc_np = score_cpu.numpy()
        lab_np = labels_cpu.numpy()
        cand_np = cand_ids_cpu.numpy()

        for i in range(B):
            labels_row = [int(x) for x in lab_np[i].tolist()]
            n_true_row = int(sum(labels_row))
            if n_true_row <= 0:
                continue

            n_true_candidate += n_true_row

            order = np.argsort(-sc_np[i])
            sorted_labels = [labels_row[j] for j in order]
            sorted_scores = [float(sc_np[i, j]) for j in order]
            sorted_cands = [int(cand_np[i, j]) for j in order]

            for s, y in zip(sorted_scores, sorted_labels):
                items_candidate.append((s, y))

            top1 = sorted_labels[:1]
            top5 = sorted_labels[:5]
            top10 = sorted_labels[:10]

            hit1 += 1 if any(top1) else 0
            hit5 += 1 if any(top5) else 0
            hit10 += 1 if any(top10) else 0

            rr = reciprocal_rank(sorted_labels)
            ndcg10 = ndcg_at_k(sorted_labels, 10)
            sum_mrr += rr
            sum_ndcg10 += ndcg10
            n_prot += 1

            rows.append({
                "method": method_name,
                "protein_id": str(protein_ids[i]),
                "num_positive_in_candidates": n_true_row,
                "hit_at_1": int(any(top1)),
                "hit_at_5": int(any(top5)),
                "hit_at_10": int(any(top10)),
                "mrr": rr,
                "ndcg_at_10": ndcg10,
                "top10_candidate_ids": "|".join(map(str, sorted_cands[:10])),
                "top10_labels": "|".join(map(str, sorted_labels[:10])),
                "top10_scores": "|".join(f"{x:.6f}" for x in sorted_scores[:10]),
            })

        del H, valid_mask, pos_go_global, protein_ids, cand_ids_cpu, retr_scores_cpu, labels_cpu, score_cpu
        maybe_empty_cache(device)
        maybe_gc()

        if (batch_idx + 1) % 25 == 0:
            logging.info("[%s] processed %d validation batches", method_name, batch_idx + 1)

    pr = compute_global_fmax_aupr_from_items(items_candidate, n_true_candidate)

    metrics = {
        "method": method_name,
        "fmax_at_200": pr["fmax"],
        "aupr_at_200": pr["aupr"],
        "hits_at_1": float(hit1) / max(1, n_prot),
        "hits_at_5": float(hit5) / max(1, n_prot),
        "hits_at_10": float(hit10) / max(1, n_prot),
        "mrr_at_200": float(sum_mrr) / max(1, n_prot),
        "ndcg_at_10": float(sum_ndcg10) / max(1, n_prot),
        "num_proteins_evaluated": float(n_prot),
        "num_true_labels_in_candidates": float(n_true_candidate),
        "num_scored_pairs": float(len(items_candidate)),
    }

    return metrics, rows


# =========================================================
# Main
# =========================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = load_structured_cfg(DEFAULT_YAML_PATH)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not args.retriever_ckpt:
        raise RuntimeError("training.retriever_ckpt yaml içinde boş görünüyor.")
    if args.go_text_folder is None:
        raise RuntimeError("stores.go_text_folder boş.")
    if args.go_cache_path is None:
        raise RuntimeError("stores.go_cache_path boş.")

    reranker_ckpt_path = Path(RERANKER_CKPT_OVERRIDE) if RERANKER_CKPT_OVERRIDE else None
    if reranker_ckpt_path is None:
        raise RuntimeError("RERANKER_CKPT_OVERRIDE ver. Bu script reranker checkpoint ister.")
    if not reranker_ckpt_path.exists():
        raise RuntimeError(f"Reranker checkpoint bulunamadı: {reranker_ckpt_path}")

    logging.info("Device: %s", device)
    logging.info("Retriever checkpoint: %s", args.retriever_ckpt)
    logging.info("Reranker checkpoint: %s", reranker_ckpt_path)

    # 1) GO text store
    go_id_to_text: Dict[int, Dict[int, str]] = {}
    go_id_to_text[PHASE_ID] = load_go_texts_by_phase(args.go_text_folder, phase=PHASE_ID)

    lora_params = LoRAParameters(adapter_name="go_encoder")
    go_enc_wrap = build_go_encoder_from_retriever_ckpt(
        args.retriever_ckpt,
        model_name=args.text_model_name,
        device="cpu",
        max_length=512,
        enable_lora=True,
        use_special_tokens=False,
        lora_parameters=lora_params,
        gradient_checkpointing=False,
    )

    go_text_store = GoTextStore(
        full_id2text=go_id_to_text,
        tokenizer=go_enc_wrap.tokenizer,
        phase=PHASE_ID,
    )

    # 2) stores + datasets
    res_store = build_stores(args)
    go_cache = build_go_cache(str(args.go_cache_path))
    datasets = build_datasets(args, res_store, go_text_store)

    val_ds = datasets["val"]
    if VAL_SUBSET_SIZE is not None:
        val_ds = Subset(val_ds, list(range(min(VAL_SUBSET_SIZE, len(val_ds)))))
        logging.info("Using validation subset: %d", len(val_ds))
    else:
        logging.info("Validation dataset size: %d", len(val_ds))

    # 3) eval GO ids
    text_ids = set(int(x) for x in go_id_to_text[PHASE_ID].keys())
    cache_ids = set(int(x) for x in go_cache.row2id)

    if args.go_path_observed is not None and args.go_path_observed.exists():
        observed_ids = set(load_id_list(args.go_path_observed))
        eval_go_ids = sorted(list(observed_ids & text_ids & cache_ids))
        logging.info("Using observed eval GO ids")
    else:
        eval_go_ids = sorted(list(text_ids & cache_ids))
        logging.info("Observed eval GO ids not found, using text_ids ∩ cache_ids")

    if len(eval_go_ids) == 0:
        raise RuntimeError("eval_go_ids boş çıktı.")

    logging.info("Eval GO count: %d", len(eval_go_ids))

    # 4) val loader
    val_loader = build_val_loader(
        val_ds=val_ds,
        go_text_store=go_text_store,
        eval_go_ids=eval_go_ids,
        batch_size=args.batch_size,
    )

    # 5) GO embeddings from retriever GO encoder, then free encoder
    G_once_cpu, _ = build_eval_G_once(
        eval_go_ids=eval_go_ids,
        go_text_store=go_text_store,
        go_encoder=go_enc_wrap.model,
        chunk=GO_EMB_CHUNK,
    )
    logging.info("G_once shape: %s", tuple(G_once_cpu.shape))

    del go_enc_wrap
    maybe_gc()
    maybe_empty_cache(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 6) load retriever
    retriever = load_retriever_model(
        args=args,
        device=device,
        d_g=768,
    )

    # 7) load reranker
    reranker = load_reranker_model(
        args=args,
        device=device,
        reranker_ckpt_path=reranker_ckpt_path,
    )

    # 8) retriever-only
    retr_metrics, retr_rows = evaluate_method_on_fixed_candidates(
        method_name="retriever",
        val_loader=val_loader,
        retriever=retriever,
        reranker=None,
        G_once_cpu=G_once_cpu,
        eval_go_ids=eval_go_ids,
        go_text_store=go_text_store,
        device=device,
        topk=args.topk,
    )

    logging.info(
        "[retriever] Fmax@200=%.4f AUPR@200=%.4f hits@1=%.4f hits@5=%.4f hits@10=%.4f MRR@200=%.4f nDCG@10=%.4f",
        retr_metrics["fmax_at_200"],
        retr_metrics["aupr_at_200"],
        retr_metrics["hits_at_1"],
        retr_metrics["hits_at_5"],
        retr_metrics["hits_at_10"],
        retr_metrics["mrr_at_200"],
        retr_metrics["ndcg_at_10"],
    )

    # 9) retriever + reranker
    rr_metrics, rr_rows = evaluate_method_on_fixed_candidates(
        method_name="retriever_reranker",
        val_loader=val_loader,
        retriever=retriever,
        reranker=reranker,
        G_once_cpu=G_once_cpu,
        eval_go_ids=eval_go_ids,
        go_text_store=go_text_store,
        device=device,
        topk=args.topk,
    )

    logging.info(
        "[retriever+rereanker] Fmax@200=%.4f AUPR@200=%.4f hits@1=%.4f hits@5=%.4f hits@10=%.4f MRR@200=%.4f nDCG@10=%.4f",
        rr_metrics["fmax_at_200"],
        rr_metrics["aupr_at_200"],
        rr_metrics["hits_at_1"],
        rr_metrics["hits_at_5"],
        rr_metrics["hits_at_10"],
        rr_metrics["mrr_at_200"],
        rr_metrics["ndcg_at_10"],
    )

    # 10) summary table
    summary_rows = [
        {
            "method": "retriever",
            "Fmax@200": retr_metrics["fmax_at_200"],
            "AUPR@200": retr_metrics["aupr_at_200"],
            "Hits@1": retr_metrics["hits_at_1"],
            "Hits@5": retr_metrics["hits_at_5"],
            "Hits@10": retr_metrics["hits_at_10"],
            "MRR@200": retr_metrics["mrr_at_200"],
            "nDCG@10": retr_metrics["ndcg_at_10"],
            "num_proteins": retr_metrics["num_proteins_evaluated"],
        },
        {
            "method": "retriever_reranker",
            "Fmax@200": rr_metrics["fmax_at_200"],
            "AUPR@200": rr_metrics["aupr_at_200"],
            "Hits@1": rr_metrics["hits_at_1"],
            "Hits@5": rr_metrics["hits_at_5"],
            "Hits@10": rr_metrics["hits_at_10"],
            "MRR@200": rr_metrics["mrr_at_200"],
            "nDCG@10": rr_metrics["ndcg_at_10"],
            "num_proteins": rr_metrics["num_proteins_evaluated"],
        },
    ]

    # 11) save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "retriever_metrics.json", retr_metrics)
    save_json(OUTPUT_DIR / "retriever_reranker_metrics.json", rr_metrics)
    save_json(OUTPUT_DIR / "summary_metrics.json", summary_rows)

    save_csv(OUTPUT_DIR / "retriever_per_protein.csv", retr_rows)
    save_csv(OUTPUT_DIR / "retriever_reranker_per_protein.csv", rr_rows)
    save_csv(OUTPUT_DIR / "summary_table.csv", summary_rows)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()