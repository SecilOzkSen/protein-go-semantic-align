from __future__ import annotations

import csv
import json
import logging
import pickle
from pathlib import Path, PosixPath
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional
from src.utils.helpers import go_str_to_int_any

import torch
import yaml
from torch.utils.data import DataLoader

# Project imports
from src.configs.paths import SRC_DIR
from src.main import build_datasets, build_stores, build_go_cache
from src.training.collate import ContrastiveEmbCollator
from src.models.alignment_model import ProteinGoAligner
from src.encoders.go_encoder import BioMedBERTEncoder, LoRAParameters
from src.datasets.go_text_store import GoTextStore
from src.utils.helpers import load_go_texts_by_phase


# =========================================================
# CHANGE ONLY THESE IF NEEDED
# =========================================================
DEFAULT_YAML_PATH = SRC_DIR / "reranker.yaml"
PHASE_ID = -2
TOPK = 200
BATCH_SIZE = 4
GO_EMB_CHUNK = 256
SCORE_CHUNK = 2048
OUTPUT_DIR = Path("outputs/rare_go_analysis")

RARE_IDS_PATH_OVERRIDE: Optional[str] = None

def safe_torch_load(path: str, map_location="cpu"):
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

def load_id_list(path: Path) -> List[int]:
    if path.name.endswith(".json"):
        with open(path, "r") as f:
            xs = json.load(f)
        return [go_str_to_int_any(x) for x in xs]

    if path.name.endswith(".pkl") or path.name.endswith(".pickle"):
        with open(path, "rb") as f:
            xs = pickle.load(f)
        return [go_str_to_int_any(x) for x in xs]

    # default txt
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(int(s))

    return out


def valid_mask_from_attn(attn: torch.Tensor) -> torch.Tensor:
    if attn.dtype != torch.bool:
        attn = attn != 0
    return attn


def load_structured_cfg(path: str | Path = DEFAULT_YAML_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model = cfg.get("model", {})
    training = cfg.get("training", {})
    stores = cfg.get("stores", {})
    data = cfg.get("data", {})

    args = SimpleNamespace(
        text_model_name=model.get(
            "text_model_name",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        ),
        protein_dim=int(model.get("protein_dim", 1280)),
        protein_pool_type=str(model.get("protein_pool_type", "go_align")),
        protein_n_slots=int(model.get("protein_n_slots", 4)),
        device=str(training.get("device", "cuda")),
        fp16=bool(training.get("fp16", True)),
        retriever_ckpt=str(training.get("retriever_ckpt", "")),
        batch_size=int(training.get("batch_size", BATCH_SIZE)),
        topk=int(training.get("topk", TOPK)),
        go_cache_path=Path(stores.get("go_cache_path", "")),
        go_text_folder=Path(stores.get("go_text_folder", "")),
        few_shot_path=Path(stores.get("few_shot_path", "")),
        go_path_observed=Path(stores.get("go_path_observed", "")),
        max_len=int(data.get("protein_max_len", 1024)),
        overlap=int(data.get("overlap", 128)),
        fs_target_ratio=float(data.get("fs_target_ratio", 0.1)),
        embed_dir_res=Path(stores.get("embed_dir_res", None)),
        pid2pos=Path(stores.get("pid2pos_path", None)),
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
        train_ids_path=Path(stores.get("train_ids_path", "")),
        val_ids_path=Path(stores.get("val_ids_path", "")),
        embed_dir_fused=Path(stores.get("embed_dir_fused", "")),
        seq_len_lookup_dir=Path(stores.get("seq_len_lookup_dir", "")),
        dag_parents_path=Path(stores.get("dag_parents_path", "")),
        go_basic_json=Path(stores.get("go_basic_json", "")),
        zero_shot_path=Path(stores.get("zero_shot_path", "")),
        go_path_seen=Path(stores.get("go_path_seen", "")),
    )
    return args


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
    logging.info("[go_encoder load] missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        logging.info("[go_encoder load] missing sample: %s", missing[:10])
    if unexpected:
        logging.info("[go_encoder load] unexpected sample: %s", unexpected[:10])

    return enc


def build_eval_G_once(
    eval_go_ids: List[int],
    go_text_store: GoTextStore,
    go_encoder: torch.nn.Module,
    device: torch.device,
    chunk: int = GO_EMB_CHUNK,
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
        pin_memory=True,
    )
    return loader


def load_retriever_model(
    args,
    device: torch.device,
    go_encoder_wrapper,
):
    retriever = ProteinGoAligner(
        d_h=args.protein_dim,
        d_g=None,
        d_z=768,
        go_encoder=go_encoder_wrapper.model,
        normalize=True,
        protein_pool_type=args.protein_pool_type,
        protein_n_slots=getattr(args, "protein_n_slots", 4),
    ).to(device)

    ckpt = safe_torch_load(args.retriever_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = state.get("model", state)

    # go_encoder already restored separately
    state = {k: v for k, v in state.items() if not k.startswith("go_encoder.")}

    missing, unexpected = retriever.load_state_dict(state, strict=False)
    logging.info("[retriever load] missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        logging.info("[retriever load] missing sample: %s", missing[:10])
    if unexpected:
        logging.info("[retriever load] unexpected sample: %s", unexpected[:10])

    retriever.eval()
    return retriever


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
    """
    Returns:
      cand_ids: [B,K] global GO ids on CPU
      cand_scores: [B,K] scores on CPU
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
        sc = sc.float()

        cur_scores, cur_rel = torch.topk(sc, k=min(K, C), dim=1)
        cur_idx = cur_rel + s

        merged_scores = torch.cat([best_scores, cur_scores], dim=1)
        merged_idx = torch.cat([best_idx, cur_idx], dim=1)

        new_scores, new_pos = torch.topk(merged_scores, k=K, dim=1)
        new_idx = merged_idx.gather(1, new_pos)

        best_scores, best_idx = new_scores, new_idx

    cand_ids = eval_ids_t.index_select(0, best_idx.reshape(-1).cpu()).view(B, K)
    return cand_ids.cpu(), best_scores.cpu()


@torch.no_grad()
def evaluate_rare_go_recall(
    retriever: ProteinGoAligner,
    val_loader: DataLoader,
    G_once_cpu: torch.Tensor,
    eval_go_ids: List[int],
    rare_go_ids: List[int],
    device: torch.device,
    topk: int = 200,
):
    rare_go_set = set(int(x) for x in rare_go_ids)
    eval_go_set = set(int(x) for x in eval_go_ids)

    proteins_with_rare = 0
    proteins_with_rare_hit_at_50 = 0
    proteins_with_rare_hit_at_200 = 0

    rare_r50_sum = 0.0
    rare_r200_sum = 0.0

    total_rare_true = 0
    total_rare_hits_at_50 = 0
    total_rare_hits_at_200 = 0

    per_go_total = {}
    per_go_hit50 = {}
    per_go_hit200 = {}

    protein_rows = []

    for batch_idx, batch in enumerate(val_loader):
        H = batch["prot_emb_pad"].to(device, non_blocking=True)
        valid_mask = valid_mask_from_attn(batch["prot_attn_mask"].to(device, non_blocking=True))
        pos_go_global = batch["pos_go_global"]
        protein_ids = batch["protein_ids"]

        cand_ids_cpu, cand_scores_cpu = retriever_topk_ids_chunked(
            retriever=retriever,
            H=H,
            valid_mask=valid_mask,
            G_once_cpu=G_once_cpu,
            eval_go_ids=eval_go_ids,
            topk=topk,
            device=device,
            chunk_k=SCORE_CHUNK,
        )

        B, K = cand_ids_cpu.shape

        for i in range(B):
            pid = str(protein_ids[i])

            true_pos = pos_go_global[i]
            if true_pos is None:
                continue

            true_pos_list = [int(x) for x in true_pos.detach().cpu().tolist()]
            # keep only observed eval-space positives
            true_pos_eval = [g for g in true_pos_list if g in eval_go_set]
            rare_true = [g for g in true_pos_eval if g in rare_go_set]

            if len(rare_true) == 0:
                continue

            proteins_with_rare += 1
            total_rare_true += len(rare_true)

            for gid in rare_true:
                per_go_total[gid] = per_go_total.get(gid, 0) + 1

            row_ids = [int(x) for x in cand_ids_cpu[i].tolist()]
            row_top50 = row_ids[:50]
            row_top200 = row_ids[:200]

            rare_true_set = set(rare_true)
            hit50 = sorted(list(rare_true_set.intersection(row_top50)))
            hit200 = sorted(list(rare_true_set.intersection(row_top200)))

            r50 = len(hit50) / len(rare_true_set)
            r200 = len(hit200) / len(rare_true_set)

            rare_r50_sum += r50
            rare_r200_sum += r200

            total_rare_hits_at_50 += len(hit50)
            total_rare_hits_at_200 += len(hit200)

            if len(hit50) > 0:
                proteins_with_rare_hit_at_50 += 1
            if len(hit200) > 0:
                proteins_with_rare_hit_at_200 += 1

            for gid in hit50:
                per_go_hit50[gid] = per_go_hit50.get(gid, 0) + 1
            for gid in hit200:
                per_go_hit200[gid] = per_go_hit200.get(gid, 0) + 1

            protein_rows.append({
                "protein_id": pid,
                "num_true_pos_eval": len(true_pos_eval),
                "num_true_rare": len(rare_true),
                "num_rare_hit_at_50": len(hit50),
                "num_rare_hit_at_200": len(hit200),
                "rare_recall_at_50": r50,
                "rare_recall_at_200": r200,
                "rare_true_go_ids": "|".join(map(str, rare_true)),
                "rare_hit_go_ids_at_50": "|".join(map(str, hit50)),
                "rare_hit_go_ids_at_200": "|".join(map(str, hit200)),
                "top20_candidate_go_ids": "|".join(map(str, row_ids[:20])),
            })

        if (batch_idx + 1) % 50 == 0:
            logging.info("Processed %d validation batches", batch_idx + 1)

    metrics = {
        "proteins_with_rare": proteins_with_rare,
        "proteins_with_rare_hit_at_50": proteins_with_rare_hit_at_50,
        "proteins_with_rare_hit_at_200": proteins_with_rare_hit_at_200,
        "rare_hit_protein_rate_at_50": (
            proteins_with_rare_hit_at_50 / proteins_with_rare if proteins_with_rare > 0 else 0.0
        ),
        "rare_hit_protein_rate_at_200": (
            proteins_with_rare_hit_at_200 / proteins_with_rare if proteins_with_rare > 0 else 0.0
        ),
        "rare_R_at_50_macro": (
            rare_r50_sum / proteins_with_rare if proteins_with_rare > 0 else 0.0
        ),
        "rare_R_at_200_macro": (
            rare_r200_sum / proteins_with_rare if proteins_with_rare > 0 else 0.0
        ),
        "avg_num_rare_true_per_protein": (
            total_rare_true / proteins_with_rare if proteins_with_rare > 0 else 0.0
        ),
        "avg_num_rare_hits_at_50": (
            total_rare_hits_at_50 / proteins_with_rare if proteins_with_rare > 0 else 0.0
        ),
        "avg_num_rare_hits_at_200": (
            total_rare_hits_at_200 / proteins_with_rare if proteins_with_rare > 0 else 0.0
        ),
        "total_rare_true_labels": total_rare_true,
        "total_rare_hits_at_50": total_rare_hits_at_50,
        "total_rare_hits_at_200": total_rare_hits_at_200,
    }

    go_rows = []
    for gid, total_cnt in per_go_total.items():
        h50 = per_go_hit50.get(gid, 0)
        h200 = per_go_hit200.get(gid, 0)
        go_rows.append({
            "go_id": gid,
            "num_proteins_with_go": total_cnt,
            "num_hits_at_50": h50,
            "num_hits_at_200": h200,
            "hit_rate_at_50": h50 / total_cnt if total_cnt > 0 else 0.0,
            "hit_rate_at_200": h200 / total_cnt if total_cnt > 0 else 0.0,
        })

    go_rows = sorted(
        go_rows,
        key=lambda x: (x["hit_rate_at_200"], x["num_proteins_with_go"]),
        reverse=True,
    )

    return metrics, protein_rows, go_rows


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

    if RARE_IDS_PATH_OVERRIDE is not None:
        rare_ids_path = Path(RARE_IDS_PATH_OVERRIDE)
    else:
        rare_ids_path = args.few_shot_path

    if not rare_ids_path or not rare_ids_path.exists():
        raise RuntimeError(
            f"Rare GO ids path bulunamadı: {rare_ids_path}. "
            f"RARE_IDS_PATH_OVERRIDE ya da few_shot_path düzelt."
        )

    logging.info("Device: %s", device)
    logging.info("Retriever checkpoint: %s", args.retriever_ckpt)
    logging.info("Rare GO ids path: %s", rare_ids_path)

    # 1) GO text store
    go_id_to_text: Dict[int, Dict[int, str]] = {}
    go_id_to_text[PHASE_ID] = load_go_texts_by_phase(args.go_text_folder, phase=PHASE_ID)

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

    go_text_store = GoTextStore(
        full_id2text=go_id_to_text,
        tokenizer=go_enc_wrap.tokenizer,
        phase=PHASE_ID,
    )

    # 2) Stores and datasets
    res_store = build_stores(args)
    go_cache = build_go_cache(str(args.go_cache_path))
    datasets = build_datasets(args, res_store, go_text_store)

    val_ds = datasets["val"]
    logging.info("Validation dataset size: %d", len(val_ds))

    # 3) Eval GO ids, use observed if available
    text_ids = set(int(x) for x in go_id_to_text[PHASE_ID].keys())
    cache_ids = set(int(x) for x in go_cache.row2id)

    if args.go_path_observed and Path(args.go_path_observed).exists():
        observed_ids = set(load_id_list(args.go_path_observed))
        eval_go_ids = sorted(list(observed_ids & text_ids & cache_ids))
        logging.info("Using observed eval GO ids")
    else:
        eval_go_ids = sorted(list(text_ids & cache_ids))
        logging.info("Observed eval GO ids not found, using text_ids ∩ cache_ids")

    if len(eval_go_ids) == 0:
        raise RuntimeError("eval_go_ids boş çıktı, GO text / cache / observed dosyalarını kontrol et.")

    logging.info("Eval GO count: %d", len(eval_go_ids))

    # 4) Rare GO ids
    rare_go_ids = sorted(set(load_id_list(rare_ids_path)))
    logging.info("Rare GO count, raw: %d", len(rare_go_ids))

    eval_go_set = set(eval_go_ids)
    rare_go_ids = [g for g in rare_go_ids if g in eval_go_set]
    logging.info("Rare GO count inside eval space: %d", len(rare_go_ids))

    if len(rare_go_ids) == 0:
        raise RuntimeError("Eval space içinde rare GO kalmadı.")

    # 5) Dataloader
    val_loader = build_val_loader(
        val_ds=val_ds,
        go_text_store=go_text_store,
        eval_go_ids=eval_go_ids,
        batch_size=args.batch_size,
    )

    # 6) Build G_once
    G_once_cpu, _ = build_eval_G_once(
        eval_go_ids=eval_go_ids,
        go_text_store=go_text_store,
        go_encoder=go_enc_wrap.model.to(device),
        device=device,
        chunk=GO_EMB_CHUNK,
    )
    logging.info("G_once shape: %s", tuple(G_once_cpu.shape))

    # 7) Retriever model
    retriever = load_retriever_model(
        args=args,
        device=device,
        go_encoder_wrapper=go_enc_wrap,
    )

    # 8) Evaluate
    metrics, protein_rows, go_rows = evaluate_rare_go_recall(
        retriever=retriever,
        val_loader=val_loader,
        G_once_cpu=G_once_cpu,
        eval_go_ids=eval_go_ids,
        rare_go_ids=rare_go_ids,
        device=device,
        topk=args.topk,
    )

    # 9) Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "rare_metrics.json", metrics)
    save_csv(OUTPUT_DIR / "rare_per_protein.csv", protein_rows)
    save_csv(OUTPUT_DIR / "rare_per_go.csv", go_rows)

    logging.info("Saved metrics -> %s", OUTPUT_DIR / "rare_metrics.json")
    logging.info("Saved protein csv -> %s", OUTPUT_DIR / "rare_per_protein.csv")
    logging.info("Saved GO csv -> %s", OUTPUT_DIR / "rare_per_go.csv")

    print("\n=== RARE GO RETRIEVAL METRICS ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()