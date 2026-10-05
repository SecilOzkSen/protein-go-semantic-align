from __future__ import annotations

import csv
import json
import logging
from pathlib import Path, PosixPath
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from src.configs.paths import SRC_DIR
from src.main import build_datasets, build_stores, build_go_cache
from src.training.collate import ContrastiveEmbCollator
from src.models.alignment_model import ProteinGoAligner
from src.encoders.go_encoder import BioMedBERTEncoder, LoRAParameters
from src.datasets.go_text_store import GoTextStore
from src.utils.helpers import load_go_texts_by_phase


DEFAULT_YAML_PATH = SRC_DIR / "reranker.yaml"
PHASE_ID = -2
TOPK = 200
GO_EMB_CHUNK = 256
SCORE_CHUNK = 2048
OUTPUT_DIR = Path("outputs/go_hierarchy_consistency")


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


def _go_str_to_int(x) -> int:
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s.upper().startswith("GO:"):
        s = s.split(":", 1)[1]
    return int(s)


def load_id_list(path: str | Path) -> List[int]:
    path = str(path)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            xs = json.load(f)
        return [_go_str_to_int(x) for x in xs]

    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(_go_str_to_int(s))
    return out


def load_child_to_parents_json(path: str | Path) -> Dict[int, List[int]]:
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
        batch_size=int(training.get("batch_size", 4)),
        topk=int(training.get("topk", TOPK)),
        go_cache_path=Path(stores.get("go_cache_path", "")),
        go_text_folder=Path(stores.get("go_text_folder", "")),
        go_path_observed=Path(stores.get("go_path_observed", "")),
        dag_parents_path=Path(stores.get("dag_parents_path", "")),
        max_len=int(data.get("protein_max_len", 1024)),
        overlap=int(data.get("overlap", 128)),
        fs_target_ratio=float(data.get("fs_target_ratio", 0.1)),
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


def load_retriever_model(args, device: torch.device, go_encoder_wrapper):
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
def evaluate_go_hierarchy_consistency(
    retriever: ProteinGoAligner,
    val_loader: DataLoader,
    G_once_cpu: torch.Tensor,
    eval_go_ids: List[int],
    child_to_parents: Dict[int, List[int]],
    device: torch.device,
    topk: int = 200,
):
    eval_go_set = set(int(x) for x in eval_go_ids)

    total_true_child_parent_pairs = 0
    total_true_child_parent_pairs_hit = 0

    total_pred_child_parent_pairs = 0
    total_pred_child_parent_pairs_hit = 0

    proteins_with_true_child_parent_pair = 0
    proteins_strict_consistent_true_pair = 0

    proteins_with_pred_child_parent_pair = 0
    proteins_strict_consistent_pred_pair = 0

    protein_rows = []
    go_rows_map = {}

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
            true_pos_eval = sorted(set(g for g in true_pos_list if g in eval_go_set))

            row_ids = [int(x) for x in cand_ids_cpu[i].tolist()]
            row_set = set(row_ids)

            # Case 1: true child exists in labels, do its true parent(s) appear in top200?
            true_pairs = []
            for child in true_pos_eval:
                parents = child_to_parents.get(child, [])
                parents = [p for p in parents if p in eval_go_set]
                for p in parents:
                    true_pairs.append((child, p))

            if len(true_pairs) > 0:
                proteins_with_true_child_parent_pair += 1

            true_hits = 0
            for child, parent in true_pairs:
                total_true_child_parent_pairs += 1
                ok = (child in row_set) and (parent in row_set)
                if ok:
                    total_true_child_parent_pairs_hit += 1
                    true_hits += 1

                key = (child, parent)
                if key not in go_rows_map:
                    go_rows_map[key] = {
                        "child_go": child,
                        "parent_go": parent,
                        "num_true_pairs": 0,
                        "num_true_pairs_hit": 0,
                        "num_pred_child_cases": 0,
                        "num_pred_child_cases_hit_parent": 0,
                    }
                go_rows_map[key]["num_true_pairs"] += 1
                if ok:
                    go_rows_map[key]["num_true_pairs_hit"] += 1

            strict_true_consistent = (len(true_pairs) > 0 and true_hits == len(true_pairs))
            if strict_true_consistent:
                proteins_strict_consistent_true_pair += 1

            # Case 2: predicted child is in top200, do its parents also appear in top200?
            pred_pairs = []
            for child in row_ids:
                parents = child_to_parents.get(child, [])
                parents = [p for p in parents if p in eval_go_set]
                for p in parents:
                    pred_pairs.append((child, p))

            if len(pred_pairs) > 0:
                proteins_with_pred_child_parent_pair += 1

            pred_hits = 0
            for child, parent in pred_pairs:
                total_pred_child_parent_pairs += 1
                ok = parent in row_set
                if ok:
                    total_pred_child_parent_pairs_hit += 1
                    pred_hits += 1

                key = (child, parent)
                if key not in go_rows_map:
                    go_rows_map[key] = {
                        "child_go": child,
                        "parent_go": parent,
                        "num_true_pairs": 0,
                        "num_true_pairs_hit": 0,
                        "num_pred_child_cases": 0,
                        "num_pred_child_cases_hit_parent": 0,
                    }
                go_rows_map[key]["num_pred_child_cases"] += 1
                if ok:
                    go_rows_map[key]["num_pred_child_cases_hit_parent"] += 1

            strict_pred_consistent = (len(pred_pairs) > 0 and pred_hits == len(pred_pairs))
            if strict_pred_consistent:
                proteins_strict_consistent_pred_pair += 1

            protein_rows.append({
                "protein_id": pid,
                "num_true_pos_eval": len(true_pos_eval),
                "num_true_child_parent_pairs": len(true_pairs),
                "num_true_child_parent_pairs_hit": true_hits,
                "true_pair_coverage": (
                    true_hits / len(true_pairs) if len(true_pairs) > 0 else 0.0
                ),
                "strict_true_pair_consistent": int(strict_true_consistent),
                "num_pred_child_parent_pairs": len(pred_pairs),
                "num_pred_child_parent_pairs_hit": pred_hits,
                "pred_pair_coverage": (
                    pred_hits / len(pred_pairs) if len(pred_pairs) > 0 else 0.0
                ),
                "strict_pred_pair_consistent": int(strict_pred_consistent),
                "top20_candidate_go_ids": "|".join(map(str, row_ids[:20])),
            })

        if (batch_idx + 1) % 50 == 0:
            logging.info("Processed %d validation batches", batch_idx + 1)

    go_rows = []
    for _, row in go_rows_map.items():
        row = dict(row)
        row["true_pair_hit_rate"] = (
            row["num_true_pairs_hit"] / row["num_true_pairs"] if row["num_true_pairs"] > 0 else 0.0
        )
        row["pred_pair_parent_coverage"] = (
            row["num_pred_child_cases_hit_parent"] / row["num_pred_child_cases"]
            if row["num_pred_child_cases"] > 0 else 0.0
        )
        go_rows.append(row)

    go_rows = sorted(
        go_rows,
        key=lambda x: (x["true_pair_hit_rate"], x["num_true_pairs"]),
        reverse=True,
    )

    metrics = {
        "total_true_child_parent_pairs": total_true_child_parent_pairs,
        "total_true_child_parent_pairs_hit": total_true_child_parent_pairs_hit,
        "parent_coverage_given_true_child_at_200": (
            total_true_child_parent_pairs_hit / total_true_child_parent_pairs
            if total_true_child_parent_pairs > 0 else 0.0
        ),
        "proteins_with_true_child_parent_pair": proteins_with_true_child_parent_pair,
        "proteins_strict_consistent_true_pair": proteins_strict_consistent_true_pair,
        "strict_hierarchy_consistency_true_pair_at_200": (
            proteins_strict_consistent_true_pair / proteins_with_true_child_parent_pair
            if proteins_with_true_child_parent_pair > 0 else 0.0
        ),
        "total_pred_child_parent_pairs": total_pred_child_parent_pairs,
        "total_pred_child_parent_pairs_hit": total_pred_child_parent_pairs_hit,
        "parent_coverage_given_predicted_child_at_200": (
            total_pred_child_parent_pairs_hit / total_pred_child_parent_pairs
            if total_pred_child_parent_pairs > 0 else 0.0
        ),
        "proteins_with_pred_child_parent_pair": proteins_with_pred_child_parent_pair,
        "proteins_strict_consistent_pred_pair": proteins_strict_consistent_pred_pair,
        "strict_hierarchy_consistency_pred_pair_at_200": (
            proteins_strict_consistent_pred_pair / proteins_with_pred_child_parent_pair
            if proteins_with_pred_child_parent_pair > 0 else 0.0
        ),
    }

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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = load_structured_cfg(DEFAULT_YAML_PATH)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not args.retriever_ckpt:
        raise RuntimeError("training.retriever_ckpt yaml içinde boş görünüyor.")
    if not args.dag_parents_path or not Path(args.dag_parents_path).exists():
        raise RuntimeError("stores.dag_parents_path bulunamadı.")

    logging.info("Device: %s", device)
    logging.info("Retriever checkpoint: %s", args.retriever_ckpt)
    logging.info("DAG parents path: %s", args.dag_parents_path)

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

    res_store = build_stores(args)
    go_cache = build_go_cache(str(args.go_cache_path))
    datasets = build_datasets(args, res_store, go_text_store)

    val_ds = datasets["val"]
    logging.info("Validation dataset size: %d", len(val_ds))

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
        raise RuntimeError("eval_go_ids boş çıktı.")

    logging.info("Eval GO count: %d", len(eval_go_ids))

    child_to_parents = load_child_to_parents_json(args.dag_parents_path)
    logging.info("Loaded child->parents entries: %d", len(child_to_parents))

    val_loader = build_val_loader(
        val_ds=val_ds,
        go_text_store=go_text_store,
        eval_go_ids=eval_go_ids,
        batch_size=args.batch_size,
    )

    G_once_cpu, _ = build_eval_G_once(
        eval_go_ids=eval_go_ids,
        go_text_store=go_text_store,
        go_encoder=go_enc_wrap.model.to(device),
        device=device,
        chunk=GO_EMB_CHUNK,
    )
    logging.info("G_once shape: %s", tuple(G_once_cpu.shape))

    retriever = load_retriever_model(
        args=args,
        device=device,
        go_encoder_wrapper=go_enc_wrap,
    )

    metrics, protein_rows, go_rows = evaluate_go_hierarchy_consistency(
        retriever=retriever,
        val_loader=val_loader,
        G_once_cpu=G_once_cpu,
        eval_go_ids=eval_go_ids,
        child_to_parents=child_to_parents,
        device=device,
        topk=args.topk,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "hierarchy_metrics.json", metrics)
    save_csv(OUTPUT_DIR / "hierarchy_per_protein.csv", protein_rows)
    save_csv(OUTPUT_DIR / "hierarchy_per_pair.csv", go_rows)

    logging.info("Saved metrics -> %s", OUTPUT_DIR / "hierarchy_metrics.json")
    logging.info("Saved protein csv -> %s", OUTPUT_DIR / "hierarchy_per_protein.csv")
    logging.info("Saved pair csv -> %s", OUTPUT_DIR / "hierarchy_per_pair.csv")

    print("\n=== GO HIERARCHY CONSISTENCY METRICS ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()