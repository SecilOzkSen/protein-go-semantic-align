from __future__ import annotations

"""
Evaluate a trained P3a-HierCross reranker checkpoint on a candidate dump and
write reusable score arrays for downstream branch/frequency breakdowns.

Recommended use:
  python -m src.script.eval_hiercross_checkpoint \
    --config src/reranker.yaml \
    --checkpoint /workspace/.../ckpt_best_fmax_full_step167890_epoch4.pt \
    --candidate_dump /workspace/candidate_dumps/P3a_SemExp_val_top1000 \
    --split val \
    --topk 500 \
    --out_dir /workspace/results/P3a_HierCross_best

Outputs:
  scores.float32.npy       [N,K]
  cand_ids.int64.npy       [N,K]
  labels.int8.npy          [N,K]
  valid.int8.npy           [N,K]
  true_go_ids.npy          [N,Pmax]
  eval_go_ids.npy          [G]
  protein_ids.json
  metrics_overall.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.main import build_datasets, build_stores
from src.training.collate import ContrastiveEmbCollator
from src.utils.helpers import load_go_texts_by_phase
from src.datasets.go_text_store import GoTextStore
from src.models.reranker_model import RerankerMeanPoolConcatMLP, ResidueGoCrossAttentionReranker

# Reuse the current project-specific helpers from reranker_main.
from src.reranker_main import (
    CandidateDumpLookup,
    compute_global_fmax_aupr_from_items,
    load_structured_cfg,
    safe_torch_load,
    valid_mask_from_attn,
    tokenize_candidates_flat,
    make_rank_feature,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _looks_like_state_dict(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    n = 0
    for v in x.values():
        if torch.is_tensor(v):
            n += 1
            if n >= 3:
                return True
    return False


def _unwrap_model_state(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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
    raise RuntimeError(f"Could not find model state_dict. Top-level keys={list(ckpt.keys())[:30]}")


def _clean_state_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
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


def build_reranker_model(args, device: torch.device) -> torch.nn.Module:
    if args.model_kind in {"hiercross", "residue_go_cross_attention", "p3a_hiercross"}:
        model = ResidueGoCrossAttentionReranker(
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
        )
    else:
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
        )
    return model.to(device)


def load_model_checkpoint(model: torch.nn.Module, checkpoint: str | Path, device: torch.device) -> Dict[str, Any]:
    ckpt = safe_torch_load(str(checkpoint), map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint must be dict, got {type(ckpt)}")
    state = _clean_state_keys(_unwrap_model_state(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info("[ckpt] loaded model missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        logging.warning("[ckpt] missing sample: %s", missing[:25])
    if unexpected:
        logging.warning("[ckpt] unexpected sample: %s", unexpected[:25])
    return ckpt


def make_eval_loader(args, go_text_store: GoTextStore, eval_go_ids: np.ndarray, split: str) -> DataLoader:
    res_store = build_stores(args)
    datasets = build_datasets(args, res_store, go_text_store)
    ds = datasets[split]
    collate = ContrastiveEmbCollator(
        zs_mask_vec=torch.ones(len(eval_go_ids), dtype=torch.bool),
        go_text_store=go_text_store,
        bidirectional=False,
        neg_k=0,
        device=torch.device("cpu"),
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )


def ensure_numpy_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def compute_metrics_from_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    true_go_ids: np.ndarray,
) -> Dict[str, float]:
    items: List[Tuple[float, int]] = []
    n_true_full = 0
    n_true_candidate = 0
    hit1 = hit5 = hit10 = 0
    n_prot = int(scores.shape[0])

    for i in range(n_prot):
        true_set = set(int(x) for x in true_go_ids[i].tolist() if int(x) >= 0)
        n_true_full += len(true_set)
        n_true_candidate += int(((labels[i] > 0) & (valid[i] > 0)).sum())

        for j in range(scores.shape[1]):
            if valid[i, j] > 0:
                items.append((float(scores[i, j]), int(labels[i, j])))

        order = np.argsort(-scores[i])
        order = [j for j in order if valid[i, j] > 0]
        # Hit metrics require candidate ids, so they are computed in the main loop.
        # Filled here as zero and overwritten by caller if needed.

    topk = compute_global_fmax_aupr_from_items(items.copy(), int(n_true_candidate))
    full = compute_global_fmax_aupr_from_items(items.copy(), int(n_true_full))
    return {
        "fmax_topk": float(topk["fmax"]),
        "aupr_topk": float(topk["aupr"]),
        "fmax_full": float(full["fmax"]),
        "aupr_full": float(full["aupr"]),
        "n_true_candidate": float(n_true_candidate),
        "n_true_full": float(n_true_full),
        "n_pairs_scored": float(len(items)),
        "retrieval_recall@K": float(n_true_candidate) / max(1.0, float(n_true_full)),
        "oracle_microF@K": float((2.0 * n_true_candidate) / max(1e-12, 2.0 * n_true_candidate + (n_true_full - n_true_candidate))),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate P3a-HierCross checkpoint and dump scores.")
    parser.add_argument("--config", type=str, default="src/reranker.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--candidate_dump", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--phase", type=int, default=-2)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=0)
    cli = parser.parse_args()

    setup_logging()
    args = load_structured_cfg(cli.config)
    if cli.topk is not None:
        args.topk = int(cli.topk)
    if cli.batch_size is not None:
        args.batch_size = int(cli.batch_size)
    if cli.device is not None:
        args.device = str(cli.device)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(cli.out_dir)
    ensure_numpy_dir(out_dir)

    candidate_lookup = CandidateDumpLookup(cli.candidate_dump, topk=args.topk)
    eval_go_ids = np.asarray(candidate_lookup.eval_go_ids, dtype=np.int64)
    np.save(out_dir / "eval_go_ids.npy", eval_go_ids)

    logging.info("[eval] candidate_dump=%s", cli.candidate_dump)
    logging.info("[eval] split=%s topk=%d eval_go_ids=%d", cli.split, args.topk, len(eval_go_ids))

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    go_id_to_text = {int(cli.phase): load_go_texts_by_phase(args.go_text_folder, phase=int(cli.phase))}
    go_text_store = GoTextStore(full_id2text=go_id_to_text, tokenizer=tokenizer, phase=int(cli.phase))

    val_loader = make_eval_loader(args, go_text_store, eval_go_ids, cli.split)

    model = build_reranker_model(args, device)
    ckpt = load_model_checkpoint(model, cli.checkpoint, device)
    model.eval()

    all_scores: List[np.ndarray] = []
    all_cand: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_valid: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    all_pids: List[str] = []

    hit1 = hit5 = hit10 = 0
    n_prot = 0

    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(val_loader, desc="eval-hiercross")):
            if cli.max_batches and bidx >= cli.max_batches:
                break

            pids = [str(x) for x in batch["protein_ids"]]
            cand_ids, cand_scores, labels, true_ids, valid_cpu = candidate_lookup.get_batch(pids)

            H = batch["prot_emb_pad"].to(device, non_blocking=True)
            valid_mask = valid_mask_from_attn(batch["prot_attn_mask"].to(device, non_blocking=True))

            toks = tokenize_candidates_flat(go_text_store, cand_ids)
            go_input_ids = toks["input_ids"].to(device, non_blocking=True)
            go_attention_mask = toks["attention_mask"].to(device, non_blocking=True)

            B, K = cand_ids.shape
            logits = model(
                H=H,
                K=K,
                valid_mask=valid_mask,
                go_input_ids=go_input_ids,
                go_attention_mask=go_attention_mask,
                retriever_score=cand_scores.to(device, non_blocking=True),
                rank_feature=make_rank_feature(B, K, device),
            ).detach().float().cpu()

            scores_np = logits.numpy().astype(np.float32)
            labels_np = labels.numpy().astype(np.int8)
            cand_np = cand_ids.numpy().astype(np.int64)
            valid_np = valid_cpu.numpy().astype(np.int8)
            true_np = true_ids.numpy().astype(np.int64)

            # Protein-level hit metrics.
            for i in range(B):
                true_set = set(int(x) for x in true_np[i].tolist() if int(x) >= 0)
                order = np.argsort(-scores_np[i])
                order = [j for j in order if valid_np[i, j] > 0]
                top1 = [int(cand_np[i, order[0]])] if len(order) > 0 else []
                top5 = [int(cand_np[i, j]) for j in order[: min(5, len(order))]]
                top10 = [int(cand_np[i, j]) for j in order[: min(10, len(order))]]
                hit1 += 1 if any(g in true_set for g in top1) else 0
                hit5 += 1 if any(g in true_set for g in top5) else 0
                hit10 += 1 if any(g in true_set for g in top10) else 0
                n_prot += 1

            all_scores.append(scores_np)
            all_cand.append(cand_np)
            all_labels.append(labels_np)
            all_valid.append(valid_np)
            all_true.append(true_np)
            all_pids.extend(pids)

    scores = np.concatenate(all_scores, axis=0)
    cand_ids = np.concatenate(all_cand, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    valid = np.concatenate(all_valid, axis=0)
    true_go_ids = np.concatenate(all_true, axis=0)

    np.save(out_dir / "scores.float32.npy", scores.astype(np.float32))
    np.save(out_dir / "cand_ids.int64.npy", cand_ids.astype(np.int64))
    np.save(out_dir / "labels.int8.npy", labels.astype(np.int8))
    np.save(out_dir / "valid.int8.npy", valid.astype(np.int8))
    np.save(out_dir / "true_go_ids.npy", true_go_ids.astype(np.int64))
    with (out_dir / "protein_ids.json").open("w", encoding="utf-8") as f:
        json.dump(all_pids, f)

    metrics = compute_metrics_from_arrays(scores, labels, valid, true_go_ids)
    metrics.update({
        "hits@1": float(hit1) / max(1, n_prot),
        "hits@5": float(hit5) / max(1, n_prot),
        "hits@10": float(hit10) / max(1, n_prot),
        "n_prot": float(n_prot),
        "checkpoint": str(cli.checkpoint),
        "candidate_dump": str(cli.candidate_dump),
        "topk": int(args.topk),
    })

    with (out_dir / "metrics_overall.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logging.info("[metrics] %s", metrics)
    logging.info("[done] wrote score dump to %s", str(out_dir))


if __name__ == "__main__":
    main()
