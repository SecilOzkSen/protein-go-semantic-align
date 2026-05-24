'''
# Train

python scripts/dump_retriever_candidates.py \
  --config src/runpod.yaml \
  --checkpoint /workspace/protein-go-align-outputs-P3a-P2cwarm-segGO-segProjOnly/checkpoint_step83945.pt \
  --split train \
  --topk 1000 \
  --batch_size 4 \
  --num_workers 0 \
  --out_dir /workspace/candidate_dumps/P3a_train_top1000 \
  --strict_exact

# Validation

python scripts/dump_retriever_candidates.py \
  --config src/runpod.yaml \
  --checkpoint /workspace/protein-go-align-outputs-P3a-P2cwarm-segGO-segProjOnly/checkpoint_step83945.pt \
  --split val \
  --topk 1000 \
  --batch_size 4 \
  --num_workers 0 \
  --out_dir /workspace/candidate_dumps/P3a_val_top1000 \
  --strict_exact

#Test

python script/dump_retriever_candidates.py \
  --config src/runpod.yaml \
  --checkpoint /workspace/protein-go-align-outputs-P3a-P2cwarm-segGO-segProjOnly/checkpoint_step83945.pt \
  --split val \
  --ids_path /workspace/data/splits/test_ids.txt \
  --pid2pos_path /workspace/data/pid_to_positives_canonical.json \
  --topk 1000 \
  --batch_size 4 \
  --num_workers 0 \
  --out_dir /workspace/candidate_dumps/P3a_test_top1000 \
  --strict_exact
'''

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.main import (
    load_structured_cfg,
    set_seed,
    build_go_cache,
    build_stores,
    build_datasets,
    enforce_cache_alignment,
)

from src.training.collate import ContrastiveEmbCollator
from src.training.trainer import OppTrainer

from src.configs.data_classes import (
    FewZeroConfig,
    TrainerConfig,
    AttrConfig,
    QueueConfig,
    TrainingContext,
    LoggingConfig,
    LoRAParameters,
)

from src.encoders import BioMedBERTEncoder
from src.go import load_go_parents, load_go_children
from src.datasets.go_text_store import GoTextStore
from src.utils import (
    load_raw_json,
    load_raw_pickle,
    load_go_set,
    load_go_texts_by_phase,
)
from src.utils.helpers import (
    build_altid_map_from_go_terms,
    canonicalize_id_list,
    canonicalize_pid2pos,
    go_str_to_int_any,
    load_go_namespaces,
)


class DummyWandbRun:
    def __init__(self):
        self.summary = {}

    def log(self, *args, **kwargs):
        return None

    def finish(self):
        return None


def setup_logging_simple() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _looks_like_state_dict(d: Any) -> bool:
    if not isinstance(d, dict):
        return False

    n_tensor = 0
    for v in d.values():
        if torch.is_tensor(v):
            n_tensor += 1
            if n_tensor >= 5:
                return True
    return False


def _unwrap_state_dict(blob: Dict[str, Any], name: str = "model") -> Dict[str, torch.Tensor]:
    """
    Handles:
      state_dict
      {"model": state_dict}
      {"model": {"model": state_dict}}
      {"state_dict": state_dict}
      {"model_state_dict": state_dict}
      {"module": state_dict}
    """
    if not isinstance(blob, dict):
        raise RuntimeError(f"[dump] {name} blob is not dict: {type(blob)}")

    state = blob
    unwrap_keys = ["model", "state_dict", "model_state_dict", "module", "net"]

    for depth in range(10):
        if _looks_like_state_dict(state):
            print(f"[dump] unwrapped {name} state_dict at depth={depth}")
            return state

        if not isinstance(state, dict):
            raise RuntimeError(f"[dump] {name} became non-dict at depth={depth}: {type(state)}")

        found = False
        for key in unwrap_keys:
            if key in state and isinstance(state[key], dict):
                print(f"[dump] unwrap {name} depth={depth}: state = state['{key}']")
                state = state[key]
                found = True
                break

        if not found:
            raise RuntimeError(
                f"[dump] Could not unwrap {name}. Current keys={list(state.keys())[:30]}"
            )

    raise RuntimeError(f"[dump] Exceeded max unwrap depth for {name}")


def _clean_state_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}

    for k, v in state.items():
        if not torch.is_tensor(v):
            continue

        kk = k
        changed = True
        while changed:
            changed = False
            for pref in ("module.", "model.", "trainer.model."):
                if kk.startswith(pref):
                    kk = kk[len(pref):]
                    changed = True

        out[kk] = v

    return out


def load_model_weights_only(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    strict_exact: bool = True,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    print(f"[dump] loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"[dump] checkpoint is not dict: {type(ckpt)}")

    if "model" not in ckpt:
        raise RuntimeError(f"[dump] checkpoint has no 'model' key. keys={list(ckpt.keys())}")

    print("[dump] checkpoint top-level keys:", list(ckpt.keys()))

    state = _unwrap_state_dict(ckpt["model"], name="model")
    state = _clean_state_keys(state)

    current = model.state_dict()

    loadable = {}
    skipped_missing = []
    skipped_shape = []

    for k, v in state.items():
        if k not in current:
            skipped_missing.append(k)
            continue

        if tuple(current[k].shape) != tuple(v.shape):
            skipped_shape.append((k, tuple(v.shape), tuple(current[k].shape)))
            continue

        loadable[k] = v

    if len(loadable) == 0:
        print("[dump][DEBUG] state sample:", list(state.keys())[:50])
        print("[dump][DEBUG] current sample:", list(current.keys())[:50])
        raise RuntimeError("[dump] loaded 0 compatible keys")

    missing, unexpected = model.load_state_dict(loadable, strict=False)

    print(
        f"[dump] model loaded: compatible={len(loadable)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )

    if missing:
        print("[dump][WARN] missing example:", missing[:30])
    if unexpected:
        print("[dump][WARN] unexpected example:", unexpected[:30])
    if skipped_missing:
        print("[dump][WARN] skipped_missing example:", skipped_missing[:30])
    if skipped_shape:
        print("[dump][WARN] skipped_shape example:", skipped_shape[:10])

    if strict_exact:
        if len(missing) > 0 or len(unexpected) > 0:
            raise RuntimeError(
                f"[dump] exact checkpoint load expected, but got "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    return {
        "compatible": len(loadable),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "skipped_missing": len(skipped_missing),
        "skipped_shape": len(skipped_shape),
        "meta": ckpt.get("meta", {}),
    }


def build_go_encoder_and_text_store(args, device: torch.device):
    lora_params = LoRAParameters(adapter_name="go_encoder")

    go_encoder = BioMedBERTEncoder(
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        device=device,
        max_length=512,
        attention_pooling_strategy=args.go_encoder_inner_pooling,
        attn_hidden=128,
        attn_dropout=0.1,
        special_token_weights=None,
        enable_lora=args.use_lora,
        lora_parameters=lora_params,
        use_special_tokens=False,
    )

    is_segmented = args.go_encoder_output_mode == "segment_pooled"

    full_id2segments = None
    full_id2seg_present = None

    if is_segmented:
        id2text, id2segments, id2seg_present = load_go_texts_by_phase(
            args.go_text_folder,
            phase=args.phase,
            return_segments=True,
        )

        go_id_to_text = {int(args.phase): id2text}
        full_id2segments = {int(args.phase): id2segments}
        full_id2seg_present = {int(args.phase): id2seg_present}

    else:
        id2text = load_go_texts_by_phase(
            args.go_text_folder,
            phase=args.phase,
            return_segments=False,
        )

        go_id_to_text = {int(args.phase): id2text}

    go_text_store = GoTextStore(
        go_id_to_text,
        go_encoder.tokenizer,
        phase=args.phase,
        lazy=True,
        max_len=args.go_text_store_max_len,
        is_segmented=is_segmented,
        segment_max_len=args.go_segment_max_len,
        full_id2segments=full_id2segments,
        full_id2seg_present=full_id2seg_present,
    )

    return go_encoder, go_text_store


def canonicalize_and_align_inputs(args, go_cache, logger):
    if args.eval_space == "seen":
        eval_id_list = load_raw_pickle(args.go_path_seen)
    else:
        eval_id_list = load_raw_pickle(args.go_path_observed)

    eval_seen_go_ids = load_raw_pickle(args.go_path_seen)
    eval_unseen_ids = load_raw_pickle(args.zero_shot_path)
    eval_rare_go_ids = load_raw_pickle(args.few_shot_path)

    go_terms = load_raw_json(args.go_basic_json)
    alt_map = build_altid_map_from_go_terms(go_terms) if go_terms else {}
    logger.info("[canon] alt_id map size = %d", len(alt_map))

    pid2pos_raw = load_raw_json(args.pid2pos)
    pid2pos = canonicalize_pid2pos(pid2pos_raw, alt_map) if alt_map else pid2pos_raw

    zs = load_go_set(args.zero_shot_path)
    fs = load_go_set(args.few_shot_path)

    zs = canonicalize_id_list(list(zs), alt_map) if alt_map else zs
    fs = canonicalize_id_list(list(fs), alt_map) if alt_map else fs

    eval_id_list = (
        canonicalize_id_list(eval_id_list, alt_map)
        if alt_map
        else [go_str_to_int_any(x) for x in eval_id_list]
    )
    eval_seen_go_ids = (
        canonicalize_id_list(eval_seen_go_ids, alt_map)
        if alt_map
        else [go_str_to_int_any(x) for x in eval_seen_go_ids]
    )
    eval_unseen_ids = (
        canonicalize_id_list(eval_unseen_ids, alt_map)
        if alt_map
        else [go_str_to_int_any(x) for x in eval_unseen_ids]
    )
    eval_rare_go_ids = (
        canonicalize_id_list(eval_rare_go_ids, alt_map)
        if alt_map
        else [go_str_to_int_any(x) for x in eval_rare_go_ids]
    )

    pid2pos, eval_id_list, eval_seen_go_ids, eval_unseen_ids, eval_rare_go_ids = enforce_cache_alignment(
        go_cache=go_cache,
        pid2pos=pid2pos,
        eval_id_list=eval_id_list,
        eval_seen_go_ids=eval_seen_go_ids,
        eval_unseen_ids=eval_unseen_ids,
        eval_rare_go_ids=eval_rare_go_ids,
        logger=logger,
        drop_empty_proteins=False,
    )

    return {
        "pid2pos": pid2pos,
        "zs": zs,
        "fs": fs,
        "eval_id_list": eval_id_list,
        "eval_seen_go_ids": eval_seen_go_ids,
        "eval_unseen_ids": eval_unseen_ids,
        "eval_rare_go_ids": eval_rare_go_ids,
    }


def make_dump_loader(dataset, args, go_text_store, batch_size: int, num_workers: int):
    zs_mask_np = getattr(dataset, "zs_mask", None)
    zs_mask_vec = torch.as_tensor(zs_mask_np, dtype=torch.bool) if zs_mask_np is not None else None

    collate = ContrastiveEmbCollator(
        go_text_store=go_text_store,
        zs_mask_vec=zs_mask_vec,
        bidirectional=True,
        neg_k=args.neg_k,
        go_dropout=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )

    return loader


def build_trainer_for_dump(args, device: torch.device, go_cache, go_encoder, go_text_store, datasets, aligned, dag_parents, dag_children):
    with torch.no_grad():
        sample_item = datasets["train"][0]

    d_h = int(sample_item["prot_emb"].shape[1])
    d_g = int(go_cache.embs.shape[1])
    d_z = int(getattr(args, "align_dim", d_g))

    training_context = TrainingContext(
        device=device,
        go_cache=go_cache,
        faiss_index=None,
        vres=None,
        current_phase=None,
        last_refresh_epoch=None,
        last_refresh_reason=None,
        batch_builder=None,
        maybe_refresh_phase_resources=None,
        dag_parents=dag_parents,
        dag_children=dag_children,
        go_namespace_map=load_go_namespaces(),
        scheduler=None,
        go_text_store=go_text_store,
        use_queue_miner=False,
        attribute_loss_enabled=False,
        return_alpha=False,
        return_slot_attn=False,
        fp16_enabled=args.fp16,
        protein_pooling_strategy=args.protein_pooling_strategy,
        eval_id_list=aligned["eval_id_list"],
        logger=logging.getLogger("dump"),
        eval_seen_go_ids=aligned["eval_seen_go_ids"],
        eval_unseen_ids=aligned["eval_unseen_ids"],
        eval_rare_go_ids=aligned["eval_rare_go_ids"],
        protein_n_slots=args.protein_n_slots,
        go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,
    )

    training_context.run_name = "candidate-dump"
    training_context.logging = LoggingConfig(
        log_every=int(args.log_every),
        log_lora_hist=False,
        probe_eval_every=0,
        probe_batch_size=8,
        gospec_tau=0.02,
        gospec_topk=32,
    )

    trainer_cfg = TrainerConfig(
        d_h=d_h,
        d_g=d_g,
        d_z=d_z,
        device=str(device),
        lr=args.lr,
        lr_lora=args.lr_lora,
        max_epochs=args.epochs,
        cand_chunk_k=args.cand_chunk_k,
        pos_chunk_t=args.pos_chunk_t,
        is_logit_scale_constant=bool(args.is_logit_scale_constant),
        go_pooling=args.go_pooling,
        eval_go_bs=args.eval_go_bs,
        max_inbatch=args.max_inbatch,
        eval_cand_chunk_k=args.eval_cand_chunk_k,
        go_segment_alpha_warmup_steps=args.go_segment_alpha_warmup_steps,
        go_segment_alpha=args.go_segment_alpha,
        trainable_mode=args.trainable_mode,
    )

    attr_cfg = AttrConfig(
        lambda_attr=getattr(args, "lambda_attr", 0.0),
        lambda_entropy_alpha=getattr(args, "lambda_entropy_alpha", 0.0),
        lambda_entropy_window=getattr(args, "lambda_entropy_window", 0.0),
        topk_per_window=int(getattr(args, "topk_per_window", 64)),
        curriculum_epochs=int(getattr(args, "curriculum_epochs", 10)),
        temperature=float(getattr(args, "temperature", 0.07)),
        lambda_vtrue=getattr(args, "lambda_vtrue", 0.0),
        tau_distill=getattr(args, "tau_distill", 1.0),
        lambda_dag=getattr(args, "lambda_dag", 0.0),
        lambda_bce=getattr(args, "lambda_bce", 0.0),
    )

    queue_cfg = QueueConfig(
        queue_K=args.queue_K,
        queue_start_step=args.queue_start_step,
        queue_hard_frac_start=args.queue_hard_frac_start,
        queue_hard_frac_end=args.queue_hard_frac_end,
        queue_hard_frac_warmup_steps=args.queue_hard_frac_warmup_steps,
        queue_weight_start=args.queue_weight_start,
        queue_weight_end=args.queue_weight_end,
        queue_weight_warmup_steps=args.queue_weight_warmup_steps,
        k_hard_queue_start=args.k_hard_queue_start,
        k_hard_queue_end=args.k_hard_queue_end,
        k_hard_queue_warmup_steps=args.k_hard_queue_warmup_steps,
    )

    trainer = OppTrainer(
        cfg=trainer_cfg,
        attr=attr_cfg,
        ctx=training_context,
        queue_cfg=queue_cfg,
        go_encoder=go_encoder,
        wandb_run=DummyWandbRun(),
    )

    trainer.eval_id_list = aligned["eval_id_list"]
    return trainer


@torch.no_grad()
def compute_go_projected_bank(trainer: OppTrainer, chunk: int = 2048) -> np.ndarray:
    """
    Returns projected normalized GO embeddings aligned with trainer._eval_ids_cpu.
    Shape: [G, Dz], dtype float16.
    """
    device = trainer.device

    if trainer._eval_G_once_cpu is None:
        raise RuntimeError("eval GO cache is not ready. Call _ensure_eval_cache_v2 first.")

    G_cpu = trainer._eval_G_once_cpu.float()
    out_chunks = []

    model = trainer.model
    model.eval()

    for s in tqdm(range(0, G_cpu.size(0), chunk), desc="project GO bank"):
        e = min(G_cpu.size(0), s + chunk)

        G = G_cpu[s:e].to(device, non_blocking=True)
        Gz = model.go_ln(G)
        Gz = model.proj_g(Gz)
        Gz = F.normalize(Gz.float(), dim=-1)

        out_chunks.append(Gz.detach().cpu().half().numpy())

    return np.concatenate(out_chunks, axis=0)


def pad_true_ids(true_ids: List[List[int]], pad_value: int = -1) -> np.ndarray:
    max_len = max((len(x) for x in true_ids), default=0)
    arr = np.full((len(true_ids), max_len), pad_value, dtype=np.int64)

    for i, xs in enumerate(true_ids):
        if xs:
            arr[i, : len(xs)] = np.asarray(xs, dtype=np.int64)

    return arr


@torch.no_grad()
def dump_candidates(
    trainer: OppTrainer,
    loader: DataLoader,
    out_dir: Path,
    *,
    checkpoint_path: str,
    config_path: str,
    split_name: str,
    topk: int,
    go_project_chunk: int,
    empty_cache_every: int = 50,
    overwrite: bool = False,
    save_top_go_ids: bool = False,
):
    out_dir = Path(out_dir)

    done_path = out_dir / "DONE"
    metadata_path = out_dir / "metadata.json"

    if out_dir.exists():
        if done_path.exists() and not overwrite:
            raise RuntimeError(
                f"[dump] Output dir already has DONE marker: {out_dir}. "
                "Use --overwrite or choose a new out_dir."
            )

        if not overwrite and any(out_dir.iterdir()):
            raise RuntimeError(
                f"[dump] Output dir exists and is not empty: {out_dir}. "
                "Use --overwrite or choose a new out_dir."
            )

        if overwrite:
            import shutil
            logging.warning("[dump] removing existing output dir: %s", str(out_dir))
            shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = trainer.device
    trainer.model.eval()

    # Important for segment_pooled checkpoints.
    # In training, this may be scheduled inside step_losses.
    # During dump, step_losses is never called, so set it explicitly.
    if getattr(trainer.ctx, "go_encoder_output_mode", None) == "segment_pooled":
        alpha = float(getattr(trainer.cfg, "go_segment_alpha", 0.05))

        if hasattr(trainer.model, "go_segment_mix_alpha"):
            trainer.model.go_segment_mix_alpha = alpha
            print("[dump] set go_segment_mix_alpha =", trainer.model.go_segment_mix_alpha)

        if hasattr(trainer.model, "go_segment_alpha"):
            trainer.model.go_segment_alpha = alpha
            print("[dump] set go_segment_alpha =", trainer.model.go_segment_alpha)

    # Write initial metadata as incomplete.
    initial_metadata = {
        "status": "incomplete",
        "split": split_name,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "topk_requested": int(topk),
        "save_top_go_ids": bool(save_top_go_ids),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(initial_metadata, f, indent=2)

    logging.info("[dump] refreshing eval GO cache")
    trainer._refresh_eval_go_cache(chunk=trainer.cfg.eval_go_bs)
    trainer._eval_cache_ready = False
    trainer._ensure_eval_cache_v2(chunk=trainer.cfg.eval_go_bs)

    eval_go_ids = trainer._eval_ids_cpu.numpy().astype(np.int64)
    n_go = int(eval_go_ids.shape[0])
    k_eff = min(int(topk), n_go)

    logging.info("[dump] projecting GO bank")
    go_z = compute_go_projected_bank(trainer, chunk=go_project_chunk)
    dz = int(go_z.shape[1])

    np.save(out_dir / "eval_go_ids.npy", eval_go_ids)
    np.save(out_dir / "go_z.float16.npy", go_z)

    n_samples = len(loader.dataset)
    logging.info(
        "[dump] n_samples=%d n_go=%d topk=%d dz=%d",
        n_samples,
        n_go,
        k_eff,
        dz,
    )

    top_cols_mm = np.lib.format.open_memmap(
        out_dir / "top_go_cols.int32.npy",
        mode="w+",
        dtype=np.int32,
        shape=(n_samples, k_eff),
    )

    top_scores_mm = np.lib.format.open_memmap(
        out_dir / "top_scores.float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, k_eff),
    )

    top_labels_mm = np.lib.format.open_memmap(
        out_dir / "top_labels.int8.npy",
        mode="w+",
        dtype=np.int8,
        shape=(n_samples, k_eff),
    )

    protein_z_mm = np.lib.format.open_memmap(
        out_dir / "protein_z.float16.npy",
        mode="w+",
        dtype=np.float16,
        shape=(n_samples, dz),
    )

    top_ids_mm = None
    if save_top_go_ids:
        top_ids_mm = np.lib.format.open_memmap(
            out_dir / "top_go_ids.int64.npy",
            mode="w+",
            dtype=np.int64,
            shape=(n_samples, k_eff),
        )

    protein_ids: List[str] = []
    true_ids_all: List[List[int]] = []

    scale = trainer.logit_scale_tensor()

    offset = 0

    for step, batch in enumerate(tqdm(loader, desc=f"dump {split_name}")):
        H = batch["prot_emb_pad"].to(device, non_blocking=True)

        if trainer.to_f32 is not None:
            H = trainer.to_f32(H)

        attn_valid, _ = trainer._valid_and_pad_masks(batch)

        G_eval, y_true = trainer._build_eval_space(batch)

        scores_raw = trainer.forward_scores(
            H,
            G_eval,
            attn_valid,
            return_alpha=False,
        )
        scores_rank = (scores_raw * scale).float()

        vals, idxs = torch.topk(
            scores_rank,
            k=k_eff,
            dim=1,
        )

        B = int(H.size(0))
        s = offset
        e = offset + B

        idxs_cpu = idxs.detach().cpu().to(torch.int64)
        vals_cpu = vals.detach().cpu().float()

        labels = torch.gather(
            (y_true > 0).to(torch.int8),
            1,
            idxs,
        ).detach().cpu()

        # Frozen retriever protein query embedding.
        Zp = trainer.model.encode_protein_for_scoring(H, attn_valid)

        if isinstance(Zp, tuple):
            Zp = Zp[0]

        if Zp.dim() == 3:
            Zp = Zp.mean(dim=1)

        Zp = F.normalize(Zp.float(), dim=-1)

        top_cols_np = idxs_cpu.numpy().astype(np.int32)
        top_cols_mm[s:e, :] = top_cols_np
        top_scores_mm[s:e, :] = vals_cpu.numpy().astype(np.float32)
        top_labels_mm[s:e, :] = labels.numpy().astype(np.int8)
        protein_z_mm[s:e, :] = Zp.detach().cpu().half().numpy()

        if top_ids_mm is not None:
            top_ids = trainer._eval_ids_cpu.index_select(
                0,
                idxs_cpu.reshape(-1),
            ).view(B, k_eff)
            top_ids_mm[s:e, :] = top_ids.numpy().astype(np.int64)

        pids = batch.get("protein_ids", None)
        if pids is None:
            pids = [f"{split_name}_{i}" for i in range(s, e)]

        protein_ids.extend([str(x) for x in pids])

        for gids in batch["pos_go_global"]:
            true_ids_all.append([int(x) for x in gids.detach().cpu().tolist()])

        offset = e

        if empty_cache_every > 0 and (step + 1) % empty_cache_every == 0:
            top_cols_mm.flush()
            top_scores_mm.flush()
            top_labels_mm.flush()
            protein_z_mm.flush()
            if top_ids_mm is not None:
                top_ids_mm.flush()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if offset != n_samples:
        raise RuntimeError(f"[dump] wrote offset={offset}, expected n_samples={n_samples}")

    # Flush large memmaps.
    top_cols_mm.flush()
    top_scores_mm.flush()
    top_labels_mm.flush()
    protein_z_mm.flush()
    if top_ids_mm is not None:
        top_ids_mm.flush()

    true_padded = pad_true_ids(true_ids_all, pad_value=-1)
    np.save(out_dir / "true_go_ids.npy", true_padded)

    with open(out_dir / "protein_ids.json", "w", encoding="utf-8") as f:
        json.dump(protein_ids, f)

    with open(out_dir / "true_go_ids.json", "w", encoding="utf-8") as f:
        json.dump(true_ids_all, f)

    files = {
        "protein_ids": "protein_ids.json",
        "true_go_ids_json": "true_go_ids.json",
        "true_go_ids_padded": "true_go_ids.npy",
        "eval_go_ids": "eval_go_ids.npy",
        "go_z": "go_z.float16.npy",
        "protein_z": "protein_z.float16.npy",
        "top_go_cols": "top_go_cols.int32.npy",
        "top_scores": "top_scores.float32.npy",
        "top_labels": "top_labels.int8.npy",
    }

    if save_top_go_ids:
        files["top_go_ids"] = "top_go_ids.int64.npy"

    metadata = {
        "status": "complete",
        "split": split_name,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "n_samples": int(n_samples),
        "n_go": int(n_go),
        "topk": int(k_eff),
        "dz": int(dz),
        "save_top_go_ids": bool(save_top_go_ids),
        "files": files,
        "schema": {
            "row_alignment": "All row-based files align by protein_ids order.",
            "top_go_cols": "Column indices into eval_go_ids and go_z.",
            "top_scores": "Post-logit-scale retriever scores.",
            "top_labels": "1 if candidate is a true GO label for that protein, else 0.",
            "protein_z": "Frozen retriever protein query embedding.",
            "go_z": "Frozen retriever projected GO embedding bank aligned with eval_go_ids.",
            "true_go_ids": "All true GO ids per protein, padded with -1. Needed for full-space Fmax.",
            "top_go_ids": "Optional. Can be reconstructed as eval_go_ids[top_go_cols].",
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(done_path, "w", encoding="utf-8") as f:
        f.write("complete\n")

    logging.info("[dump] done: %s", str(out_dir))


def parse_args():
    p = argparse.ArgumentParser("Dump frozen retriever top-K GO candidates for reranker.")

    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--ids_path", type=str, default=None, help="Optional override for val_ids_path. Use this for test.")
    p.add_argument("--pid2pos_path", type=str, default=None, help="Optional override for pid2pos path.")

    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--go_project_chunk", type=int, default=2048)
    p.add_argument("--strict_exact", action="store_true", help="Require missing=0 and unexpected=0 when loading checkpoint.")
    p.add_argument("--empty_cache_every", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_top_go_ids", action="store_true")

    return p.parse_args()


def main():
    cli = parse_args()
    setup_logging_simple()

    args = load_structured_cfg(cli.config)

    if cli.device is not None:
        args.general_device = cli.device

    if cli.ids_path is not None:
        # For test split, pass --split val --ids_path /path/to/test_ids.txt
        args.val_ids_path = Path(cli.ids_path)

    if cli.pid2pos_path is not None:
        args.pid2pos = Path(cli.pid2pos_path)

    # This script loads the checkpoint explicitly.
    # Avoid accidental warmstart/resume side effects.
    args.resume = None
    args.warmstart_path = None
    args.eval_only = True
    args.wandb = False

    set_seed(args.seed)

    device = torch.device(args.general_device if args.general_device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    logging.info("[dump] device=%s", str(device))

    go_cache = build_go_cache(str(args.go_cache_path))

    dag_parents = load_go_parents() if args.use_dag_in_ds else None
    dag_children = load_go_children() if args.use_dag_in_ds else None

    go_encoder, go_text_store = build_go_encoder_and_text_store(args, device)

    logging.info("[dump] materializing GO text tokens")
    go_text_store.materialize_tokens_once(batch_size=512, show_progress=True)

    aligned = canonicalize_and_align_inputs(
        args=args,
        go_cache=go_cache,
        logger=logging.getLogger("dump"),
    )

    res_store = build_stores(args)

    datasets = build_datasets(
        args,
        res_store,
        go_text_store,
        dag_parents=dag_parents,
        pid2pos=aligned["pid2pos"],
        zs=aligned["zs"],
        fs=aligned["fs"],
    )

    if cli.split == "train":
        dataset = datasets["train"]
        split_name = "train"
    else:
        dataset = datasets["val"]
        split_name = "val" if cli.ids_path is None else "custom"

    batch_size = int(cli.batch_size or args.eval_batch_size or args.batch_size)

    loader = make_dump_loader(
        dataset=dataset,
        args=args,
        go_text_store=go_text_store,
        batch_size=batch_size,
        num_workers=int(cli.num_workers),
    )

    trainer = build_trainer_for_dump(
        args=args,
        device=device,
        go_cache=go_cache,
        go_encoder=go_encoder,
        go_text_store=go_text_store,
        datasets=datasets,
        aligned=aligned,
        dag_parents=dag_parents,
        dag_children=dag_children,
    )

    load_info = load_model_weights_only(
        trainer.model,
        cli.checkpoint,
        device=device,
        strict_exact=bool(cli.strict_exact),
    )

    logging.info("[dump] checkpoint meta: %s", load_info.get("meta", {}))

    dump_candidates(
        trainer=trainer,
        loader=loader,
        out_dir=Path(cli.out_dir),
        checkpoint_path=cli.checkpoint,
        config_path=cli.config,
        split_name=split_name,
        topk=int(cli.topk),
        go_project_chunk=int(cli.go_project_chunk),
        empty_cache_every=int(cli.empty_cache_every),
        overwrite=bool(cli.overwrite),
        save_top_go_ids=bool(cli.save_top_go_ids),
    )


if __name__ == "__main__":
    main()