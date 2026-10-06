from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import shutil
import signal
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from math import inf
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import types

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
import yaml
from torch.utils.data import DataLoader

from src.configs.data_classes import FewZeroConfig, TrainerConfig, TrainingContext, LoggingConfig
from src.configs.paths import YAML_FILE
from src.datasets import ESMResidueStore, GoTextStore
from src.datasets.protein_dataset import ProteinEmbDataset
from src.encoders import BioMedBERTEncoder
from src.go import GoLookupCache
from src.training.collate import ContrastiveEmbCollator
from src.training.trainer import OppTrainer
from src.training.wandb_helper import RetrieverWandbLogger
from src.utils import load_raw_json, load_raw_txt
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.helpers import (
    _coerce_id2row, _coerce_int_list, _coerce_row2id_list_from_dict,
    build_altid_map_from_go_terms, canonicalize_id_list,
    canonicalize_pid2pos, go_str_to_int_any, load_go_texts_canonical,
)

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _sigint_handler(signum, frame):
    print(f"\\n[DBG] Caught SIGINT at {time.strftime('%H:%M:%S')}")
    traceback.print_stack(frame)


signal.signal(signal.SIGINT, _sigint_handler)


def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.info("Logging initialized. Log file: %s", log_path)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def cleanup_old_checkpoints(output_dir: Path, keep_last_n: int = 3) -> None:
    checkpoints = sorted(output_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    for path in checkpoints[:-keep_last_n] if keep_last_n is not None else []:
        try:
            path.unlink()
        except OSError:
            pass


# =============================================================================
# PFresGO helpers, now native to main.py
# =============================================================================

def _go_int(value: Any) -> int:
    text = str(value).strip()
    return int(text.split(":", 1)[1] if text.upper().startswith("GO:") else text)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_ids(path: Path) -> List[int]:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    elif suffix == ".json":
        value = _read_json(path)
        if isinstance(value, dict):
            value = value.get("ids", value.get("go_ids", value.get("terms", value)))
    else:
        value = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if isinstance(value, dict):
        value = list(value.keys())
    return sorted({_go_int(x) for x in value})


def _active_go_terms(go_vocab_path: Path) -> Dict[str, dict]:
    raw = _read_json(go_vocab_path)
    return {
        str(go_id): info for go_id, info in raw.items()
        if isinstance(info, dict) and not bool(info.get("is_obsolete", False))
    }


def _build_dag(go_vocab_path: Path) -> Tuple[dict, dict]:
    parents, children = {}, {}
    active = _active_go_terms(go_vocab_path)
    active_ids = {_go_int(g) for g in active}
    for go_id, info in active.items():
        child = _go_int(go_id)
        edges = []
        for raw in info.get("is_a", []) or []:
            parent = _go_int(raw)
            if parent in active_ids:
                edges.append((parent, "is_a"))
        for raw in info.get("part_of", []) or []:
            parent = _go_int(raw)
            if parent in active_ids:
                edges.append((parent, "part_of"))
        parents[child] = edges
        for parent, relation in edges:
            children.setdefault(parent, []).append((child, relation))
    for go_id in active_ids:
        parents.setdefault(go_id, [])
        children.setdefault(go_id, [])
    return parents, children


def _namespace_map(go_vocab_path: Path) -> Dict[int, str]:
    return {_go_int(g): str(info.get("namespace", "")) for g, info in _active_go_terms(go_vocab_path).items()}


def _training_buckets(train_ids_path: Path, pid2pos_path: Path, branch_ids: Iterable[int], rare_lt: int):
    train_ids = {x.strip() for x in train_ids_path.read_text(encoding="utf-8").splitlines() if x.strip()}
    pid2pos = _read_json(pid2pos_path)
    branch = set(map(int, branch_ids))
    counts = Counter()
    benchmark = set()
    for pid, labels in pid2pos.items():
        labels_i = {_go_int(g) for g in labels} & branch
        benchmark.update(labels_i)
        if pid in train_ids:
            counts.update(labels_i)
    seen = sorted(counts)
    rare = sorted(g for g, c in counts.items() if c < rare_lt)
    zero = sorted(branch - set(seen))
    return seen, rare, zero, sorted(benchmark)


# =============================================================================
# Config
# =============================================================================

def load_structured_cfg(path: str):
    cfg_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    pf = raw.get("pfresgo", {})
    data = raw.get("data", {})
    general = raw.get("general", {})
    training = raw.get("training", {})
    optim = raw.get("optim", {})
    model = raw.get("model", {})
    loss = raw.get("loss", {})
    evaluation = raw.get("evaluation", {})
    diagnostics = raw.get("diagnostics", {})
    wandb_cfg = raw.get("wandb", {})
    stores = raw.get("stores", {})

    allowed_segments = ["name", "namespace", "definition", "is_a", "part_of"]
    requested = [str(x).strip() for x in pf.get("enabled_segments", ["name", "definition"]) if str(x).strip()]
    unknown = set(requested) - set(allowed_segments)
    if unknown:
        raise ValueError(f"Unknown PFresGO segments: {sorted(unknown)}")
    enabled_segments = [x for x in allowed_segments if x in set(requested)]
    if not enabled_segments:
        raise ValueError("At least one GO segment must be enabled.")

    branch = str(pf.get("branch", "")).strip().upper()
    if branch not in {"BP", "MF", "CC"}:
        raise ValueError("pfresgo.branch must be BP, MF, or CC")

    protocol = str(pf.get("benchmark_protocol", "standard")).strip().lower()
    if protocol not in {"standard", "zeroshot"}:
        raise ValueError("pfresgo.benchmark_protocol must be standard or zeroshot")

    branch_ids_path = Path(pf["branch_go_ids_path"]).expanduser().resolve()
    go_text_path = Path(pf["go_text_path"]).expanduser().resolve()

    def opt_path(v):
        return Path(v).expanduser().resolve() if v else None

    args = types.SimpleNamespace(
        config_path=cfg_path,
        pfresgo_branch=branch,
        pfresgo_benchmark_protocol=protocol,
        enabled_segments=enabled_segments,
        branch_go_ids_path=branch_ids_path,
        go_text_path=go_text_path,
        evaluation_space=str(pf.get("evaluation_space", "benchmark")).strip().lower(),
        evaluation_split=str(pf.get("evaluation_split", "valid")).strip().lower(),
        test_ids_path=opt_path(pf.get("test_ids_path")),
        rare_lt=int(pf.get("rare_lt", 20)),

        overlap=int(data.get("overlap", 256)),
        max_len=int(data.get("max_len", 1024)),
        use_dag_in_ds=bool(data.get("use_dag_in_ds", False)),
        fs_target_ratio=float(data.get("fs_target_ratio", 0.0)),

        phase=int(general.get("phase", -2)),
        ablation_id=general.get("ablation_id"),
        go_pooling_strategy=str(general.get("go_pooling_strategy", "mean")),
        go_encoder_inner_pooling=str(general.get("go_encoder_inner_pooling", "mean")),
        go_pool_type=str(general.get("go_pool_type", "mean")),
        go_encoder_output_mode=str(general.get("go_encoder_output_mode", "segment_pooled")),
        go_segment_representation_mode=str(general.get("go_segment_representation_mode", "segments_only")),
        protein_pooling_strategy=str(general.get("protein_pooling_strategy", "mean_attn_gate")),
        use_lora=bool(general.get("use_lora", False)),

        epochs=int(training.get("epochs", 15)),
        batch_size=int(training.get("batch_size", 4)),
        eval_batch_size=int(training.get("eval_batch_size", training.get("batch_size", 4))),
        num_workers=int(training.get("num_workers", 0)),
        fp16=bool(training.get("fp16", True)),
        cpu=bool(training.get("cpu", False)),
        seed=int(training.get("seed", 42)),
        output_dir=str(training.get("output_dir", "outputs/retriever_v2")),
        save_every=int(training.get("save_every", 10000)),
        keep_last_n=int(training.get("keep_last_n", 3)),
        resume=training.get("resume"),
        warmstart_path=training.get("warmstart_path"),
        eval_only=bool(training.get("eval_only", False)),
        log_every=int(training.get("log_every", 500)),
        log_level=str(training.get("log_level", "INFO")),
        monitor_metric=str(training.get("monitor_metric", "oracle_microF@500")),
        secondary_monitor_metric=str(training.get("secondary_monitor_metric", "macro_term_recall@500")),
        monitor_mode=str(training.get("monitor_mode", "max")),
        early_stop_patience=int(training.get("early_stop_patience", 3)),
        eval_go_bs=int(training.get("eval_go_bs", 256)),
        go_text_store_max_len=int(training.get("go_text_store_max_len", 128)),
        go_segment_max_len=int(training.get("go_segment_max_len", 64)),
        go_pooling=str(training.get("go_pooling", "none")),
        general_device=str(training.get("device", "cuda:0")),

        lr=float(optim.get("lr", 1e-5)),
        weight_decay=float(optim.get("weight_decay", 0.0)),
        grad_clip=float(optim.get("grad_clip", 1.0)),

        align_dim=int(model.get("align_dim", 768)),
        temperature=float(model.get("temperature", 0.07)),
        attn_heads=int(model.get("attn_heads", 2)),
        attn_dropout=float(model.get("attn_dropout", 0.1)),

        lambda_con=float(loss.get("lambda_con", 1.0)),
        pbr_lambda=float(loss.get("pbr_lambda", 0.0)),
        pbr_margin=float(loss.get("pbr_margin", 0.05)),
        pbr_tau=float(loss.get("pbr_tau", 0.05)),

        retrieval_eval_ks=tuple(int(x) for x in evaluation.get("retrieval_eval_ks", [50, 100, 200, 500, 1000])),
        positive_rank_quantiles=tuple(float(x) for x in evaluation.get("positive_rank_quantiles", [0.50, 0.75, 0.90, 0.95])),
        log_gradient_norms=bool(diagnostics.get("log_gradient_norms", True)),
        log_positive_rank_cdf=bool(diagnostics.get("log_positive_rank_cdf", True)),
        log_cardinality_metrics=bool(diagnostics.get("log_cardinality_metrics", True)),
        log_go_segment_weights=bool(diagnostics.get("log_go_segment_weights", True)),

        wandb=bool(wandb_cfg.get("enabled", False)),
        wandb_project=str(wandb_cfg.get("project", "protein-go-align-pfresgo")),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_run_name=wandb_cfg.get("wandb_run_name"),
        wandb_mode=str(wandb_cfg.get("mode", "online")),

        train_ids_path=opt_path(stores.get("train_ids_path")),
        val_ids_path=opt_path(stores.get("val_ids_path")),
        go_basic_json=opt_path(stores.get("go_basic_json")),
        pid2pos=opt_path(stores.get("pid2pos_path")),
        zero_shot_path=opt_path(stores.get("zero_shot_path")),
        few_shot_path=opt_path(stores.get("few_shot_path")),
        embed_dir_res=opt_path(stores.get("embed_dir_res")),
        go_text_folder=opt_path(stores.get("go_text_folder")),
        go_cache_path=opt_path(stores.get("go_cache_path")),
        seq_len_lookup=opt_path(stores.get("seq_len_lookup")),
    )

    if args.use_lora:
        raise ValueError("Retriever v2 requires use_lora=false; GO text encoder is frozen.")

    required = [
        args.train_ids_path, args.val_ids_path, args.go_basic_json, args.pid2pos,
        args.embed_dir_res, args.go_cache_path, args.go_text_path, args.branch_go_ids_path,
    ]
    missing = [str(x) for x in required if x is None or not Path(x).exists()]
    if missing:
        raise FileNotFoundError("Missing required resources: " + ", ".join(missing))

    if args.evaluation_split == "test":
        if args.test_ids_path is None:
            raise ValueError("test_ids_path is required for evaluation_split=test")
        args.val_ids_path = args.test_ids_path
    elif args.evaluation_split != "valid":
        raise ValueError("evaluation_split must be valid or test")

    if args.evaluation_space not in {"benchmark", "full_branch"}:
        raise ValueError("evaluation_space must be benchmark or full_branch")

    return args


# =============================================================================
# Stores / data
# =============================================================================

def build_go_cache(go_cache_path: str) -> GoLookupCache:
    path = Path(go_cache_path)
    logger = logging.getLogger("build_go_cache")
    logger.info("Loading GO cache: %s", path)

    memmap_path = path if path.suffix.lower() == ".npy" and path.exists() else None
    if memmap_path is None:
        for candidate in (path.with_suffix(".npy"), path.parent / "go_text_embeddings.npy"):
            if candidate.exists():
                memmap_path = candidate
                break

    if memmap_path is None:
        return GoLookupCache(torch.load(str(path), map_location="cpu", weights_only=False), device="cpu")

    id2row = None
    row2id = None
    for fname in ("id2row.json", "row2id.json", "ids.json", "ids.txt"):
        sidecar = memmap_path.with_name(fname)
        if not sidecar.exists():
            continue
        if sidecar.suffix == ".json":
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"Unexpected JSON format in {sidecar}")
            if "id2row" in data and id2row is None:
                id2row = _coerce_id2row(data["id2row"])
            if "row2id" in data and row2id is None:
                v = data["row2id"]
                row2id = _coerce_row2id_list_from_dict(v) if isinstance(v, dict) else _coerce_int_list(v)
            if "ids" in data and (row2id is None or id2row is None):
                row2id = _coerce_int_list(data["ids"])
                id2row = {int(g): i for i, g in enumerate(row2id)}
        else:
            row2id = _coerce_int_list([x.strip() for x in sidecar.read_text().splitlines() if x.strip()])
            id2row = {int(g): i for i, g in enumerate(row2id)}

    arr = np.load(memmap_path, mmap_mode="r", allow_pickle=False)
    if row2id is None:
        row2id = list(range(int(arr.shape[0])))
    if id2row is None:
        id2row = {int(g): i for i, g in enumerate(row2id)}
    if len(row2id) != int(arr.shape[0]):
        raise RuntimeError("GO cache mapping length does not match embedding rows.")
    if int((np.linalg.norm(arr.astype(np.float32), axis=1) < 1e-8).sum()) > 0:
        raise RuntimeError("GO cache contains zero embedding rows.")

    return GoLookupCache(
        {"memmap_path": str(memmap_path), "id2row": id2row, "row2id": row2id},
        device="cpu",
    )


def build_residue_store(args):
    return ESMResidueStore(
        embed_dir=args.embed_dir_res,
        max_len=args.max_len,
        overlap=args.overlap,
        prefer_fp16=args.fp16,
    )


def build_go_text_store(args, go_encoder):
    id2text, id2segments, id2seg_present = load_go_texts_canonical(
        str(args.go_text_path),
        phase=args.phase,
        return_segments=True,
        enabled_segments=args.enabled_segments,
    )
    return GoTextStore(
        {int(args.phase): id2text},
        go_encoder.tokenizer,
        phase=args.phase,
        lazy=True,
        max_len=args.go_text_store_max_len,
        is_segmented=True,
        segment_max_len=args.go_segment_max_len,
        full_id2segments={int(args.phase): id2segments},
        full_id2seg_present={int(args.phase): id2seg_present},
    )


def align_to_cache(go_cache, pid2pos, *id_lists):
    cache_ids = set(map(int, go_cache.id2row.keys()))
    aligned_pid2pos = {
        pid: sorted({int(g) for g in gos if int(g) in cache_ids})
        for pid, gos in pid2pos.items()
    }
    aligned_lists = [sorted({int(g) for g in xs if int(g) in cache_ids}) for xs in id_lists]
    return (aligned_pid2pos, *aligned_lists)


def build_datasets(args, residue_store, go_text_store, pid2pos, dag_parents, zero, rare):
    train_ids = [pid for pid in load_raw_txt(args.train_ids_path) if residue_store.has(pid)]
    val_ids = [pid for pid in load_raw_txt(args.val_ids_path) if residue_store.has(pid)]

    fewzero = FewZeroConfig(
        zero_shot_terms=set(zero),
        few_shot_terms=set(rare),
        fs_target_ratio=args.fs_target_ratio,
    )

    common = dict(
        pid2pos=pid2pos,
        go_text_store=go_text_store,
        fewzero=fewzero,
        dag_parents=dag_parents,
        store=residue_store,
    )
    train_ds = ProteinEmbDataset(protein_ids=train_ids, **common)
    val_ds = ProteinEmbDataset(protein_ids=val_ids, **common)
    logging.getLogger("data").info("Datasets ready. Train=%d Val=%d", len(train_ds), len(val_ds))
    return {"train": train_ds, "val": val_ds}


def build_dataloaders(datasets, args):
    collate = ContrastiveEmbCollator()
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True, collate_fn=collate, drop_last=False,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=args.eval_batch_size, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=False,
        collate_fn=collate, drop_last=False,
    )
    return train_loader, val_loader


def validate_active_universe(pid2pos, active_go_ids, datasets):
    active = set(map(int, active_go_ids))
    missing = set()
    for ds in datasets.values():
        for pid in getattr(ds, "pids", getattr(ds, "protein_ids", [])):
            missing.update(int(g) for g in pid2pos.get(pid, []) if int(g) not in active)
    if missing:
        raise RuntimeError(
            f"{len(missing)} dataset positive GO ids are outside the active training universe. "
            f"Examples: {sorted(missing)[:20]}"
        )


# =============================================================================
# Runner
# =============================================================================

def run_training(args):
    logger = logging.getLogger("main")
    device = torch.device("cpu" if args.cpu else args.general_device)
    logger.info("Device: %s", device)

    branch_ids = _read_ids(args.branch_go_ids_path)
    seen, rare, zero, benchmark_ids = _training_buckets(
        args.train_ids_path, args.pid2pos, branch_ids, args.rare_lt
    )
    active_go_ids = benchmark_ids if args.evaluation_space == "benchmark" else branch_ids

    logger.info(
        "[PFresGO] branch=%s split=%s protocol=%s evaluation_space=%s "
        "full_branch_terms=%d benchmark_terms=%d active_terms=%d segments=%s",
        args.pfresgo_branch, args.evaluation_split, args.pfresgo_benchmark_protocol,
        args.evaluation_space, len(branch_ids), len(benchmark_ids), len(active_go_ids),
        args.enabled_segments,
    )

    go_cache = build_go_cache(str(args.go_cache_path))
    go_terms = load_raw_json(args.go_basic_json)
    alt_map = build_altid_map_from_go_terms(go_terms) if go_terms else {}

    pid2pos_raw = load_raw_json(args.pid2pos)
    pid2pos = canonicalize_pid2pos(pid2pos_raw, alt_map) if alt_map else pid2pos_raw
    active_go_ids = canonicalize_id_list(active_go_ids, alt_map) if alt_map else list(map(int, active_go_ids))
    seen = canonicalize_id_list(seen, alt_map) if alt_map else seen
    rare = canonicalize_id_list(rare, alt_map) if alt_map else rare
    zero = canonicalize_id_list(zero, alt_map) if alt_map else zero

    pid2pos, active_go_ids, seen, zero, rare = align_to_cache(
        go_cache, pid2pos, active_go_ids, seen, zero, rare
    )
    logger.info("[GO universe] active=%d seen=%d rare=%d zero=%d", len(active_go_ids), len(seen), len(rare), len(zero))

    dag_parents = (
        _build_dag(args.go_basic_json)[0]
        if args.use_dag_in_ds
        else None
    )

    go_encoder = BioMedBERTEncoder(
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        device=device,
        max_length=512,
        attention_pooling_strategy=args.go_encoder_inner_pooling,
        attn_hidden=128,
        attn_dropout=0.1,
        gradient_checkpointing=False,
    )

    # Retriever v2 invariant: text encoder is frozen.
    for p in go_encoder.parameters():
        p.requires_grad = False
    go_encoder.eval()

    go_text_store = build_go_text_store(args, go_encoder)
    go_text_store.materialize_tokens_once(batch_size=512, show_progress=True)

    residue_store = build_residue_store(args)
    datasets = build_datasets(
        args, residue_store, go_text_store, pid2pos, dag_parents, zero, rare
    )
    validate_active_universe(pid2pos, active_go_ids, datasets)
    train_loader, val_loader = build_dataloaders(datasets, args)

    sample = datasets["train"][0]
    d_h = int(sample["prot_emb"].shape[1])
    d_g = int(go_cache.embs.shape[1])
    d_z = int(args.align_dim)

    context = TrainingContext(
        go_cache=go_cache,
        device=device,
        go_text_store=go_text_store,
        run_name=args.wandb_run_name or f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        fp16_enabled=args.fp16,
        protein_pooling_strategy=args.protein_pooling_strategy,
        go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,
        go_segment_representation_mode=args.go_segment_representation_mode,
        eval_id_list=list(active_go_ids),
        eval_seen_go_ids=list(seen),
        eval_unseen_ids=list(zero),
        eval_rare_go_ids=list(rare),
        logger=logger,
        logging=LoggingConfig(log_every=args.log_every),
    )

    trainer_cfg = TrainerConfig(
        d_h=d_h, d_g=d_g, d_z=d_z, device=str(device),
        lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        max_epochs=args.epochs, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, fp16=args.fp16,
        warmstart_path=args.warmstart_path,
        monitor_metric=args.monitor_metric,
        secondary_monitor_metric=args.secondary_monitor_metric,
        monitor_mode=args.monitor_mode,
        protein_pooling_strategy=args.protein_pooling_strategy,
        use_lora=False,
        go_pooling=args.go_pooling,
        go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,
        go_segment_representation_mode=args.go_segment_representation_mode,
        eval_go_bs=args.eval_go_bs,
        temperature=args.temperature,
        lambda_con=args.lambda_con,
        pbr_lambda=args.pbr_lambda,
        pbr_margin=args.pbr_margin,
        pbr_tau=args.pbr_tau,
        retrieval_eval_ks=args.retrieval_eval_ks,
        positive_rank_quantiles=args.positive_rank_quantiles,
        log_gradient_norms=args.log_gradient_norms,
        log_positive_rank_cdf=args.log_positive_rank_cdf,
        log_cardinality_metrics=args.log_cardinality_metrics,
        log_go_segment_weights=args.log_go_segment_weights,
    )

    run = None
    if args.wandb:
        wandb.login()
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=context.run_name,
            config={**context.to_dict(), **vars(args)},
            mode=args.wandb_mode,
            settings=wandb.Settings(code_dir=".", _disable_stats=True),
            reinit=False,
        )
        context.wandb_run = run

    # Retriever v2 trainer: full-GO InfoNCE + PBR, no legacy candidate machinery.
    trainer = OppTrainer(
        cfg=trainer_cfg,
        ctx=context,
        go_encoder=go_encoder,
        wandb_run=None,
    )

    wb = RetrieverWandbLogger(
        run=run,
        cfg=trainer_cfg,
        go_ids=active_go_ids,
        segment_names=getattr(trainer, "_segment_names", []),
    )

    wb.log_run_metadata(
        model=trainer.model,
        active_go_ids=active_go_ids,
        protein_pooling=args.protein_pooling_strategy,
        go_segments=args.enabled_segments,
        objective="full_go_infonce+pbr",
    )

    # Dataset-level metadata remains orchestration/data information.
    if run is not None:
        counts = np.asarray(
            [
                len(datasets["train"].pid2pos.get(pid, []))
                for pid in datasets["train"].pids
            ],
            dtype=np.int64,
        )
        if counts.size:
            run.log(
                {
                    "data/positives_per_protein": wandb.Histogram(counts),
                    "trainer_step": 0,
                },
                step=0,
            )
            run.summary["data/positives_per_protein_mean"] = float(
                counts.mean()
            )
            run.summary["data/positives_per_protein_p95"] = float(
                np.percentile(counts, 95)
            )

    out_dir = Path(args.output_dir)
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch = load_checkpoint(trainer, str(args.resume), map_location=trainer.device)
        global_step = int(getattr(trainer, "_global_step", 0))

    if args.eval_only:
        with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bool(args.fp16 and device.type == "cuda"),
        ):
            val_logs = trainer.eval_epoch(val_loader, epoch_idx=0)
        logger.info(
            "[eval_only] %s",
            " | ".join(f"{k}:{v:.4f}" for k, v in val_logs.items()),
        )
        wb.log_validation(
            metrics=val_logs,
            diagnostics=getattr(trainer, "last_eval_diagnostics", None),
            step=int(getattr(trainer, "_global_step", 0)),
            epoch=0,
        )
        wb.finalize()
        return

    best_val = -inf if args.monitor_mode == "max" else inf
    best_macro = -inf
    no_improve_epochs = 0
    eps = 1e-6

    if run is not None:
        run.define_metric("trainer_step")
        run.define_metric("*", step_metric="trainer_step")

    use_bf16 = bool(args.fp16 and device.type == "cuda")
    logger.info(
        "Start Retriever v2 training for %d epochs | autocast=%s",
        args.epochs,
        "bf16" if use_bf16 else "off",
    )

    for epoch in range(start_epoch, args.epochs):
        trainer.model.train()
        # Frozen text encoder must remain eval even when the alignment model trains.
        trainer.model.go_encoder.eval()

        running = {"total": 0.0, "contrastive": 0.0, "pbr": 0.0}
        n_batches = 0

        for batch in train_loader:
            with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
            ):
                losses = trainer.step_losses(batch, epoch)
                loss = losses["total"]

            trainer.opt.zero_grad(set_to_none=True)
            loss.backward()

            # Gradients exist here and are still unclipped.
            # This is the correct point for optimization diagnostics.
            if args.log_gradient_norms:
                wb.log_gradients(
                    model=trainer.model,
                    step=int(getattr(trainer, "_global_step", global_step + 1)),
                )

            if args.grad_clip > 0:
                params = [p for p in trainer.model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            trainer.opt.step()
            n_batches += 1

            for key in running:
                if key in losses and losses[key] is not None:
                    value = losses[key]
                    running[key] += float(value.item() if hasattr(value, "item") else value)

            global_step = int(getattr(trainer, "_global_step", global_step + 1))

            if global_step % max(1, args.log_every) == 0:
                avg = {k: v / max(1, n_batches) for k, v in running.items()}
                lr0 = trainer.opt.param_groups[0].get("lr")
                logger.info(
                    "[train] epoch %d step %d :: %s | lr:%.2e",
                    epoch, global_step,
                    " | ".join(f"{k}:{v:.4f}" for k, v in avg.items()),
                    lr0,
                )
                # Use averaged core losses for the interval, while retaining
                # current-batch diagnostics such as score gap and PBR ranks.
                wb_losses = dict(losses)
                wb_losses["total"] = avg["total"]
                wb_losses["contrastive"] = avg["contrastive"]
                wb_losses["pbr"] = avg["pbr"]

                wb.log_train(
                    losses=wb_losses,
                    lr=lr0,
                    step=global_step,
                )

            if global_step % max(1, args.save_every) == 0:
                save_checkpoint(
                    out_dir=str(out_dir), tag=f"step{global_step}",
                    trainer=trainer, args=args, epoch=epoch, step=global_step,
                )
                cleanup_old_checkpoints(out_dir, args.keep_last_n)

        with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
        ):
            val_logs = trainer.eval_epoch(val_loader, epoch)
        logger.info(
            "[val] epoch %d :: %s",
            epoch,
            " | ".join(f"{k}:{v:.4f}" for k, v in val_logs.items()),
        )

        wb.log_validation(
            metrics=val_logs,
            diagnostics=getattr(trainer, "last_eval_diagnostics", None),
            step=global_step,
            epoch=epoch,
        )

        metric_name = (
            args.monitor_metric
            if args.monitor_metric in val_logs
            else "align_MRR"
        )
        score = float(val_logs[metric_name])
        improved = score > best_val + eps if args.monitor_mode == "max" else score < best_val - eps

        macro_name = args.secondary_monitor_metric
        macro_score = float(val_logs.get(macro_name, -inf))
        macro_improved = macro_score > best_macro + eps

        if improved:
            best_val = score
            ckpt = save_checkpoint(
                out_dir=str(out_dir), tag=f"step{global_step}",
                trainer=trainer, args=args, epoch=epoch, step=global_step,
            )
            shutil.copy2(ckpt, out_dir / "best_primary.pt")
            if run is not None:
                run.summary[f"best/{metric_name}"] = best_val
                run.summary["best/epoch"] = epoch
                run.summary["best/step"] = global_step

        if macro_improved:
            best_macro = macro_score
            ckpt = save_checkpoint(
                out_dir=str(out_dir), tag=f"macro_step{global_step}",
                trainer=trainer, args=args, epoch=epoch, step=global_step,
            )
            shutil.copy2(ckpt, out_dir / "best_macro.pt")
            if run is not None:
                run.summary[f"best/{macro_name}"] = best_macro

        if improved or macro_improved:
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                logger.info("[early-stop] no improvement for %d epochs", no_improve_epochs)
                break

        save_checkpoint(
            out_dir=str(out_dir), tag=f"epoch{epoch + 1}",
            trainer=trainer, args=args, epoch=epoch, step=global_step,
        )
        cleanup_old_checkpoints(out_dir, args.keep_last_n)

    save_checkpoint(
        out_dir=str(out_dir), tag="final",
        trainer=trainer, args=args, epoch=epoch, step=global_step,
    )
    logger.info("Training finished. Artifacts saved under: %s", args.output_dir)

    wb.finalize(
        best_metrics={
            args.monitor_metric: best_val,
            args.secondary_monitor_metric: best_macro,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description="PFresGO Retriever v2 training/evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    cli = parser.parse_args()

    config_path = cli.config or str(YAML_FILE)
    args = load_structured_cfg(config_path)
    if cli.device is not None:
        args.general_device = cli.device
    return args


def main():
    args = parse_args()
    setup_logging(Path(args.output_dir), level=args.log_level)
    set_seed(args.seed)
    t0 = time.time()
    try:
        run_training(args)
    except Exception as exc:
        logging.exception("Fatal error: %r", exc)
        raise
    finally:
        logging.info("Total runtime: %.1f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()