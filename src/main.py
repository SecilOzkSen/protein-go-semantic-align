import copy
import os
import sys
import time
import math
import yaml
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, List, Set, Iterable, Tuple, Any, Optional
import torch.multiprocessing as mp
from datetime import datetime
import glob
import types
import signal, traceback
from pathlib import Path
import logging
import json
import numpy as np
import torch
import argparse

import wandb
from src.datasets import ESMResidueStore, GoTextStore
from src.datasets.protein_dataset import ProteinEmbDataset
from src.training.collate import ContrastiveEmbCollator
from src.training.trainer import OppTrainer
from src.utils.helpers import _coerce_int_list, _coerce_id2row, _coerce_row2id_list_from_dict, build_altid_map_from_go_terms, canonicalize_id_list, canonicalize_pid2pos, go_str_to_int_any, load_go_namespaces
from src.configs.data_classes import (
    FewZeroConfig, TrainerConfig, AttrConfig, LoRAParameters, TrainingContext, LoggingConfig, QueueConfig
)
from src.go import GoLookupCache, GoDropoutConfig, GoTokenDropout
from src.go import load_go_parents, load_go_children
from src.utils import (
    load_go_set, load_raw_json, load_raw_txt, load_go_texts_by_phase, load_raw_pickle
)
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.encoders import BioMedBERTEncoder
from math import inf
from src.configs.paths import YAML_FILE

# To prevent sigint (potential cause)
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA check ->", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device 0 name ->", torch.cuda.get_device_name(0))

# ============== Utilities ==============

def _collect_fused_ids(fused_dir: str) -> set:
    fused_dir = Path(fused_dir)
    have = set()
    for p in sorted(fused_dir.glob("fused_esm1b_*.ids.txt")):
        with p.open("r") as fh:
            for line in fh:
                pid = line.strip()
                if pid:
                    have.add(pid)
    return have

def _sigint_handler(signum, frame):
    print(f"\n[DBG] Caught SIGINT at {time.strftime('%H:%M:%S')}")
    traceback.print_stack(frame)
signal.signal(signal.SIGINT, _sigint_handler)


def setup_logging(output_dir: Path, level: str = "INFO"):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8")
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.info("Logging initialized. Log file: %s", str(log_path))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def steps_per_epoch(n_items: int, batch_size: int) -> int:
    return max(1, math.ceil(n_items / max(1, batch_size)))


def _ckpt_path(output_dir: Path, tag: str):
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return output_dir / f"ckpt_{tag}_{ts}.pt"

def cleanup_old_checkpoints(output_dir: Path, keep_last_n: int = 3):
    cks = sorted(glob.glob(str(output_dir / "ckpt_*.pt")))
    if keep_last_n is not None and len(cks) > keep_last_n:
        for p in cks[:-keep_last_n]:
            try:
                os.remove(p)
                logging.getLogger("ckpt").info("Removed old checkpoint: %s", p)
            except OSError:
                pass

def collect_eval_ids_from_datasets(train_ds, val_ds=None) -> List[int]:
    """
    Train ve (varsa) val dataset'teki tüm GO id'lerinin birleşimini çıkarır.
    Eval uzayı olarak bunu kullanacağız.
    """
    ids: Set[int] = set()

    for gids in train_ds.pid2pos.values():
        for g in gids:
            ids.add(int(g))

    if val_ds is not None:
        for gids in val_ds.pid2pos.values():
            for g in gids:
                ids.add(int(g))

    eval_ids = sorted(ids)
    return eval_ids



# ============== Builders ==============

def build_go_cache(go_cache_path: str) -> GoLookupCache:
    logger = logging.getLogger("build_go_cache")
    p = Path(go_cache_path)
    logger.info("Loading GO cache: %s", str(p))

    memmap_path = None
    if p.suffix.lower() == ".npy" and p.exists():
        memmap_path = p
    else:
        cand = p.with_suffix(".npy")
        if cand.exists():
            memmap_path = cand
        else:
            alt = p.parent / "go_text_embeddings.npy"
            if alt.exists():
                memmap_path = alt

    if memmap_path is not None:
        id2row: Optional[Dict[int, int]] = None
        row2id: Optional[List[int]] = None

        # 1) sidecar mapping files
        for fname in ("id2row.json", "row2id.json", "ids.json", "ids.txt"):
            f = memmap_path.with_name(fname)
            if not f.exists():
                continue

            if f.suffix == ".json":
                with open(f, "r") as fp:
                    data = json.load(fp)

                if isinstance(data, dict):
                    if "id2row" in data and id2row is None:
                        id2row = _coerce_id2row(data["id2row"])
                    if "row2id" in data and row2id is None:
                        v = data["row2id"]
                        if isinstance(v, dict):
                            row2id = _coerce_row2id_list_from_dict(v)
                        else:
                            row2id = _coerce_int_list(v)
                    if "ids" in data and (row2id is None or id2row is None):
                        ids = _coerce_int_list(data["ids"])
                        row2id = ids
                        id2row = {int(g): i for i, g in enumerate(ids)}
                else:
                    raise ValueError(f"Unexpected JSON format in {f}")

            else:
                # ids.txt (one id per line)
                with open(f, "r") as fp:
                    ids = [line.strip() for line in fp if line.strip()]
                ids = _coerce_int_list(ids)
                row2id = ids
                id2row = {int(g): i for i, g in enumerate(ids)}

        # 2) fallback: if still missing, build identity mapping 0..n-1 based on memmap shape
        if row2id is None:
            arr0 = np.load(memmap_path, mmap_mode="r", allow_pickle=False)
            n = int(arr0.shape[0])
            row2id = list(range(n))

        if id2row is None:
            id2row = {int(g): i for i, g in enumerate(row2id)}

        # 3) sanity: shape must match mapping length
        arr = np.load(memmap_path, mmap_mode="r", allow_pickle=False)  # [N,D]
        if int(arr.shape[0]) != len(row2id):
            raise RuntimeError(f"GO cache rows={int(arr.shape[0])} != len(row2id)={len(row2id)} for {memmap_path}")

        # 4) sanity: reject zero rows
        norms = np.linalg.norm(arr.astype(np.float32), axis=1)
        n_zero = int((norms < 1e-8).sum())
        if n_zero > 0:
            raise RuntimeError(
                f"GO cache has {n_zero} zero-embedding rows in {memmap_path}. "
                "Rebuild GO embeddings."
            )

        blob = {
            "memmap_path": str(memmap_path),
            "id2row": id2row,   # int->int
            "row2id": row2id,   # LIST[int], not dict
        }
        return GoLookupCache(blob, device="cpu")  # or your desired device

    # non-memmap blob
    blob = torch.load(str(p), map_location="cpu", weights_only=False)
    return GoLookupCache(blob, device="cpu")

def build_stores(args):
    logger = logging.getLogger("build_stores")
    logger.info("Building residue+fused stores (lazy, no snapshot)...")

    # 3) embed dirs
    embed_dir_res = getattr(args, "embed_dir_res", None)
    # 4) toggles

    logger.info(
        "Store config:\n"
        f"  embed_dir_res   = {embed_dir_res}\n"
        f"  prefer_fp16     = {args.fp16}\n"
        f"  max_len/overlap = {getattr(args,'max_len',None)}/{getattr(args,'overlap',None)}"
    )

    # 5) Build residue
    res_store = ESMResidueStore(
        embed_dir=embed_dir_res,
        max_len=args.max_len,
        overlap=args.overlap,
        prefer_fp16=args.fp16,
    )

    logger.info("Stores ready.")
    return res_store

def build_val_dataset(
    val_pids,
    pid2pos_val,
    go_text_store,
    fewzero_cfg,
    dag_parents,
    residue_store: ESMResidueStore,
):
    ds_val = ProteinEmbDataset(
        protein_ids=val_pids,
        pid2pos=pid2pos_val,
        go_text_store=go_text_store,
        fewzero=fewzero_cfg,
        dag_parents=dag_parents,
        store=residue_store,
    )
    return ds_val

def build_datasets(args, res_store: ESMResidueStore, go_text_store: GoTextStore, dag_parents=None, pid2pos = None, zs=None, fs=None) -> Dict[str, torch.utils.data.Dataset]:
    logger = logging.getLogger("build_datasets")
    logger.info("Building datasets...")
    logger.info(f"PID_TO_POSITIVES path = {args.pid2pos}")

    if pid2pos == None:
        pid2pos = load_raw_json(args.pid2pos)

    train_ids = load_raw_txt(args.train_ids_path)
    val_ids = load_raw_txt(args.val_ids_path)

    train_ids = [pid for pid in train_ids if res_store.has(pid)]
    val_ids = [pid for pid in val_ids if res_store.has(pid)]

    logging.getLogger("data").info(
        "[residue-filter] train=%d val=%d (filtered by _pid2span)",
        len(train_ids), len(val_ids)
    )
    # TODO: erase
    have = set(res_store._pid2span.keys())
    dropped_train = [pid for pid in load_raw_txt(args.train_ids_path) if pid not in have]
    dropped_val = [pid for pid in load_raw_txt(args.val_ids_path) if pid not in have]
    logging.getLogger("data").warning(
        "[residue-filter] dropped train=%d val=%d ex_train=%s ex_val=%s",
        len(dropped_train), len(dropped_val), dropped_train[:5], dropped_val[:5]
    )

    if zs is None:
        zs = load_go_set(args.zero_shot_path)

    if fs is None:
        fs = load_go_set(args.few_shot_path)

    fz = FewZeroConfig(zero_shot_terms=zs, few_shot_terms=fs,
                       fs_target_ratio=args.fs_target_ratio)

    train_ds = ProteinEmbDataset(
        protein_ids=train_ids,
        pid2pos=pid2pos,
        go_text_store=go_text_store,
        fewzero=fz,
        dag_parents=dag_parents,
        store=res_store,
    )

    val_ds = build_val_dataset(val_pids=val_ids, pid2pos_val=pid2pos, go_text_store=go_text_store, fewzero_cfg=fz,
                              dag_parents=dag_parents, residue_store=res_store)

    logger.info("Datasets ready. Train=%d%s", len(train_ds), f", Val={len(val_ds)}" if val_ds else "")
    return {"train": train_ds, "val": val_ds}

def build_dataset_sample_weights(dataset):
    weights = []
    for i in range(len(dataset)):
        sample = dataset[i]
        n_rare = int(sample.get("num_rare_go", 0))
        w = 1.0 + np.log1p(n_rare)
        weights.append(w)
    return torch.as_tensor(weights, dtype=torch.double)

def build_dataloaders(datasets, args, go_text_store: GoTextStore, go_dropout:GoTokenDropout=None):
    logger = logging.getLogger("build_dataloaders")

    train_ds = datasets["train"]
    zs_mask_np = getattr(train_ds, "zs_mask", None)
    zs_mask_vec = torch.as_tensor(zs_mask_np, dtype=torch.bool) if zs_mask_np is not None else None

    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    shuffled = False
    shuffled_text_store = None
    if args.ablation_id == "text_shuffle":
        print("Go text store shuffling for training...")
        shuffled = True
        shuffled_text_store = copy.deepcopy(go_text_store)
        shuffled_text_store.shuffle()

    train_collate = ContrastiveEmbCollator(
        go_text_store=shuffled_text_store if shuffled else go_text_store, #tokenizer
        zs_mask_vec=zs_mask_vec,
        bidirectional=True,
        neg_k=args.neg_k,
        go_dropout=go_dropout
    )
    val_collate = ContrastiveEmbCollator(
        go_text_store=go_text_store,  # tokenizer
        zs_mask_vec=zs_mask_vec,
        bidirectional=True,
        neg_k=args.neg_k,
        go_dropout=None # dropout off
    )
#    train_weights = build_dataset_sample_weights(datasets["train"])
#    train_sampler = WeightedRandomSampler(
#        weights=train_weights,
#        num_samples=len(train_weights),
#        replacement=True,
#    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True,
        collate_fn=train_collate,
    )

    val_loader = None
    if datasets.get("val") is not None:
        val_loader = DataLoader(
            datasets["val"],
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            collate_fn=val_collate,
#            multiprocessing_context="forkserver",
            drop_last=False
        )

    logger.info("Dataloaders ready. batch_size=%d", args.batch_size)
    return train_loader, val_loader

def refresh_go_cache_chunked(ids_to_update, go_text_store, go_encoder, go_cache, device, chunk_size=256):
    ids_to_update = list(ids_to_update)
    go_encoder.eval()
    out_chunks = []

    with torch.no_grad():
        for s in range(0, len(ids_to_update), chunk_size):
            chunk_ids = ids_to_update[s:s + chunk_size]
            toks = go_text_store.batch(chunk_ids)

            input_ids = toks["input_ids"].to(device, non_blocking=True)
            attn = toks["attention_mask"].to(device, non_blocking=True)

            embs = go_encoder(input_ids=input_ids, attention_mask=attn)  # [C, D]

            # IMPORTANT: cache update için CPU float32 sabitle
            embs = embs.detach().float().cpu().contiguous()
            out_chunks.append(embs)

            del toks, input_ids, attn, embs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    new_embs = torch.cat(out_chunks, dim=0).contiguous()  # CPU [N, D]
    go_cache.update(ids_to_update, new_embs)


def wandb_preview_curriculum(wandb_mod, args, total_steps: int):
    mode = args.curriculum_mode
    T = max(1, total_steps - 1)
    steps = np.linspace(0, T, num=min(200, total_steps), dtype=int)

    def interp(a, b, t, T, mode_="cosine"):
        if mode_ == "linear":
            x = t / max(1, T)
            return a + (b - a) * x
        x = 0.5 * (1 - math.cos(math.pi * t / max(1, T)))
        return a + (b - a) * x

    curves = {
        "hard_frac": [interp(args.hard_frac_start, args.hard_frac_end, s, T, mode) for s in steps],
        "shortlist_M": [interp(args.shortlist_M_start, args.shortlist_M_end, s, T, mode) for s in steps],
        "k_hard": [interp(args.k_hard_start, args.k_hard_end, s, T, mode) for s in steps],
        "inbatch_easy": [interp(args.inbatch_easy_start, args.inbatch_easy_end, s, T, mode) for s in steps],
        "random_k": [interp(args.random_k_start, args.random_k_end, s, T, mode) for s in steps],
        "hier_up": [interp(args.hier_up_start, args.hier_up_end, s, T, mode) for s in steps],
        "hier_dn": [interp(args.hier_dn_start, args.hier_dn_end, s, T, mode) for s in steps],
    }

    # lambda_attr schedule: attr curriculum’ünü gör
    lambda_attr_curve = []
    lambda_attr_start_epoch = getattr(args, "lambda_attr_start", None)
    lambda_attr_max = getattr(args, "lambda_attr_max", getattr(args, "lambda_attr", 0.0))
    if lambda_attr_start_epoch is not None:
        steps_per_epoch = max(1, T // max(1, args.curriculum_epochs))
        start_step = lambda_attr_start_epoch * steps_per_epoch
        for s in steps:
            if s < start_step:
                lambda_attr_curve.append(0.0)
            else:
                # simple ramp to max
                frac = min(1.0, (s - start_step) / max(1, T - start_step))
                lambda_attr_curve.append(float(lambda_attr_max) * frac)
        curves["lambda_attr"] = lambda_attr_curve

    # warmup flag
    warmup_steps = int(args.warmup_frac * T)
    warmup_flags = [1 if s <= warmup_steps else 0 for s in steps]

    table = wandb_mod.Table(columns=["step"] + list(curves.keys()) + ["is_warmup"])
    for i, s in enumerate(steps):
        row = [int(s)] + [curves[k][i] for k in curves.keys()] + [warmup_flags[i]]
        table.add_data(*row)
    wandb_mod.log({"curriculum/preview": table})

    # hızlı özet, W&B summary'ye de yaz
    try:
        wandb_mod.summary["curriculum/total_steps"] = int(total_steps)
        wandb_mod.summary["curriculum/warmup_steps"] = int(warmup_steps)
        wandb_mod.summary["curriculum/hard_frac_start"] = float(args.hard_frac_start)
        wandb_mod.summary["curriculum/hard_frac_end"] = float(args.hard_frac_end)
        if lambda_attr_curve:
            wandb_mod.summary["curriculum/lambda_attr_max"] = float(lambda_attr_max)
    except Exception:
        pass



from collections import defaultdict

def wandb_dataset_quickstats(wandb_mod, train_ds, sample_n: int = 512):
    try:
        # 1) pozitif sayısı per protein
        pos_counts: List[int] = []
        if hasattr(train_ds, "protein_ids") and hasattr(train_ds, "pid2pos"):
            for pid in train_ds.protein_ids[:sample_n]:
                pos = train_ds.pid2pos.get(pid, [])
                pos_counts.append(len(pos))
        if pos_counts:
            arr = np.array(pos_counts)
            wandb_mod.log({"data/positives_per_protein": wandb_mod.Histogram(arr)})
            wandb_mod.summary["data/positives_per_protein_mean"] = float(arr.mean())
            wandb_mod.summary["data/positives_per_protein_p95"] = float(np.percentile(arr, 95))

        # 2) sequence length dağılımı
        lengths: List[int] = []
        for i in range(min(sample_n, len(train_ds))):
            try:
                item = train_ds[i]
                L = None
                if isinstance(item, dict):
                    for k in ["res_len", "length", "residue_len"]:
                        if k in item:
                            L = int(item[k]); break
                    if L is None:
                        for k in ["H", "residue_emb", "emb"]:
                            if k in item and hasattr(item[k], "shape"):
                                L = int(item[k].shape[0]); break
                elif isinstance(item, (list, tuple)) and len(item) > 0:
                    head = item[0]
                    if hasattr(head, "shape") and len(head.shape) >= 2:
                        L = int(head.shape[-2])
                if L is not None:
                    lengths.append(L)
            except Exception:
                break
        if lengths:
            arr = np.array(lengths)
            wandb_mod.log({"data/lengths": wandb_mod.Histogram(arr)})
            wandb_mod.summary["data/length_mean"] = float(arr.mean())
            wandb_mod.summary["data/length_p95"] = float(np.percentile(arr, 95))

        # 3) GO label frekansları
        go_counts = defaultdict(int)
        if hasattr(train_ds, "pid2pos"):
            for gids in train_ds.pid2pos.values():
                for g in gids:
                    go_counts[int(g)] += 1

        if go_counts:
            freq = np.array(list(go_counts.values()), dtype=np.int64)
            wandb_mod.log({"data/go_label_freq": wandb_mod.Histogram(freq)})
            wandb_mod.summary["data/go_label_freq_mean"] = float(freq.mean())
            wandb_mod.summary["data/go_label_freq_p95"] = float(np.percentile(freq, 95))
            wandb_mod.summary["data/n_unique_go_terms"] = int(len(go_counts))

            # en sık 20 GO term tablosu
            top_k = 20
            top_items = sorted(go_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            table = wandb_mod.Table(columns=["go_id", "count"])
            for gid, c in top_items:
                table.add_data(int(gid), int(c))
            wandb_mod.log({"data/top_go_terms": table})

        # 4) few-shot, zero-shot, common oranları
        fewzero = getattr(train_ds, "fewzero", None)
        if fewzero is not None:
            wandb_mod.summary["data/n_zero_shot_terms"] = int(len(fewzero.zero_shot_terms))
            wandb_mod.summary["data/n_few_shot_terms"] = int(len(fewzero.few_shot_terms))
            wandb_mod.summary["data/n_common_terms"] = int(len(fewzero.common_terms))

        if hasattr(train_ds, "is_fs"):
            is_fs_arr = np.asarray(train_ds.is_fs, dtype=np.float32)
            wandb_mod.summary["data/fs_protein_frac"] = float(is_fs_arr.mean())
            wandb_mod.log({"data/fs_protein_flags": wandb_mod.Histogram(is_fs_arr)})

    except Exception as e:
        logging.getLogger("wandb").warning("Dataset quickstats failed: %r", e)


def sanitize_dataset_with_go_text(train_ds, go_text_store) -> None:
    """
    ProteinEmbDataset'i GoTextStore domain'i ile hizalar.

    - train_ds.pid2pos içindeki GO id'lerini, go_text_store.id2tok domainine intersect eder
    - Buna paralel olarak pos_weights_map ve pos_is_generalized'ı da kırpar
    - Label'ı tamamen boşalan proteinleri dataset'ten atar
    - train_ds.pids ve train_ds.is_fs'i yeniden yazar
    """

    valid_ids: Set[int] = set(int(g) for g in go_text_store.id2text.keys())

    if len(valid_ids) == 0:
        raise RuntimeError(
            "GoTextStore has empty id2text domain. Phase/text loading is broken."
        )

    old_pids = list(train_ds.pids)
    new_pids: List[str] = []

    new_pid2pos: Dict[str, List[int]] = {}
    new_pos_weights: Dict[str, List[float]] = {}
    new_pos_is_gen: Dict[str, List[bool]] = {}
    new_is_fs: List[bool] = []

    dropped_prots = 0
    dropped_terms: Set[int] = set()
    dropped_labels = 0

    fewzero = getattr(train_ds, "fewzero", None)

    for pid in old_pids:
        gids = train_ds.pid2pos.get(pid, [])
        ws = train_ds.pos_weights_map.get(pid, [1.0] * len(gids))
        is_gen = train_ds.pos_is_generalized.get(pid, [False] * len(gids))

        if not gids:
            dropped_prots += 1
            continue

        gids_int = [int(g) for g in gids]

        kept_g: List[int] = []
        kept_w: List[float] = []
        kept_gen: List[bool] = []

        for g, w, ig in zip(gids_int, ws, is_gen):
            if g in valid_ids:
                kept_g.append(g)
                kept_w.append(float(w))
                kept_gen.append(bool(ig))
            else:
                dropped_terms.add(g)

        if not kept_g:
            dropped_prots += 1
            dropped_labels += len(gids_int)
            continue

        new_pids.append(pid)
        new_pid2pos[pid] = kept_g
        new_pos_weights[pid] = kept_w
        new_pos_is_gen[pid] = kept_gen

        if fewzero is not None:
            lbl_set = set(kept_g)
            new_is_fs.append(any((g in fewzero.few_shot_terms) for g in lbl_set))

    print(
        f"[sanitize_dataset_with_go_text] proteins: {len(old_pids)} -> {len(new_pids)}, "
        f"dropped_proteins={dropped_prots}, dropped_labels={dropped_labels}, "
        f"dropped_terms={len(dropped_terms)}"
    )
    if dropped_terms:
        sample = sorted(dropped_terms)[:10]
        print(f"[sanitize_dataset_with_go_text] example missing terms (no text): {sample}")

    # In-place update
    train_ds.pids = new_pids
    train_ds.pid2pos = new_pid2pos
    train_ds.pos_weights_map = new_pos_weights
    train_ds.pos_is_generalized = new_pos_is_gen

    if hasattr(train_ds, "protein_ids"):
        train_ds.protein_ids = new_pids
        print("[sanitize_dataset_with_go_text] Updated train_ds.protein_ids")
    if hasattr(train_ds, "ids"):
        train_ds.ids = new_pids

    if new_is_fs:
        train_ds.is_fs = new_is_fs


def sanity_check_go_text(train_ids, pid2pos, go_text_store):
    used_terms = set()
    for pid in train_ids:
        gids = pid2pos.get(pid, [])
        for g in gids:
            used_terms.add(int(g))

    missing = [g for g in used_terms if g not in go_text_store.id2text]
    if missing:
        raise RuntimeError(
            f"GoTextStore is missing {len(missing)} GO ids, e.g. {missing[:10]}"
        )

def enforce_cache_alignment(
    *,
    go_cache,
    pid2pos: Dict[str, List[int]],
    eval_id_list: List[int],
    eval_seen_go_ids: List[int],
    eval_unseen_ids: List[int],
    eval_rare_go_ids: List[int],
    logger=None,
    drop_empty_proteins: bool = False,
) -> Tuple[Dict[str, List[int]], List[int], List[int], List[int], List[int]]:
    """
    Drops any GO ids not present in go_cache.id2row from:
      - training labels (pid2pos)
      - eval spaces (eval_id_list, seen/unseen/rare)

    Returns updated (pid2pos, eval_id_list, eval_seen_go_ids, eval_unseen_ids, eval_rare_go_ids).

    Notes:
      - Use after canonicalization (alt_id -> primary).
      - Call BEFORE build_datasets.
    """
    cache_ids = set(int(x) for x in go_cache.id2row.keys())

    def _filter_list(xs: Iterable[Any]) -> List[int]:
        return sorted({int(x) for x in xs if int(x) in cache_ids})

    # ---- eval lists ----
    e0, s0, u0, r0 = len(eval_id_list), len(eval_seen_go_ids), len(eval_unseen_ids), len(eval_rare_go_ids)
    eval_id_list_f = _filter_list(eval_id_list)
    eval_seen_f = _filter_list(eval_seen_go_ids)
    eval_unseen_f = _filter_list(eval_unseen_ids)
    eval_rare_f = _filter_list(eval_rare_go_ids)

    # ---- pid2pos ----
    dropped_labels = 0
    dropped_proteins = 0
    new_pid2pos: Dict[str, List[int]] = {}

    for pid, gos in pid2pos.items():
        gos_i = [int(g) for g in (gos or [])]
        kept = [g for g in gos_i if g in cache_ids]
        dropped_labels += (len(gos_i) - len(kept))

        if drop_empty_proteins and len(kept) == 0:
            dropped_proteins += 1
            continue

        # keep deterministic + unique
        new_pid2pos[pid] = sorted(set(kept))

    msg = (
        f"[align-cache] eval: {e0}->{len(eval_id_list_f)} | "
        f"seen: {s0}->{len(eval_seen_f)} | unseen: {u0}->{len(eval_unseen_f)} | rare: {r0}->{len(eval_rare_f)} | "
        f"train_labels_dropped={dropped_labels} | proteins_dropped={dropped_proteins}"
    )
    if logger is not None:
        try:
            logger.info(msg)
        except Exception:
            print(msg)
    else:
        print(msg)

    return new_pid2pos, eval_id_list_f, eval_seen_f, eval_unseen_f, eval_rare_f


# ============== Runner ==============
def run_training(args):
    signal.signal(signal.SIGINT, _sigint_handler)
    logger = logging.getLogger("main")
    if args.general_device is not None:
        device = args.general_device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if args.wandb:
        wandb.login()


    print("[MAIN] No schedule provided, running in single-phase mode (phase0 = -1).")
    #go_cache_path = GO_INDEX[args.phase]["TEXT_EMB"]
    #go_cache_path = GO_INDEX[phase0+1]["TEXT_EMB"]
    go_cache_path = args.go_cache_path

    go_cache = build_go_cache(str(go_cache_path))
    dag_parents = load_go_parents() if args.use_dag_in_ds else None
    dag_children = load_go_children() if args.use_dag_in_ds else None

    # Text encoder (GO)
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

    # GO text dict per phase
    is_segmented = (args.go_encoder_output_mode == "segment_pooled")

    full_id2segments = None
    full_id2seg_present = None

    if is_segmented:
        # segment_pooled için return_segments kesin True olmalı
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

    # GoTextStore + dataloaders
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

    # TODO: Erase
    if is_segmented:
        # hard-code ids yerine store içinden gerçek id seçmek daha güvenli
        ids = list(go_text_store.id2text.keys())[:2]

        b = go_text_store.batch(ids)

        print("input_ids:", b["input_ids"].shape)
        print("attention_mask:", b["attention_mask"].shape)
        print("seg_input_ids:", b["seg_input_ids"].shape)
        print("seg_attention_mask:", b["seg_attention_mask"].shape)
        print("seg_present:", b["seg_present"].shape)
        print("segment_names:", b["segment_names"])
        print("seg_present:", b["seg_present"])

    print("GoTextStore text size:", len(go_text_store.id2text))
    print("GoTextStore token cache size:", len(go_text_store.id2tok))

    if go_text_store.is_go_segmented:
        print("GoTextStore segment text size:", len(go_text_store.id2segments))
        print("GoTextStore segment token cache size:", len(go_text_store.id2seg_tok))

    eval_space = getattr(args, "eval_space", "observed")
    if eval_space == "seen":
        eval_id_list = load_raw_pickle(args.go_path_seen)  # seen on training (at least one positive on training.)
    else:
        eval_id_list = load_raw_pickle(args.go_path_observed)  # observed: train+val+test positives

    eval_seen_go_ids = load_raw_pickle(args.go_path_seen)
    eval_unseen_ids = load_raw_pickle(args.zero_shot_path)
    eval_rare_go_ids = load_raw_pickle(args.few_shot_path)

    logging.getLogger("main").info(
        "[eval] using %d GO terms in eval_id_list", len(eval_id_list)
    )


    go_terms = load_raw_json(args.go_basic_json)

    alt_map = build_altid_map_from_go_terms(go_terms) if go_terms else {}
    logger.info("[canon] alt_id map size = %d", len(alt_map))

    # 2) pid2pos canonicalize
    pid2pos_raw = load_raw_json(args.pid2pos)  # sen zaten burada load ediyorsun
    pid2pos = canonicalize_pid2pos(pid2pos_raw, alt_map) if alt_map else pid2pos_raw
    logger.info("[canon] pid2pos canonicalized=%s", "yes" if alt_map else "no")

    zs = load_go_set(args.zero_shot_path)
    fs = load_go_set(args.few_shot_path)

    # 4) optional: also canonicalize few/zero shot sets used by FewZeroConfig
    zs = canonicalize_id_list(list(zs), alt_map) if alt_map else zs
    fs = canonicalize_id_list(list(fs), alt_map) if alt_map else fs

    # 3) eval lists canonicalize (observed/seen/unseen/rare)
    eval_id_list = canonicalize_id_list(eval_id_list, alt_map) if alt_map else [go_str_to_int_any(x) for x in
                                                                                eval_id_list]
    eval_seen_go_ids = canonicalize_id_list(eval_seen_go_ids, alt_map) if alt_map else [go_str_to_int_any(x) for x in
                                                                                        eval_seen_go_ids]
    eval_unseen_ids = canonicalize_id_list(eval_unseen_ids, alt_map) if alt_map else [go_str_to_int_any(x) for x in
                                                                                      eval_unseen_ids]
    eval_rare_go_ids = canonicalize_id_list(eval_rare_go_ids, alt_map) if alt_map else [go_str_to_int_any(x) for x in
                                                                                        eval_rare_go_ids]

    res_store = build_stores(args)
    pid2pos, eval_id_list, eval_seen_go_ids, eval_unseen_ids, eval_rare_go_ids = enforce_cache_alignment(
        go_cache=go_cache,
        pid2pos=pid2pos,
        eval_id_list=eval_id_list,
        eval_seen_go_ids=eval_seen_go_ids,
        eval_unseen_ids=eval_unseen_ids,
        eval_rare_go_ids=eval_rare_go_ids,
        logger=logger,
        drop_empty_proteins=False,  # True yaparsan label'sız proteinleri de atar
    )
    datasets = build_datasets(args, res_store, go_text_store, pid2pos=pid2pos, zs=zs, fs=fs, dag_parents=dag_parents)
    n_spe = steps_per_epoch(len(datasets["train"]), args.batch_size)

    go_text_store.materialize_tokens_once(batch_size=512, show_progress=True)
    go_token_dropout = None
    if args.go_token_dropout:
        special_tokens_to_protect = set(tid for tid in [go_encoder.tokenizer.pad_token_id, go_encoder.tokenizer.cls_token_id, go_encoder.tokenizer.sep_token_id] if tid is not None)
        special_tokens_to_protect.update(go_encoder.tokenizer.additional_special_tokens_ids)
        go_dropout_config = GoDropoutConfig(enabled=True, p=0.08, pad_id=go_encoder.tokenizer.pad_token_id,
                    protect_ids=tuple(special_tokens_to_protect))
        go_token_dropout = GoTokenDropout(go_dropout_config)
    train_loader, val_loader = build_dataloaders(datasets, args, go_text_store, go_dropout=go_token_dropout)

    seen_go_ids_prev: set = set()

    if args.wandb:
        try:
            wandb.config.update({'go_encoder_enabled': True}, allow_val_change=True)
        except Exception:
            pass

    out_dir = Path(args.output_dir)

    # Lightweight runtime context
    training_context = TrainingContext(
        device=device,
        go_cache=go_cache,
        faiss_index=None,
        vres=None, #For similarity searches etc.
        current_phase=None,
        last_refresh_epoch=None,
        last_refresh_reason=None,
        batch_builder=None,
        maybe_refresh_phase_resources=None,
        dag_parents=dag_parents,
        dag_children=dag_children,
        go_namespace_map=load_go_namespaces(),
        scheduler=None, #scheduler,
        go_text_store=go_text_store,
        use_queue_miner=bool(args.use_queue_miner),
        attribute_loss_enabled=bool(args.use_attribution_loss),
        return_alpha = bool(args.return_alpha),
        return_slot_attn = bool(args.return_slot_attn),
        fp16_enabled=args.fp16,
        protein_pooling_strategy=args.protein_pooling_strategy,
        eval_id_list=eval_id_list,
        logger=logger,
        eval_seen_go_ids=eval_seen_go_ids,
        eval_unseen_ids=eval_unseen_ids,
        eval_rare_go_ids=eval_rare_go_ids,
        protein_n_slots=args.protein_n_slots,
        go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,
        go_segment_representation_mode=args.go_segment_representation_mode,
    )
    training_context.run_name = args.wandb_run_name or f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    training_context.logging = LoggingConfig(
        log_every=int(args.log_every),
        log_lora_hist=False,
        probe_eval_every=500,
        probe_batch_size=8,
        gospec_tau=0.02,
        gospec_topk=32,
    )

    # ---- SPLIT SANITY CHECK (no eval needed) ----
    eval_set = set(int(x) for x in training_context.eval_id_list)
    seen_set = set(int(x) for x in training_context.eval_seen_go_ids)
    rare_set = set(int(x) for x in training_context.eval_rare_go_ids)
    unseen_set = set(int(x) for x in training_context.eval_unseen_ids)

    print("\n[DBG-SPLIT]")
    print("|eval| =", len(eval_set))
    print("|seen| =", len(seen_set), " intersection(eval) =", len(eval_set & seen_set))
    print("|rare| =", len(rare_set), " intersection(eval) =", len(eval_set & rare_set))
    print("|unseen| =", len(unseen_set), " intersection(eval) =", len(eval_set & unseen_set))
    print("seen ∩ unseen =", len(seen_set & unseen_set))
    print("rare ∩ unseen =", len(rare_set & unseen_set))


    missing = []
    gos = set(int(x) for x in training_context.go_cache.id2row.keys())
    for g in training_context.eval_id_list:
        g_int = int(g)
        if g_int not in gos:
            missing.append(g_int)
    if missing:
        raise RuntimeError(f"go_cache missing {len(missing)} eval GO ids, e.g. {missing[:10]}")

    # infer dims
    with torch.no_grad():
        sample_item = datasets["train"][0]
    d_h = int(sample_item["prot_emb"].shape[1])
    d_g = int(training_context.go_cache.embs.shape[1])
    d_z = int(getattr(args, "align_dim", d_g))

    trainer_cfg = TrainerConfig(
        d_h=d_h,
        d_g=d_g,
        d_z=d_z,
        device=str(device),
        lr=args.lr,
        lr_lora=None if args.lr_lora is None else float(args.lr_lora),
        use_lora=args.use_lora,
        max_epochs=args.epochs,
        cand_chunk_k=args.cand_chunk_k,
        pos_chunk_t=args.pos_chunk_t,
        is_logit_scale_constant=bool(args.is_logit_scale_constant),
        go_pooling=args.go_pooling,
        eval_go_bs=args.eval_go_bs,
        max_inbatch=args.max_inbatch,
        eval_cand_chunk_k=args.eval_cand_chunk_k,
        warmstart_path=args.warmstart_path,
        go_segment_alpha_warmup_steps=args.go_segment_alpha_warmup_steps,
        go_segment_alpha=args.go_segment_alpha,
        trainable_mode=args.trainable_mode,
        weight_decay=args.weight_decay,
        local_window_size=args.local_window_size,
        local_window_stride=args.local_window_stride,
        lambda_slot_div = float(getattr(args, "lambda_slot_div", 0.0)),
        multivec_slot_lse_tau = float(getattr(args, "multivec_slot_lse_tau", 0.10)),
        multivec_global_residual_init = float(getattr(args, "multivec_global_residual_init", 0.25)),
        pairwise_start_step=int(getattr(args, "pairwise_start_step", 0)),
        pairwise_margin=float(getattr(args, "pairwise_margin", 0.0)),
        pairwise_lambda=float(getattr(args, "pairwise_lambda", 0.0)),
        coverage_lambda=float(getattr(args, "coverage_lambda", 0.0)),
        coverage_bottom_frac=float(getattr(args, "coverage_bottom_frac", 0.25)),
        coverage_hard_neg_k=int(getattr(args, "coverage_hard_neg_k", 4)),
        coverage_margin=float(getattr(args, "coverage_margin", 0.05)),
        coverage_start_step=int(getattr(args, "coverage_start_step", 2000)),
        protein_expert_mode=str(getattr(args, "protein_expert_mode", "legacy")),
        local_slot_aggregation=str(getattr(args, "local_slot_aggregation", "lse")),
        local_slot_lse_tau=float(getattr(args, "local_slot_lse_tau", 0.10)),
        expert_global_weight=float(getattr(args, "expert_global_weight", 0.50)),
        expert_fusion_learnable=bool(getattr(args, "expert_fusion_learnable", True)),
        cardinality_weighting=bool(getattr(args, "cardinality_weighting", True)),
    )
    attr_cfg = AttrConfig(
        lambda_attr=getattr(args, "lambda_attr", 0.1),
        lambda_entropy_alpha=getattr(args, "lambda_entropy_alpha", 0.05),
        lambda_entropy_window=getattr(args, "lambda_entropy_window", 0.01),
        topk_per_window=int(getattr(args, "topk_per_window", 64)),
        curriculum_epochs=int(getattr(args, "curriculum_epochs", 10)),
        temperature=float(getattr(args, "temperature", 0.07)),
        lambda_vtrue=getattr(args, "lambda_vtrue", 0.2),
        tau_distill=getattr(args, "tau_distill", 1.5),
        lambda_dag=getattr(args, "lambda_dag", 0.3),
        lambda_bce=getattr(args, "lambda_bce", 0.1),
    )
    run = wandb.init(
        project=args.wandb_project or "protein-go-semantic-align",
        entity=args.wandb_entity,
        name=training_context.run_name,
        config=training_context.to_dict(),
        mode=args.wandb_mode or "online",
        settings=wandb.Settings(code_dir=".", _disable_stats=True),
        reinit=False,
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

    trainer = OppTrainer(cfg=trainer_cfg, attr=attr_cfg, ctx=training_context, queue_cfg=queue_cfg, go_encoder=go_encoder, wandb_run=run)

    if bool(args.use_queue_miner):
        print("[INFO] Using Queue Miner.")
    # Resume
    start_epoch = 0
    global_step = 0

    if getattr(args, "resume", None):
        ckpt_path = str(args.resume)
        start_epoch = load_checkpoint(trainer, ckpt_path, map_location=trainer.device)

        global_step = int(getattr(trainer, "_global_step", 0))
        logger.info(f"[resume] start_epoch={start_epoch}, global_step={global_step}")
    else:
        start_epoch = 0
        global_step = 0

        # --- EVAL ONLY MODU ---
    if getattr(args, "eval_only", False):
        logger.info("Eval-only mode: running validation and exiting, no training.")
        val_logs = trainer.eval_epoch(val_loader, epoch_idx=0)
        msg = " | ".join([f"{k}: {v:.4f}" for k, v in val_logs.items()])
        logger.info(f"[eval_only] {msg}")
        return

    # -------------------------   Training loop -------------------------
    logger.info("Start training for %d epochs", args.epochs)
    wandb.define_metric("trainer_step")
    wandb.define_metric("*", step_metric="trainer_step")
    if training_context.maybe_refresh_phase_resources is not None:
        training_context.maybe_refresh_phase_resources(current_epoch=0, force=False)

    best_val = -inf if args.monitor_mode == "max" else inf
    best_step = global_step
    no_improve_epochs = 0
    EPS = 1e-6

    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # ---- Partial MemoryBank refresh  ----
        try:
            if run is not None and epoch == 0 and global_step == 0:
                try:
                    wandb_preview_curriculum(run, args, total_steps=n_spe * args.epochs)
                    wandb_dataset_quickstats(run, datasets["train"], sample_n=256)
                except Exception as e:
                    logging.getLogger("wandb").warning("deferred previews failed: %r", e)

            if "seen_go_ids_prev" not in locals() or seen_go_ids_prev is None:
                seen_go_ids_prev = set()

            # ---- Partial MemoryBank refresh ----
        #    if getattr(args, "ablation_id", None) != "A0":
        #        ids_eval = list(map(int, training_context.eval_id_list))
        #        ids_seen = sorted(set(int(i) for i in seen_go_ids_prev)) if len(seen_go_ids_prev) > 0 else []

                # eval ids are mandatory
        #        ids_to_update = ids_eval + [i for i in ids_seen if i not in set(ids_eval)]

                # enforce budget
        #        max_r = int(getattr(args, "max_refresh_go", 5000))
        #        if max_r > 0 and len(ids_to_update) > max_r:
        #            ids_to_update = ids_to_update[:max_r]

        #        if len(ids_to_update) > 0:
        #            refresh_go_cache_chunked(
        #                ids_to_update=ids_to_update,
        #                go_text_store=go_text_store,
        #                go_encoder=go_encoder,
        #                go_cache=training_context.go_cache,
        #                device=device,
        #                chunk_size=128
        #            )

        except Exception as _e:
            logging.getLogger("bank").warning("Partial refresh failed: %r", _e)

        seen_go_ids = set()
        if training_context.maybe_refresh_phase_resources is not None:
            training_context.maybe_refresh_phase_resources(current_epoch=epoch, force=False)

        trainer.model.train()
        running = {"total": 0.0, "contrastive": 0.0, "pairwise": 0.0 }
        n_batches = 0

        for batch in train_loader:
            # Collect GO ids seen in this batch for later partial refresh
            try:
                if isinstance(batch, dict) and ('uniq_go_ids' in batch) and batch['uniq_go_ids'] is not None:
                    ids_list = batch['uniq_go_ids'].tolist() if hasattr(batch['uniq_go_ids'], 'tolist') else list(batch['uniq_go_ids'])
                    for _gid in ids_list:
                        seen_go_ids.add(int(_gid))
            except Exception:
                pass

            # forward + losses
            losses = trainer.step_losses(batch, epoch)
            loss = losses["total"]

            # backward
            trainer.opt.zero_grad(set_to_none=True)
            loss.backward()

            if global_step % 200 == 0:
                # protein projection grad
                s = 0.0
                c = 0
                for n, p in trainer.model.proj_p.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        s += float(p.grad.detach().float().norm().item())
                        c += 1
                print(f"[DBG] proj_p grad_norm_sum={s:.4g} over {c}")

                # protein_ln grad
                s = 0.0
                c = 0
                for n, p in trainer.model.protein_ln.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        s += float(p.grad.detach().float().norm().item())
                        c += 1
                print(f"[DBG] protein_ln grad_norm_sum={s:.4g} over {c}")

                # local evidence should be off / absent
                s = 0.0
                c = 0
                lep = getattr(trainer.model, "protein_local_evidence_pool", None)
                if lep is not None:
                    for n, p in lep.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            s += float(p.grad.detach().float().norm().item())
                            c += 1
                print(f"[DBG] local_evidence grad_norm_sum={s:.4g} over {c}")

                # GO projection should stay frozen
                s = 0.0
                c = 0
                for n, p in trainer.model.proj_g.named_parameters():
                    if p.grad is not None:
                        s += float(p.grad.detach().float().norm().item())
                        c += 1
                print(f"[DBG] proj_g grad_norm_sum={s:.4g} over {c}")

                # LoRA should stay frozen
                if getattr(trainer.model, "go_encoder", None) is not None:
                    s = 0.0
                    c = 0
                    for n, p in trainer.model.go_encoder.named_parameters():
                        if p.requires_grad and p.grad is not None and "lora_" in n:
                            s += float(p.grad.detach().float().norm().item())
                            c += 1
                    print(f"[DBG] go_lora grad_norm_sum={s:.4g} over {c}")

            if global_step % 200 == 0:
                # LocalEvidencePool grad
                s = 0.0
                c = 0
                lep = getattr(trainer.model, "protein_local_evidence_pool", None)

                if lep is not None:
                    for n, p in lep.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            s += float(p.grad.detach().float().norm().item())
                            c += 1

                print(f"[DBG] local_evidence grad_norm_sum={s:.4g} over {c}")

                # protein projection grad
                s = 0.0
                c = 0
                for n, p in trainer.model.proj_p.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        s += float(p.grad.detach().float().norm().item())
                        c += 1
                print(f"[DBG] proj_p grad_norm_sum={s:.4g} over {c}")

                # GO projection should stay frozen
                pg = trainer.model.proj_g
                s = 0.0
                c = 0
                for n, p in pg.named_parameters():
                    if p.grad is not None:
                        s += float(p.grad.detach().float().norm().item())
                        c += 1
                print(f"[DBG] proj_g grad_norm_sum={s:.4g} over {c}")

                # LoRA should stay frozen
                if getattr(trainer.model, "go_encoder", None) is not None:
                    s = 0.0
                    c = 0
                    for n, p in trainer.model.go_encoder.named_parameters():
                        if p.requires_grad and p.grad is not None and "lora_" in n:
                            s += float(p.grad.detach().float().norm().item())
                            c += 1
                    print(f"[DBG] go_lora grad_norm_sum={s:.4g} over {c}")

            if global_step % 200 == 0:
                # proj_g grad
                pg = trainer.model.proj_g
                gnorm = 0.0
                cnt = 0
                for n, p in pg.named_parameters():
                    if p.grad is not None:
                        gnorm += float(p.grad.detach().float().norm().item());
                        cnt += 1
                print(f"[DBG] proj_g grad_norm_sum={gnorm:.4g} over {cnt}")

                # go_encoder LoRA grad
                if getattr(trainer.model, "go_encoder", None) is not None:
                    s = 0.0;
                    c = 0
                    for n, p in trainer.model.go_encoder.named_parameters():
                        if p.requires_grad and p.grad is not None and "lora_" in n:
                            s += float(p.grad.detach().float().norm().item());
                            c += 1
                    print(f"[DBG] go_lora grad_norm_sum={s:.4g} over {c}")

            gc = float(getattr(args, "grad_clip", 0.0) or 0.0)
            if gc > 0:
                clip_params = [p for p in trainer.model.parameters() if p.requires_grad]

                if getattr(trainer, "logit_scale", None) is not None:
                    if getattr(trainer.logit_scale, "requires_grad", False):
                        clip_params.append(trainer.logit_scale)

                if clip_params:
                    torch.nn.utils.clip_grad_norm_(clip_params, gc)

            if global_step % 1000 == 0:
                # go_segment_gate grad
                s = 0.0
                c = 0
                if hasattr(trainer.model, "go_segment_gate"):
                    for n, p in trainer.model.go_segment_gate.named_parameters():
                        if p.grad is not None:
                            s += float(p.grad.detach().float().norm().item())
                            c += 1
                print(f"[DBG] go_segment_gate grad_norm_sum={s:.4g} over {c}")

                # proj_g grad
                pg = trainer.model.proj_g
                gnorm = 0.0
                cnt = 0
                for n, p in pg.named_parameters():
                    if p.grad is not None:
                        gnorm += float(p.grad.detach().float().norm().item())
                        cnt += 1
                print(f"[DBG] proj_g grad_norm_sum={gnorm:.4g} over {cnt}")

                # go_encoder LoRA grad
                if getattr(trainer.model, "go_encoder", None) is not None:
                    s = 0.0
                    c = 0
                    for n, p in trainer.model.go_encoder.named_parameters():
                        if p.requires_grad and p.grad is not None and "lora_" in n:
                            s += float(p.grad.detach().float().norm().item())
                            c += 1
                    print(f"[DBG] go_lora grad_norm_sum={s:.4g} over {c}")

            trainer.opt.step()

            # bookkeeping
            n_batches += 1
            for k in running:
                if k in losses and losses[k] is not None:
                    running[k] += float(losses[k].item() if hasattr(losses[k], "item") else float(losses[k]))

            global_step = int(trainer._global_step)

            # per-step logging
            if (global_step % max(1, args.log_every)) == 0:
                avg = {k: (running[k] / max(1, n_batches)) for k in running}
                lr0 = trainer.opt.param_groups[0].get("lr", None)
                logger.info(f"[train] epoch {epoch} step {global_step} :: " +
                            " | ".join([f"{k}:{v:.4f}" for k, v in avg.items()]) +
                            (f" | lr:{lr0:.2e}" if lr0 is not None else ""))
                if args.wandb and run is not None:
                    payload = {f"train/{k}": float(v) for k, v in avg.items()}
                    payload["trainer_step"] = int(global_step)
                    if lr0 is not None:
                        payload["train/lr"] = float(lr0)
                    # logit_scale faydalı
                    try:
                        raw = trainer.logit_scale.detach()
                        clamped = raw.clamp(min=-10.0, max=3.9)
                        payload["train/logit_scale_raw"] = float(raw.item())
                        payload["train/logit_scale_clamped"] = float(clamped.item())
                        payload["train/scale_used"] = float(clamped.exp().item())  # bu forward'da kullanılan
                    except Exception:
                        pass
                    wandb.log(payload, step=int(global_step))

            # periodic checkpoint
            if (global_step % max(1, args.save_every)) == 0:
                ckpt_path = save_checkpoint(
                    out_dir=str(out_dir),
                    tag=f"step{global_step}",
                    trainer=trainer,
                    args=args,
                    epoch=epoch,
                    step=global_step
                )
                cleanup_old_checkpoints(out_dir, keep_last_n=args.keep_last_n)
                # (opsiyonel) wandb artifact

            if global_step % 1000 == 0:
                with torch.no_grad():
                    scale = trainer.logit_scale.exp().item()
                logger.info(f"[debug] step={global_step} logit_scale={scale:.4f}")


        # validation
        if val_loader is not None:
            logger.info("[eval-cache] materializing full GO cache for evaluation")
            eval_ids = list(map(int, training_context.eval_id_list))
            logger.info(f"[eval-debug] eval_id_list size: {len(eval_ids)}")
            try:
                trainer.model.go_encoder.gradient_checkpointing_disable()
            except Exception:
                pass

            trainer.model.go_encoder.eval()

            #VAL
            val_logs = trainer.eval_epoch(val_loader, epoch)
            msg = " | ".join([f"{k}: {val_logs[k]:.4f}" for k in val_logs])
            logger.info(f"[val]   epoch {epoch} :: {msg}")

            # --- monitor ---
            metric_name = args.monitor_metric
            if metric_name not in val_logs:
                logger.warning(f"monitor_metric={metric_name} not in val_logs. Falling back to align_MRR.")
                metric_name = "align_MRR"

            score = float(val_logs[metric_name])

            improved = (score > best_val + EPS) if args.monitor_mode == "max" else (score < best_val - EPS)

            if improved:
                best_val = score
                best_step = global_step
                no_improve_epochs = 0
                # en iyi modeli ayrı etiketle kaydet
                ckpt_path = save_checkpoint(
                    out_dir=str(out_dir),
                    tag=f"step{global_step}",
                    trainer=trainer,
                    args=args,
                    epoch=epoch,
                    step=global_step
                )
                logger.info(f"[ckpt] new BEST {metric_name}={best_val:.4f} @ epoch {epoch} step {global_step}")
                # W&B özetine yazmak istersen:
                if run is not None:
                    run.summary[f"best/{metric_name}"] = best_val
                    run.summary["best/step"] = best_step
                    run.summary["best/epoch"] = epoch
            else:
                no_improve_epochs += 1
                if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                    logger.info(f"[early-stop] no improvement in {no_improve_epochs} epochs; stopping.")
                    raise SystemExit(f"Stopped early at epoch {epoch} (best {metric_name}={best_val:.4f})")


        # Persist seen GO ids for next epoch's partial refresh
        try:
            seen_go_ids_prev = set(seen_go_ids)
        except Exception:
            pass

        # epoch checkpoint
        try:
            ckpt_path = save_checkpoint(
                out_dir=str(out_dir),
                tag=f"epoch{epoch + 1}",
                trainer=trainer,
                args=args,
                epoch=epoch,
                step=global_step
            )
            cleanup_old_checkpoints(out_dir, keep_last_n=args.keep_last_n)
        except Exception:
            pass

    # final checkpoint
    try:
        final_path = save_checkpoint(str(out_dir), tag="final", trainer=trainer,
                                     args=args, epoch=epoch, step=global_step)
    except Exception:
        pass

    logger.info("Training finished. Artifacts saved under: %s", args.output_dir)

    # close wandb
    try:
        if run is not None:
            run.finish()
    except Exception:
        pass


# ============== YAML parser ==============

def load_structured_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data = cfg.get("data", {})
    training = cfg.get("training", {})
    optim = cfg.get("optim", {})
    model = cfg.get("model", {})
    loss = cfg.get("loss", {})
    curriculum = cfg.get("curriculum", {})
    wandb_block = cfg.get("wandb", {})
    stores = cfg.get("stores", {})
    general = cfg.get("general", {})
    queue = cfg.get("queue", {})

    args = types.SimpleNamespace(
        # general
        use_queue_miner = bool(general.get("use_queue_miner", True)),
        go_pooling_strategy = general.get("go_pooling_strategy", "mean"),
        use_lora = bool(general.get("use_lora", False)),
        use_attribution_loss = bool(general.get("use_attribution_loss", False)),
        return_alpha = bool(general.get("return_alpha", False)),
        return_slot_attn = bool(general.get("return_slot_attn", False)),
        protein_pooling_strategy = general.get("protein_pooling_strategy", "mean"),
        go_encoder_inner_pooling=general.get("go_encoder_inner_pooling", "mean"),
        ablation_id = general.get("ablation_id", None),
        phase=general.get("phase", -2),
        protein_n_slots=int(general.get("protein_n_slots", 0)),
        go_pool_type=general.get("go_pool_type", "mean"),
        go_encoder_output_mode=general.get("go_encoder_output_mode", "pooled"),
        go_segment_representation_mode=general.get("go_segment_representation_mode", "segments_only"), #mixed

        # paths / store
        train_ids_path=Path(stores.get("train_ids_path")),
        pid2pos=Path(stores.get("pid2pos_path")),
        val_ids_path=Path(stores.get("val_ids_path")),
        embed_dir_res=Path(stores.get("embed_dir_res", None)),
        go_text_folder=Path(stores.get("go_text_folder")) if stores.get("go_text_folder") else None,
        go_cache_path=Path(stores.get("go_cache_path")) if stores.get("go_cache_path") else None,
        go_path_seen=Path(stores.get("go_path_seen")) if stores.get("go_path_seen") else None,
        go_path_observed=Path(stores.get("go_path_observed")) if stores.get("go_path_observed") else None,
        go_basic_json=Path(stores.get("go_basic_json")) if stores.get("go_basic_json") else None,
        seq_len_lookup=Path(stores.get("seq_len_lookup")) if stores.get("seq_len_lookup") else None,

        # dataset DAG / expansion / few-zero
        overlap=data.get("overlap"),
        max_len=data.get("max_len", 1024),
        use_dag_in_ds=bool(data.get("use_dag_in_ds", False)),
        zero_shot_path=stores.get("zero_shot_path"),
        few_shot_path=stores.get("few_shot_path"),
        fs_target_ratio=float(data.get("fs_target_ratio", 0.3)),

        # training
        epochs=int(training.get("epochs", 10)),
        batch_size=int(training.get("batch_size", 4)),
        eval_batch_size=training.get("eval_batch_size"),
        num_workers=int(training.get("num_workers", 4)),
        fp16=bool(training.get("fp16", True)),
        cpu=bool(training.get("cpu", False)),
        seed=int(training.get("seed", 42)),
        output_dir=training.get("output_dir", "outputs/run1"),
        save_every=int(training.get("save_every", 1000)),
        keep_last_n=int(training.get("keep_last_n", 3)),
        resume=training.get("resume"),
        warmstart_path=training.get("warmstart_path"),
        eval_only=bool(training.get("eval_only", False)),
        log_every=int(training.get("log_every", 50)),
        log_level=training.get("log_level", "INFO"),
        monitor_metric=training.get("monitor_metric", "align_MRR"),
        monitor_mode=training.get("monitor_mode", "max"),
        early_stop_patience=int(training.get("early_stop_patience", 0)),
        cand_chunk_k=int(training.get("cand_chunk_k", 8)),
        pos_chunk_t=int(training.get("pos_chunk_t", 128)),
        general_device=str(training.get("device", "cuda:0")),
        max_refresh_go=int(training.get("max_refresh_go", 5000)),
        eval_go_bs=int(training.get("eval_go_bs", 256)),
        go_text_store_max_len=int(training.get("go_text_store_max_len", 512)),
        is_logit_scale_constant=bool(training.get("is_logit_scale_constant", False)),
        go_token_dropout=bool(training.get("go_token_dropout", False)),
        go_pooling=str(training.get("go_pooling", "mean")),
        eval_space=str(training.get("eval_space", "seen")),
        max_inbatch=int(training.get("max_inbatch", 64)),
        eval_cand_chunk_k=int(training.get("eval_cand_chunk_k", 8)),
        go_segment_max_len=int(training.get("go_segment_max_len", 64)),
        go_segment_alpha=float(training.get("go_segment_alpha", 0.5)),
        go_segment_alpha_warmup_steps=int(training.get("go_segment_alpha_warmup_steps", 10000)),
        local_window_size=int(training.get("local_window_size", 64)),
        local_window_stride=int(training.get("local_window_stride", 32)),
        pairwise_margin=float(training.get("pairwise_margin", 0.0)),
        pairwise_start_step=int(training.get("pairwise_start_step", 0)),
        coverage_start_step=int(training.get("coverage_start_step", 2000)),
        cardinality_weighting=bool(training.get("cardinality_weighting", False)),

        # optim
        lr=float(optim.get("lr", 3e-4)),
        weight_decay=float(optim.get("weight_decay", 0.01)),
        grad_clip=float(optim.get("grad_clip", 1.0)),
        lr_lora=optim.get("lr_lora", None),
        trainable_mode=str(training.get("trainable_mode", "full")),

        # queue params
        queue_K=int(queue.get("queue_K", 16384)),
        queue_start_step=int(queue.get("queue_start_step", 0)),
        queue_hard_frac_start=float(queue.get("queue_hard_frac_start", 0.0)),
        queue_hard_frac_end=float(queue.get("queue_hard_frac_end", 0.0)),
        queue_hard_frac_warmup_steps=float(queue.get("queue_hard_frac_warmup_steps", 0.0)),
        queue_weight_start=float(queue.get("queue_weight_start", 0.01)),
        queue_weight_end=float(queue.get("queue_weight_end", 0.01)),
        queue_weight_warmup_steps=float(queue.get("queue_weight_warmup_steps", 0.0)),
        k_hard_queue_start=int(queue.get("k_hard_queue_start", 0)),
        k_hard_queue_end=int(queue.get("k_hard_queue_end", 0)),
        k_hard_queue_warmup_steps=int(queue.get("k_hard_queue_warmup_steps", 0.0)),

        # model / attention
        align_dim=int(model.get("align_dim", 768)),
        temperature=float(model.get("temperature", 0.07)),
        attn_heads=int(model.get("attn_heads", 2)),
        attn_dropout=float(model.get("attn_dropout", 0.1)),
        entropy_reg=float(model.get("entropy_reg", 0.01)),
        win_size=int(model.get("win_size", 1024)),
        win_stride=int(model.get("win_stride", 256)),
        multivec_slot_lse_tau=float(model.get("multivec_slot_lse_tau", 0.10)),
        multivec_global_residual_init=float(model.get("multivec_global_residual_init", 0.25)),
        protein_expert_mode=str(model.get("protein_expert_mode", "legacy")),
        local_slot_aggregation=str(model.get("local_slot_aggregation", "lse")),
        local_slot_lse_tau=float(model.get("local_slot_lse_tau", 0.10)),
        expert_global_weight=float(model.get("expert_global_weight", 0.50)),
        expert_fusion_learnable=bool(model.get("expert_fusion_learnable", True)),

        # loss
        lambda_con=float(loss.get("lambda_con", 1.0)),
        lambda_dag=float(loss.get("lambda_dag", 0.0)),
        lambda_attr=float(loss.get("lambda_attr", 0.0)),
        lambda_entropy_alpha=float(loss.get("lambda_entropy_alpha", 0.0)),
        dag_margin=float(loss.get("dag_margin", 0.05)),
        dag_scale=float(loss.get("dag_scale", 10.0)),
        lambda_vtrue=float(loss.get("lambda_vtrue", 0.2)),
        tau_distill=float(loss.get("tau_distill", 1.5)),
        lambda_bce=float(loss.get("lambda_bce", 0.1)),
        lambda_slot_div=float(loss.get("lambda_slot_div", 0.0)),
        pairwise_lambda=float(loss.get("pairwise_lambda", 0.0)),
        coverage_lambda=float(loss.get("coverage_lambda", 0.0)),
        coverage_bottom_frac=float(loss.get("coverage_bottom_frac", 0.25)),
        coverage_hard_neg_k=int(loss.get("coverage_hard_neg_k", 4)),
        coverage_margin=float(loss.get("coverage_margin", 0.05)),

        # curriculum
        curriculum_epochs=int(curriculum.get("epochs", 4)),
        warmup_frac=float(curriculum.get("warmup_frac", 0.1)),
        curriculum_mode=curriculum.get("mode", "cosine"),
        hard_frac_start=float((curriculum.get("hard_frac") or [0.2, 0.7])[0]),
        hard_frac_end=float((curriculum.get("hard_frac") or [0.2, 0.7])[1]),
        shortlist_M_start=int((curriculum.get("shortlist_M") or [256, 1024])[0]),
        shortlist_M_end=int((curriculum.get("shortlist_M") or [256, 1024])[1]),
        k_hard_start=int((curriculum.get("k_hard") or [16, 64])[0]),
        k_hard_end=int((curriculum.get("k_hard") or [16, 64])[1]),
        hier_up_start=int((curriculum.get("hier_up") or [1, 0])[0]),
        hier_up_end=int((curriculum.get("hier_up") or [1, 0])[1]),
        hier_dn_start=int((curriculum.get("hier_dn") or [0, 0])[0]),
        hier_dn_end=int((curriculum.get("hier_dn") or [0, 0])[1]),
        random_k_start=int((curriculum.get("random_k") or [8, 0])[0]),
        random_k_end=int((curriculum.get("random_k") or [8, 0])[1]),
        inbatch_easy_start=float((curriculum.get("inbatch_easy") or [1.0, 0.0])[0]),
        inbatch_easy_end=float((curriculum.get("inbatch_easy") or [1.0, 0.0])[1]),
        neg_k=int(curriculum.get("neg_k", 4)),

        # wandb
        wandb=bool(wandb_block.get("enabled", False)),
        wandb_project=wandb_block.get("project", "protein-go-align"),
        wandb_entity=wandb_block.get("entity"),
        wandb_run_name=wandb_block.get("wandb_run_name"),
        wandb_mode=wandb_block.get("mode", "online"),
    )
    return args

def parse_args():
    parser = argparse.ArgumentParser(description="Protein–GO Alignment Training")

    # --- temel ayarlar ---
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file path (örn: src/configs/colab.yaml)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="cuda device")

    args = parser.parse_args()
    if not args.config:
        args.config = YAML_FILE
        print(f"[main] No --config passed, defaulting to {args.config}")

    args = load_structured_cfg(args.config)

    if args.ablation_id is not None:
        print("[Ablation] Applying ablation ID:", args.ablation_id)

    return args

def main():
    args = parse_args()
    out = Path(args.output_dir)
    setup_logging(out, level=args.log_level)
    set_seed(args.seed)
    t0 = time.time()
    try:
        run_training(args)
    except Exception as e:
        logging.exception("Fatal error: %s", repr(e))
        raise
    finally:
        logging.info("Total runtime: %.1f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
