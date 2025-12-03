import os
import sys
import time
import math
import yaml
import random
from torch.utils.data import DataLoader
from typing import Dict, List, Set
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
import torch.nn.functional as F
import argparse

import wandb
from src.configs.paths import TRAINING_CONFIG as _TRAINING_CONFIG_DEFAULT, GO_INDEX
from src.datasets import ESMResidueStore, ESMFusedStore, GoTextStore, VectorResources
from src.datasets.protein_dataset import ProteinEmbDataset, ProteinFusedQueryDataset
from src.training.collate import ContrastiveEmbCollator, fused_collator
from src.training.trainer import OppTrainer, ema_update
from src.configs.data_classes import (
    FewZeroConfig, TrainSchedule, TrainerConfig, AttrConfig, LoRAParameters, TrainingContext, LoggingConfig
)
from src.training.curriculum import CurriculumConfig, CurriculumScheduler
from src.go import GoLookupCache
from src.go import load_go_parents, load_go_children
from src.utils import (
    load_go_set, load_raw_json, load_raw_txt, load_go_texts_by_phase
)
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.encoders import BioMedBERTEncoder
from math import inf

from src.configs.paths import (
    PROTEIN_TRAIN_IDS,
    PROTEIN_VAL_IDS,
    PID_TO_POSITIVES,
    ZERO_SHOT_TERMS_ID_ONLY_JSON,
    FEW_SHOT_IC_TERMS_ID_ONLY_JSON,
    P_SEQ_LEN_LOOKUP,
    GOOGLE_DRIVE_MANIFEST_CACHE,
    TRAINING_CONFIG,
    COMMON_IC_GO_TERMS_ID_ONLY_JSON,
    GO_INDEX_NON_PHASE
)

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
        # Eşlemleri opsiyonel yan dosyalardan yükle (varsa)
        id2row = row2id = None
        for fname in ("id2row.json", "row2id.json", "ids.json", "ids.txt"):
            f = memmap_path.with_name(fname)
            if f.exists():
                if f.suffix == ".json":
                    with open(f, "r") as fp:
                        data = json.load(fp)
                    if "id2row" in data and id2row is None:
                        id2row = data["id2row"]
                    if "row2id" in data and row2id is None:
                        row2id = data["row2id"]
                    if "ids" in data and (id2row is None or row2id is None):
                        ids = data["ids"]
                        id2row = {pid: i for i, pid in enumerate(ids)}
                        row2id = {i: pid for i, pid in enumerate(ids)}
                else:
                    # ids.txt (satır başına bir id)
                    with open(f, "r") as fp:
                        ids = [line.strip() for line in fp if line.strip()]
                    id2row = {pid: i for i, pid in enumerate(ids)}
                    row2id = {i: pid for i, pid in enumerate(ids)}

        # Yan dosyalar yoksa: boyutu okumadan basit map üret (isteğe bağlı)
        if id2row is None or row2id is None:
            try:
                # Sadece shape öğrenmek için memmap aç (RAM'e yüklemez)
                arr = np.load(memmap_path, mmap_mode="r", allow_pickle=False)
                n = int(arr.shape[0])
                id2row = {str(i): i for i in range(n)}
                row2id = {i: str(i) for i in range(n)}
            except Exception:
                # Eşlemeleri boş bırak; GoLookupCache içi tolere edebiliyorsa
                id2row = None
                row2id = None

        blob = {
            "memmap_path": str(memmap_path),
            "id2row": id2row,
            "row2id": row2id,
        }
        return GoLookupCache(blob)

    blob = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "embs" in blob and isinstance(blob["embs"], torch.Tensor):
        blob["embs"] = F.normalize(blob["embs"].float(), p=2, dim=1)
    return GoLookupCache(blob)



def build_stores(args):
    """
    Returns:
      res_store:  ESMResidueStore  ([L,D])
      fused_store: ESMFusedStore or None  ([D])
    CLI/args beklenen alanlar:
      - args.seq_len_lookup (pickle path)
      - args.embed_dir_res (residue kökü)   [zorunlu]
      - args.embed_dir_fused (fused kökü)   [opsiyonel]
      - args.fp16 (bool), args.max_len, args.overlap
    """
    import logging, os
    logger = logging.getLogger("build_stores")
    logger.info("Building residue+fused stores (lazy, no snapshot)...")

    # 1) seq len lookup
 #   seq_len_lookup = load_raw_pickle(args.seq_len_lookup)

    # 2) HF cache kökü (opsiyonel; şu an lazy fetch kapalı)
    hub_local_dir = getattr(args, "hub_local_dir", None) or "/content/hf_cache"
    Path(hub_local_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hub_local_dir))

    # 3) embed dirs
    embed_dir_res = getattr(args, "embed_dir_res", None) or hub_local_dir

    embed_dir_fused = getattr(args, "embed_dir_fused", None)  # None olabilir

    # 4) toggles
    gdrive_cache = False
    cache_shards = getattr(args, "no_cache_shards", False)

    logger.info(
        "Store config:\n"
        f"  embed_dir_res   = {embed_dir_res}\n"
        f"  embed_dir_fused = {embed_dir_fused}\n"
        f"  cache_shards    = {cache_shards}\n"
        f"  prefer_fp16     = {args.fp16}\n"
        f"  max_len/overlap = {getattr(args,'max_len',None)}/{getattr(args,'overlap',None)}"
    )

    # 5) Build residue
    res_store = ESMResidueStore(
        embed_dir=embed_dir_res,
      #  seq_len_lookup=seq_len_lookup,
        max_len=args.max_len,
        overlap=args.overlap,
        gdrive_cache=gdrive_cache,
        prefer_fp16=args.fp16,
        # HF lazy fetch paramlarını bilinçli olarak kapalı bırakıyoruz
    )

    # 6) Build fused (opsiyonel)
    fused_store = None
    if embed_dir_fused:
        fused_store = ESMFusedStore(
            embed_dir=embed_dir_fused,
            gdrive_cache=gdrive_cache,
            prefer_fp16=args.fp16,
        )

    logger.info("Stores ready.")
    return res_store, fused_store

def build_val_dataset(
    val_pids,
    pid2pos_val,
    go_cache,
    fewzero_cfg,
    dag_parents,
    residue_store: ESMResidueStore,
):
    ds_val = ProteinEmbDataset(
        protein_ids=val_pids,
        pid2pos=pid2pos_val,
        go_cache=go_cache,
        fewzero=fewzero_cfg,
        dag_parents=dag_parents,
        store=residue_store,
        fused_store=None,
        include_fused=False,   # validation için fused gereksiz
    )
    return ds_val

def build_datasets(args, res_store: ESMResidueStore, fused_store:ESMFusedStore, go_cache: GoLookupCache) -> Dict[str, torch.utils.data.Dataset]:
    logger = logging.getLogger("build_datasets")
    logger.info("Building datasets...")

    logger = logging.getLogger("build_datasets")
    logger.info("Building datasets...")
    logger.info(f"PID_TO_POSITIVES path = {args.pid2pos}")

    pid2pos = load_raw_json(args.pid2pos)
    train_ids = load_raw_txt(PROTEIN_TRAIN_IDS)
    val_ids = load_raw_txt(PROTEIN_VAL_IDS)

    fused_dir = str(Path(args.embed_dir_fused))  # senin kullandığın yol
    have_fused = _collect_fused_ids(fused_dir)

    def _filter_existing(ids):
        kept = [pid for pid in ids if pid in have_fused]
        missing = len(ids) - len(kept)
        return kept, missing

    train_ids_kept, train_missing = _filter_existing(train_ids)
    val_ids_kept, val_missing = _filter_existing(val_ids)

    if train_missing or val_missing:
        logging.getLogger("build_datasets").warning(
            "Fused bank'te olmayan ID'ler elendi: train_missing=%d, val_missing=%d",
            train_missing, val_missing
        )
    train_ids = train_ids_kept
    val_ids = val_ids_kept

    zs = load_go_set(ZERO_SHOT_TERMS_ID_ONLY_JSON)
    fs = load_go_set(FEW_SHOT_IC_TERMS_ID_ONLY_JSON)
    common = load_go_set(COMMON_IC_GO_TERMS_ID_ONLY_JSON)
    fz = FewZeroConfig(zero_shot_terms=zs, few_shot_terms=fs, common_terms=common,
                       fs_target_ratio=args.fs_target_ratio)

    dag_parents = load_go_parents()
    #ancestor_stoplist = load_raw_txt(GO_ANCESTOR_STOPLIST)

    train_ds = ProteinEmbDataset(
        protein_ids=train_ids,
        pid2pos=pid2pos,
        go_cache=go_cache,
        fewzero=fz,
        dag_parents=dag_parents,
        store=res_store,
        fused_store=None,  # eğitimde gerekmez
        include_fused=False
    )

    val_ds = build_val_dataset(val_pids=val_ids, pid2pos_val=pid2pos, go_cache=go_cache, fewzero_cfg=fz,
                              dag_parents=dag_parents, residue_store=res_store)
    if fused_store is None:
        raise RuntimeError("Query search için fused_store zorunlu.")
    query_ds = ProteinFusedQueryDataset(train_ids+val_ids, fused_store=fused_store)

    logger.info("Datasets ready. Train=%d%s", len(train_ds), f", Val={len(val_ds)}" if val_ds else "")
    return {"train": train_ds, "val": val_ds, "query_ds": query_ds}


def build_dataloaders(datasets, args, go_cache: GoLookupCache, go_text_store: GoTextStore):
    logger = logging.getLogger("build_dataloaders")

    train_ds = datasets["train"]
    zs_mask_np = getattr(train_ds, "zs_mask", None)
    zs_mask_vec = torch.as_tensor(zs_mask_np, dtype=torch.bool) if zs_mask_np is not None else None

    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    collate = ContrastiveEmbCollator(
        go_lookup=go_cache,
        go_text_store=go_text_store, #tokenizer
        zs_mask_vec=zs_mask_vec,
        bidirectional=True,
        neg_k=args.neg_k
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
#        multiprocessing_context="forkserver",
        pin_memory=True,
    #    prefetch_factor=2,
        collate_fn=collate,
    )
    b = next(iter(train_loader))
    assert "protein_ids" in b and isinstance(b["protein_ids"], list) and len(b["protein_ids"]) == b["prot_emb_pad"].shape[0]
    val_loader = None
    if datasets.get("val") is not None:
        val_loader = DataLoader(
            datasets["val"],
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            collate_fn=collate,
#            multiprocessing_context="forkserver",
            drop_last=False
        )

    if datasets.get("query_ds") is None:
        raise RuntimeError("Query dataset is required for retrieval/indexing.")

    query_loader = DataLoader(
        datasets.get("query_ds"),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=fused_collator)

    logger.info("Dataloaders ready. batch_size=%d", args.batch_size)
    return train_loader, val_loader, query_loader, collate


def build_scheduler_cfg(args, n_steps_per_epoch: int) -> CurriculumConfig:
    total_steps = max(1, n_steps_per_epoch * args.curriculum_epochs)
    warmup = int(args.warmup_frac * n_steps_per_epoch)

    hier_dn_start = int(getattr(args, "hier_dn_start", 0))
    hier_dn_end = int(getattr(args, "hier_dn_end", 0))
    hier_up_start = int(getattr(args, "hier_up_start", 0))
    hier_up_end = int(getattr(args, "hier_up_end", 0))

    shortlist_M_start = int(getattr(args, "shortlist_M_start", 256))
    shortlist_M_end = int(getattr(args, "shortlist_M_end", 1024))
    k_hard_start = int(getattr(args, "k_hard_start", 16))
    k_hard_end = int(getattr(args, "k_hard_end", 64))
    hard_frac_start = float(getattr(args, "hard_frac_start", 0.2))
    hard_frac_end = float(getattr(args, "hard_frac_end", 0.7))
    inbatch_easy_start = float(getattr(args, "inbatch_easy_start", 1.0))
    inbatch_easy_end = float(getattr(args, "inbatch_easy_end", 0.0))
    random_k_start = int(getattr(args, "random_k_start", 8))
    random_k_end = int(getattr(args, "random_k_end", 0))

    cfg = CurriculumConfig(
        total_steps=total_steps,
        hard_frac=(hard_frac_start, hard_frac_end),
        shortlist_M=(shortlist_M_start, shortlist_M_end),
        k_hard=(k_hard_start, k_hard_end),
        hier_max_hops_up=(hier_up_start, hier_up_end),
        hier_max_hops_down=(hier_dn_start, hier_dn_end),
        random_k=(random_k_start, random_k_end),
        use_inbatch_easy=(inbatch_easy_start, inbatch_easy_end),
        mode=args.curriculum_mode,
        warmup=warmup,
    )
    return cfg

# === Add near the other utils in main.py ===
def materialize_fused_bank(query_loader, *, device: str = "cpu"):
    """
    Consume query_loader (ProteinFusedQueryDataset) once and build:
      - ids:   List[str]               length N
      - vecs:  torch.FloatTensor [N,D] on CPU (or device)
      - id2row: Dict[str, int]
    """
    import logging
    logger = logging.getLogger("fused_bank")
    ids_all: List[str] = []
    vecs_all: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in query_loader:
            # fused_collator -> {"protein_ids": [...], "prot_fused": [B,D]}
            ids = batch["protein_ids"]
            Z = batch["prot_fused"]  # [B,D] float32
            if device and device != "cpu":
                Z = Z.to(device, non_blocking=True)
            ids_all.extend(ids)
            vecs_all.append(Z.cpu() if device == "cpu" else Z)

    if not vecs_all:
        raise RuntimeError("query_loader yielded no fused vectors.")

    vecs = torch.cat(vecs_all, dim=0).contiguous()  # [N,D]
    if device != "cpu":  # keep one canonical CPU copy for indexing by row
        vecs_cpu = vecs.detach().cpu()
    else:
        vecs_cpu = vecs
    if len(ids_all) != vecs_cpu.size(0):
        raise RuntimeError(f"IDs and vectors length mismatch: {len(ids_all)} vs {vecs_cpu.size(0)}")

    id2row = {pid: i for i, pid in enumerate(ids_all)}
    logger.info("Fused bank built: N=%d, D=%d", vecs_cpu.size(0), vecs_cpu.size(1))
    return {"ids": ids_all, "vecs": vecs_cpu, "id2row": id2row}


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

    valid_ids: Set[int] = set(int(g) for g in go_text_store.id2tok.keys())

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

    if new_is_fs:
        train_ds.is_fs = new_is_fs


def sanity_check_go_text(train_ids, pid2pos, go_text_store):
    used_terms = set()
    for pid in train_ids:
        gids = pid2pos.get(pid, [])
        for g in gids:
            used_terms.add(int(g))

    missing = [g for g in used_terms if g not in go_text_store.id2tok]
    if missing:
        raise RuntimeError(
            f"GoTextStore is missing {len(missing)} GO ids, e.g. {missing[:10]}"
        )


# ============== Runner ==============
def run_training(args, schedule: TrainSchedule):
    signal.signal(signal.SIGINT, _sigint_handler)
    logger = logging.getLogger("main")
    if args.general_device is not None:
        device = args.general_device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if args.wandb:
        wandb.login()


    # Phase 0 resources
    if schedule is not None:
        phase0 = 0
        go_cache_path = schedule.resolve_go_cache_path(phase0)
    else:
        phase0 = -1 # ablation 1 - no phase
        go_cache_path = GO_INDEX[phase0]["TEXT_EMB"]

    go_cache = build_go_cache(str(go_cache_path))
    dag_parents = load_go_parents()
    dag_children = load_go_children()

    # GO text dict per phase
    total_phases = (len(schedule.phase_breaks) + 1) if schedule is not None and hasattr(schedule, "phase_breaks") else 1
    go_id_to_text: Dict[int, Dict[int, str]] = {}
    if phase0 == -1:
        go_id_to_text[phase0] = load_go_texts_by_phase(args.go_text_folder, phase=phase0)
    else:
        for ph in range(total_phases):
            go_id_to_text[ph] = load_go_texts_by_phase(args.go_text_folder, phase=phase0)

    res_store, fused_store = build_stores(args)
    datasets = build_datasets(args, res_store, fused_store, go_cache)
    n_spe = steps_per_epoch(len(datasets["train"]), args.batch_size)

    # Text encoder (GO)
    lora_params = LoRAParameters(adapter_name="go_encoder")
    go_encoder = BioMedBERTEncoder(
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        device=device,
        max_length=512,
        use_attention_pool=args.use_attention_pooling,
        attn_hidden=128,
        attn_dropout=0.1,
        special_token_weights=None,
        enable_lora=args.use_lora,
        lora_parameters=lora_params,
    )

    # GoTextStore + dataloaders
    lazy = True if phase0 >=0 else False
    go_text_store = GoTextStore(go_id_to_text, go_encoder.tokenizer, phase=phase0, lazy=lazy)

    print("GoTextStore size:", len(go_text_store.id2tok))

    # Training dataset cleaning
    print("[MAIN] Sanitizing training dataset with GoTextStore...")
    sanitize_dataset_with_go_text(datasets["train"], go_text_store)
    sanity_check_go_text(datasets["train"].pids, datasets["train"].pid2pos, go_text_store)

    # Val dataset cleaning
    print("[MAIN] Sanitizing validation dataset with GoTextStore...")
    sanitize_dataset_with_go_text(datasets["val"], go_text_store)
    sanity_check_go_text(datasets["val"].pids, datasets["val"].pid2pos, go_text_store)

    eval_id_list = collect_eval_ids_from_datasets(datasets["train"], datasets["val"])
    logging.getLogger("main").info(
        "[eval] using %d GO terms in eval_id_list", len(eval_id_list)
    )

    go_text_store.materialize_tokens_once(batch_size=512, show_progress=True)
    train_loader, val_loader, query_loader, collate = build_dataloaders(datasets, args, go_cache, go_text_store)

    # Memory bank - GoCache init (Memory bank deprecated, to be cleaned.)
    memory_bank = go_cache if args.use_go_memory_bank else None
    if args.use_go_memory_bank:
        logger.info(f"Using unified GoLookupCache as MemoryBank ({memory_bank.device} fp32, no copies).")

    seen_go_ids_prev: set = set()

    if args.wandb:
        try:
            wandb.config.update({'go_encoder_enabled': True}, allow_val_change=True)
        except Exception:
            pass

    if args.ablation_id == "A1":
        scheduler = None
    else:
        cur_cfg = build_scheduler_cfg(args, n_spe)
        scheduler = CurriculumScheduler(cur_cfg)


    # Vector resources — FAISS YOK: sadece bank embs ile
    vres = VectorResources(faiss_index=None, go_embs=go_cache.embs, device=device)

    out_dir = Path(args.output_dir)

    # Lightweight runtime context
    training_context = TrainingContext(
        device=device,
        schedule=schedule,
        go_cache=go_cache,
        faiss_index=None,
        vres=vres, #For similarity searches etc.
        memory_bank=memory_bank,
        current_phase=None,
        last_refresh_epoch=None,
        last_refresh_reason=None,
        batch_builder=None,
        maybe_refresh_phase_resources=None,
        dag_parents=dag_parents,
        dag_children=dag_children,
        scheduler=scheduler,
        go_text_store=go_text_store,
        use_queue_miner=bool(args.use_queue_miner),
        attribute_loss_enabled=bool(args.use_attribution_loss),
        return_alpha = bool(args.return_alpha),
        fp16_enabled=args.fp16,
        pooling_strategy=args.pooling_strategy,
        eval_id_list=eval_id_list
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
    fused_bank = materialize_fused_bank(query_loader, device=str(device))
    training_context.fused_bank = fused_bank

    assert isinstance(fused_bank, dict) and {"ids", "vecs", "id2row"} <= fused_bank.keys()
    assert len(fused_bank["ids"]) == fused_bank["vecs"].shape[0] > 0, "fused bank boş/eksik"

    def maybe_refresh_phase_resources(current_epoch: int, *, force: bool = False):
        if phase0 == -1:
            return
        new_phase = training_context.schedule.phase_for_epoch(current_epoch)
        prev_phase = training_context.current_phase

        if prev_phase is None:
            training_context.current_phase = new_phase
            try:
                training_context.vres.set_backends(None, training_context.go_cache.embs)
            except Exception:
                pass
            training_context.last_refresh_epoch = current_epoch
            training_context.last_refresh_reason = "init"
            training_context.go_text_store.update_phase_and_tokenize(new_phase)
            return

        if force or (new_phase != prev_phase):
            logger.info(f"[PHASE SWITCH] epoch={current_epoch} :: {prev_phase + 1} -> {new_phase + 1}")

            new_go_path = training_context.schedule.resolve_go_cache_path(new_phase)
            new_go_cache = build_go_cache(new_go_path)
            training_context.go_cache = new_go_cache
            training_context.memory_bank = new_go_cache if args.use_go_memory_bank else None
            training_context.vres.set_backends(None, training_context.go_cache.embs)

            # GoTextStore fazı
            training_context.go_text_store.update_phase_and_tokenize(new_phase)

            # Dataloader’ları yeniden kur
            nonlocal train_loader, val_loader, collate
            train_loader, val_loader, query_loader, collate = build_dataloaders(datasets=datasets, args=args, go_cache=training_context.go_cache, go_text_store=training_context.go_text_store)

            training_context.current_phase = new_phase
            training_context.last_refresh_epoch = current_epoch
            training_context.last_refresh_reason = "phase_change" if not force else "force"

    # expose refresher
    training_context.maybe_refresh_phase_resources = maybe_refresh_phase_resources

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
        max_epochs=args.epochs,
        cand_chunk_k=args.cand_chunk_k,
        pos_chunk_t=args.pos_chunk_t,
        k_hard_queue=args.k_hard_queue,
        queue_K=args.queue_K
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
    )
    encoder_for_trainer = go_encoder if args.ablation_id != "A1" else None
    trainer = OppTrainer(cfg=trainer_cfg, attr=attr_cfg, ctx=training_context, go_encoder=encoder_for_trainer)

    if bool(args.use_queue_miner):
        print("[INFO] Using Queue Miner.")
        trainer._maybe_init_queue_miner(d_g)
    # Resume
    if getattr(args, "resume", None):
        ckpt_path = str(args.resume)
        load_checkpoint(trainer, ckpt_path, map_location=trainer.device)

        # --- EVAL ONLY MODU ---
    if getattr(args, "eval_only", False):
        logger.info("Eval-only mode: running validation and exiting, no training.")
        val_logs = trainer.eval_epoch(val_loader, epoch_idx=0)
        logger.info(
                "[eval_only] total={total:.4f} contrastive={contrastive:.4f} "
                "dag={dag:.4f} attr={attr:.4f} entropy={entropy:.4f} "
                "cafa_fmax={cafa_fmax:.4f} cafa_aupr={cafa_aupr:.4f}".format(**val_logs)
            )
        return  # KRİTİK: training loop'a girmeden çık

    # -------------------------   Training loop -------------------------
    logger.info("Start training for %d epochs", args.epochs)
    wandb.define_metric("trainer_step")
    wandb.define_metric("*", step_metric="trainer_step")
    training_context.maybe_refresh_phase_resources(current_epoch=0, force=False)

    best_val = -inf if args.monitor_mode == "max" else inf
    best_step = 0
    no_improve_epochs = 0
    EPS = 1e-6
    global_step = 0

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # ---- Partial MemoryBank refresh using GO ids seen in the previous epoch ----
        try:
            if args.wandb and (wandb.run is not None) and global_step == 1:
                try:
                    wandb_preview_curriculum(wandb, args, total_steps=n_spe * args.epochs)
                    wandb_dataset_quickstats(wandb, datasets["train"], sample_n=256)
                except Exception as e:
                    logging.getLogger("wandb").warning("deferred previews failed: %r", e)

            # A1'de GO cache SABİT, hiçbir refresh yok
            if getattr(args, "ablation_id", None) == "A1":
                pass
            else:
                if len(seen_go_ids_prev) > 0:
                    ids_to_update = sorted(set(int(i) for i in seen_go_ids_prev))
                    toks = go_text_store.batch(ids_to_update)
                    with torch.no_grad():
                        new_embs = go_encoder(
                            input_ids=toks['input_ids'].to(device),
                            attention_mask=toks['attention_mask'].to(device)
                        ).detach().cpu()
                    new_embs = F.normalize(new_embs, p=2, dim=1).cpu()
                    training_context.go_cache.update(ids_to_update, new_embs)

        except Exception as _e:
            logging.getLogger('bank').warning('Partial refresh failed: %r', _e)


        seen_go_ids = set()
        training_context.maybe_refresh_phase_resources(current_epoch=epoch, force=False)

        trainer.model.train()
        running = {"total": 0.0, "contrastive": 0.0, "dag": 0.0, "attr": 0.0, "entropy": 0.0}
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
            with torch.autocast("cuda", dtype=torch.bfloat16): #TODO: autocast
                losses = trainer.step_losses(batch, epoch)
                loss = losses["total"]

            # backward
            trainer.opt.zero_grad(set_to_none=True)
            loss.backward()

            # ==== DEBUG: grad norm ====
            if global_step % 500 == 0:
                # no_grad kullanma, grad'ı okumak için gerek yok
                grads = []
                for name, p in trainer.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        g = p.grad.detach().view(-1)
                        grads.append(g)
                if getattr(trainer, "logit_scale", None) is not None and trainer.logit_scale.grad is not None:
                    grads.append(trainer.logit_scale.grad.detach().view(-1))

                if grads:
                    all_grads = torch.cat(grads)
                    gnorm = float(all_grads.norm().item())
                else:
                    gnorm = 0.0

                print(f"[debug] step={global_step} grad_norm={gnorm:.3e}")
                # istersen tek tek de bak:
                for name, p in trainer.model.named_parameters():
                    if p.grad is not None:
                        print("  ", name, "grad_norm", float(p.grad.norm().item()))

                if trainer.logit_scale.grad is not None:
                    print("  logit_scale grad_norm", float(trainer.logit_scale.grad.norm().item()))

            gc = float(getattr(args, "grad_clip", 0.0) or 0.0)
            if gc > 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), gc)

            trainer.opt.step()

            # bookkeeping
            n_batches += 1
            for k in running:
                if k in losses and losses[k] is not None:
                    running[k] += float(losses[k].item() if hasattr(losses[k], "item") else float(losses[k]))

            global_step += 1

            # per-step logging
            if (global_step % max(1, args.log_every)) == 0:
                avg = {k: (running[k] / max(1, n_batches)) for k in running}
                lr0 = trainer.opt.param_groups[0].get("lr", None)
                logger.info(f"[train] epoch {epoch} step {global_step} :: " +
                            " | ".join([f"{k}:{v:.4f}" for k, v in avg.items()]) +
                            (f" | lr:{lr0:.2e}" if lr0 is not None else ""))

            # periodic checkpoint
            if (global_step % max(1, args.save_every)) == 0:
                ckpt_path = save_checkpoint(
                    out_dir=out_dir.name,
                    tag=f"step{global_step}",
                    trainer=trainer,
                    args=args,
                    epoch=epoch,
                    step=global_step
                )
                cleanup_old_checkpoints(out_dir, keep_last_n=args.keep_last_n)
                # (opsiyonel) wandb artifact

        # epoch finished - update ema
        trainer.on_epoch_finish()

        # validation
        if val_loader is not None:
            val_logs = trainer.eval_epoch(val_loader, epoch)
            msg = " | ".join([f"{k}: {val_logs[k]:.4f}" for k in val_logs])
            logger.info(f"[val]   epoch {epoch} :: {msg}")

            # --- monitor ---
            metric_name = args.monitor_metric
            if metric_name not in val_logs:
                # fallback: total loss’u minimize et
                metric_name = "total"
            score = float(val_logs[metric_name])

            improved = (score > best_val + EPS) if args.monitor_mode == "max" else (score < best_val - EPS)

            if improved:
                best_val = score
                best_step = global_step
                no_improve_epochs = 0
                # en iyi modeli ayrı etiketle kaydet
                ckpt_path = save_checkpoint(
                    out_dir=out_dir.name,
                    tag=f"step{global_step}",
                    trainer=trainer,
                    args=args,
                    epoch=epoch,
                    step=global_step
                )
                logger.info(f"[ckpt] new BEST {metric_name}={best_val:.4f} @ epoch {epoch} step {global_step}")
                # W&B özetine yazmak istersen:
                if args.wandb and (wandb.run is not None):
                    wandb.summary[f"best/{metric_name}"] = best_val
                    wandb.summary["best/step"] = best_step
                    wandb.summary["best/epoch"] = epoch
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
                out_dir=out_dir.name,
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
        final_path = save_checkpoint(out_dir.name, tag="final", trainer=trainer,
                                     args=args, epoch=epoch, step=global_step)
    except Exception:
        pass

    logger.info("Training finished. Artifacts saved under: %s", args.output_dir)

    # close wandb
    try:
        if wandb.run is not None:
            wandb.run.finish()
    except Exception:
        pass


# ============== YAML parser ==============

def load_structured_cfg(path: str = _TRAINING_CONFIG_DEFAULT):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data = cfg.get("data", {})
    caches = cfg.get("caches", {})
    training = cfg.get("training", {})
    optim = cfg.get("optim", {})
    model = cfg.get("model", {})
    loss = cfg.get("loss", {})
    curriculum = cfg.get("curriculum", {})
    wandb_block = cfg.get("wandb", {})
    sched = cfg.get("schedule", {})
    stores = cfg.get("stores", {})
    general = cfg.get("general", {})
    memory_bank = cfg.get("memory_bank", {})

    args = types.SimpleNamespace(
        # general
        use_queue_miner = bool(general.get("use_queue_miner", True)),
        use_go_memory_bank = bool(general.get("use_go_memory_bank", False)),
        use_attention_pooling = bool(general.get("use_attention_pooling", False)),
        use_lora = bool(general.get("use_lora", False)),
        use_attribution_loss = bool(general.get("use_attribution_loss", False)),
        return_alpha = bool(general.get("return_alpha", False)),
        phase_based = bool(general.get("phase_based", False)),
        pooling_strategy = general.get("pooling_strategy", "mean"),
        ablation_id = general.get("ablation_id", None),
        # paths / store
        train_ids=Path(stores.get("train_ids_path", PROTEIN_TRAIN_IDS)),
        pid2pos=Path(stores.get("pid2pos_path", PID_TO_POSITIVES)),
        val_ids=Path(stores.get("val_ids_path", PROTEIN_VAL_IDS)),
        embed_dir=Path(stores.get("embed_dir", None) or stores.get("hub_local_dir", None)),
        embed_dir_res=Path(stores.get("embed_dir_res", None)),
        embed_dir_fused=Path(stores.get("embed_dir_fused", None)),
        seq_len_lookup=Path(stores.get("seq_len_lookup_dir", P_SEQ_LEN_LOOKUP)),
        pro_manifest=Path(stores.get("protein_manifest_file", GOOGLE_DRIVE_MANIFEST_CACHE)),
        go_text_folder=Path(stores.get("go_text_folder")) if stores.get("go_text_folder") else None,
        hub_local_dir=Path(stores.get("hub_local_dir", None)),

        overlap=data.get("overlap"),
        max_len=data.get("max_len", 1024),
        no_cache_shards=bool(data.get("no_cache_shards", False)),

        # dataset DAG / expansion / few-zero
        use_dag_in_ds=bool(data.get("use_dag_in_ds", False)),
        min_pos_for_expand=int(data.get("min_pos_for_expand", 3)),
        max_ancestor_add=int(data.get("max_ancestor_add", 4)),
        max_hops=int(data.get("max_hops", 3)),
        ancestor_gamma=float(data.get("ancestor_gamma", 0.7)),
        zero_shot_terms=stores.get("zero_shot_terms_file", ZERO_SHOT_TERMS_ID_ONLY_JSON),
        few_shot_terms=stores.get("few_shot_terms_file", FEW_SHOT_IC_TERMS_ID_ONLY_JSON),
        fs_target_ratio=float(data.get("fs_target_ratio", 0.3)),

        # caches (legacy keys kept for compatibility; FAISS yok)
        go_cache=caches.get("go_cache"),

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
        eval_only=bool(training.get("eval_only", False)),
        log_every=int(training.get("log_every", 50)),
        log_level=training.get("log_level", "INFO"),
        monitor_metric=training.get("monitor_metric", "cafa_fmax"),
        monitor_mode=training.get("monitor_mode", "max"),
        early_stop_patience=int(training.get("early_stop_patience", 0)),
        cand_chunk_k=int(training.get("cand_chunk_k", 8)),
        pos_chunk_t=int(training.get("pos_chunk_t", 128)),
        k_hard_queue=int(training.get("k_hard_queue", 128)),
        queue_K=int(training.get("queue_K", 65536)),
        general_device=str(training.get("device", "cuda:0")),

        # optim
        lr=float(optim.get("lr", 3e-4)),
        weight_decay=float(optim.get("weight_decay", 0.01)),
        grad_clip=float(optim.get("grad_clip", 1.0)),

        # model / attention
        align_dim=int(model.get("align_dim", 768)),
        temperature=float(model.get("temperature", 0.07)),
        attn_heads=int(model.get("attn_heads", 2)),
        attn_dropout=float(model.get("attn_dropout", 0.1)),
        entropy_reg=float(model.get("entropy_reg", 0.01)),
        win_size=int(model.get("win_size", 1024)),
        win_stride=int(model.get("win_stride", 256)),

        # loss
        lambda_con=float(loss.get("lambda_con", 1.0)),
        lambda_dag=float(loss.get("lambda_dag", 0.2)),
        lambda_attr=float(loss.get("lambda_attr", 0.1)),
        dag_margin=float(loss.get("dag_margin", 0.05)),
        dag_scale=float(loss.get("dag_scale", 10.0)),
        lambda_vtrue=float(loss.get("lambda_vtrue", 0.2)),
        tau_distill=float(loss.get("tau_distill", 1.5)),

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

        #memory_bank
        memory_bank_device=str(memory_bank.get("device", "cuda")),
        memory_bank_dtype=str(memory_bank.get("d_type", "fp16")),


        # wandb
        wandb=bool(wandb_block.get("enabled", False)),
        wandb_project=wandb_block.get("project", "protein-go-align"),
        wandb_entity=wandb_block.get("entity"),
        wandb_run_name=wandb_block.get("run_name"),
        wandb_mode=wandb_block.get("mode", "online"),

        # misc
        n_go=cfg.get("n_go", None),
    )

    if args.phase_based is False:
        return args, None
    schedule = TrainSchedule(
        phase_breaks=tuple(sched.get("phase_breaks", (5, 12, 25))),
        stageA_mix=tuple(sched.get("stageA_mix", (1.0, 0.0, 0.0))),
        stageB_mix=tuple(sched.get("stageB_mix", (0.7, 0.3, 0.0))),
        stageC_mix=tuple(sched.get("stageC_mix", (0.4, 0.4, 0.2))),
        stageD_mix=tuple(sched.get("stageD_mix", (0.4, 0.4, 0.2))),
        lambda_attr_start=int(sched.get("lambda_attr_start", 6)),
        lambda_attr_max=float(sched.get("lambda_attr_max", 0.2)),
    )
    return args, schedule

def parse_args():
    parser = argparse.ArgumentParser(description="Protein–GO Alignment Training")

    # --- temel ayarlar ---
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file path (örn: src/configs/colab.yaml)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="cuda device")

    args = parser.parse_args()
    if not args.config:
        args.config = TRAINING_CONFIG
        print(f"[main] No --config passed, defaulting to {args.config}")

    args, schedule = load_structured_cfg(args.config)

    if args.ablation_id is not None:
        print("[Ablation] Applying ablation ID:", args.ablation_id)

    return args, schedule

def main():
    args, schedule = parse_args()
    out = Path(args.output_dir)
    setup_logging(out, level=args.log_level)
    set_seed(args.seed)
    t0 = time.time()
    try:
        run_training(args, schedule)
    except Exception as e:
        logging.exception("Fatal error: %s", repr(e))
        raise
    finally:
        logging.info("Total runtime: %.1f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
