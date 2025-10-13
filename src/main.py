# src/main.py
import os
import sys
import time
import math
import yaml
import torch
import random
import logging
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch.multiprocessing as mp
from datetime import datetime
import glob
import types
import signal, traceback, time as _time
import itertools
from huggingface_hub import snapshot_download

import wandb

from src.datasets import ESMResidueStore, GoTextStore, VectorResources, GoMemoryBank
from src.datasets.protein_dataset import ProteinEmbDataset
from src.training.collate import ContrastiveEmbCollator
from src.training.trainer import OppTrainer
from src.configs.data_classes import (
    FewZeroConfig, TrainSchedule, TrainerConfig, AttrConfig, LoRAParameters, TrainingContext, LoggingConfig
)
from src.training.curriculum import CurriculumConfig, CurriculumScheduler
from src.go import GoLookupCache
from src.go import load_go_parents, load_go_children
from src.utils import (
    load_go_set, load_raw_pickle, load_raw_json, load_raw_txt, load_go_texts_by_phase
)
from src.encoders import BioMedBERTEncoder

from src.configs.paths import (
    PROTEIN_TRAIN_IDS,
    PROTEIN_VAL_IDS,
    PID_TO_POSITIVES,
    GOOGLE_DRIVE_ESM3B_EMBEDDINGS,
    ZERO_SHOT_TERMS_ID_ONLY_JSON,
    FEW_SHOT_IC_TERMS_ID_ONLY_JSON,
    P_SEQ_LEN_LOOKUP,
    GOOGLE_DRIVE_MANIFEST_CACHE,
    TRAINING_CONFIG,
    GO_ANCESTOR_STOPLIST,
    COMMON_IC_GO_TERMS_ID_ONLY_JSON,
)

# To prevent sigint (potential cause)
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # çok önemli
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ============== Utilities ==============

def _sigint_handler(signum, frame):
    print(f"\n[DBG] Caught SIGINT at {_time.strftime('%H:%M:%S')}")
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


def save_checkpoint(output_dir: Path, tag: str, trainer, args, epoch: int, step: int, extra: dict = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(output_dir, tag)
    state = {
        "model": trainer.model.state_dict(),
        "optimizer": trainer.opt.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "global_step": step,
    }
    if hasattr(trainer, "scaler") and trainer.scaler is not None:
        state["scaler"] = trainer.scaler.state_dict()
    if hasattr(trainer, "sched") and trainer.sched is not None:
        try:
            state["lr_scheduler"] = trainer.sched.state_dict()
        except Exception:
            pass
    if extra:
        state["extra"] = extra
    torch.save(state, path)
    logging.getLogger("ckpt").info("Saved checkpoint: %s", str(path))
    return path


def cleanup_old_checkpoints(output_dir: Path, keep_last_n: int = 3):
    cks = sorted(glob.glob(str(output_dir / "ckpt_*.pt")))
    if keep_last_n is not None and len(cks) > keep_last_n:
        for p in cks[:-keep_last_n]:
            try:
                os.remove(p)
                logging.getLogger("ckpt").info("Removed old checkpoint: %s", p)
            except OSError:
                pass


# ============== Builders ==============
def build_go_cache(go_cache_path: str) -> GoLookupCache:
    logger = logging.getLogger("build_go_cache")
    logger.info("Loading GO cache: %s", go_cache_path)
    blob = torch.load(go_cache_path, map_location="cpu")
    blob['embs'] = F.normalize(blob['embs'], p=2, dim=1)
    cache = GoLookupCache(blob)
    return cache


def build_store(args) -> ESMResidueStore:
    logger = logging.getLogger("build_store")
    logger.info("Building store...")
    seq_len_lookup = load_raw_pickle(args.seq_len_lookup)
    local_dir = snapshot_download(
        repo_id="secilozksen/esm-embeddings",
        repo_type="dataset",
        revision="main",
        allow_patterns=["*.pt", "*.json", "*.faiss"]  # yapına göre
    )
    store = ESMResidueStore(
        embed_dir=args.embed_dir,
        seq_len_lookup=seq_len_lookup,
        max_len=args.max_len,
        overlap=args.overlap,
        cache_shards=(not args.no_cache_shards),
        pro_manifest_file=args.pro_manifest if args.pro_manifest else None,
    )
    return store


def build_datasets(args, store: ESMResidueStore, go_cache: GoLookupCache) -> Dict[str, torch.utils.data.Dataset]:
    logger = logging.getLogger("build_datasets")
    logger.info("Building datasets...")

    pid2pos = load_raw_json(PID_TO_POSITIVES)
    train_ids = load_raw_txt(PROTEIN_TRAIN_IDS)
    val_ids = load_raw_txt(PROTEIN_VAL_IDS)

    zs = load_go_set(ZERO_SHOT_TERMS_ID_ONLY_JSON)
    fs = load_go_set(FEW_SHOT_IC_TERMS_ID_ONLY_JSON)
    common = load_go_set(COMMON_IC_GO_TERMS_ID_ONLY_JSON)
    fz = FewZeroConfig(zero_shot_terms=zs, few_shot_terms=fs, common_terms=common,
                       fs_target_ratio=args.fs_target_ratio)

    dag_parents = load_go_parents()
    ancestor_stoplist = load_raw_txt(GO_ANCESTOR_STOPLIST)

    ds_kwargs = dict(
        pid2pos=pid2pos,
        go_cache=go_cache,
        fewzero=fz,
        dag_parents=dag_parents if args.use_dag_in_ds else None,
        min_pos_for_expand=args.min_pos_for_expand,
        max_ancestor_add=args.max_ancestor_add,
        max_hops=args.max_hops,
        ancestor_gamma=args.ancestor_gamma,
        ancestor_stoplist=ancestor_stoplist,
        store=store,
    )

    train_ds = ProteinEmbDataset(protein_ids=train_ids, **ds_kwargs)
    val_ds = ProteinEmbDataset(protein_ids=val_ids, **ds_kwargs) if val_ids else None
    logger.info("Datasets ready. Train=%d%s", len(train_ds), f", Val={len(val_ds)}" if val_ds else "")
    return {"train": train_ds, "val": val_ds}


def build_dataloaders(datasets, args, go_cache: GoLookupCache, go_text_store: GoTextStore):
    def _worker_init(_):
        import os, random, torch
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(1)
        random.seed(torch.initial_seed() % 2 ** 32)
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
        go_text_store=go_text_store,
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
        worker_init_fn=_worker_init,
    #    prefetch_factor=2,
        collate_fn=collate,
    )
    val_loader = None
    if datasets.get("val") is not None:
        val_loader = DataLoader(
            datasets["val"],
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            pin_memory=True,
            collate_fn=collate,
#            multiprocessing_context="forkserver",
            worker_init_fn=_worker_init,
            drop_last=False
        )
    logger.info("Dataloaders ready. batch_size=%d", args.batch_size)
    return train_loader, val_loader, collate


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


# ============== W&B helpers ==============
def build_wandb_config(args, schedule: TrainSchedule) -> Dict[str, Any]:
    return {
        "data": {
            "train_ids": str(args.train_ids),
            "val_ids": str(args.val_ids),
            "embed_dir": str(args.embed_dir),
            "pid2pos": str(args.pid2pos),
            "zero_shot_terms": str(args.zero_shot_terms),
            "few_shot_terms": str(args.few_shot_terms),
            "seq_len_lookup": str(args.seq_len_lookup),
            "overlap": args.overlap,
            "pro_manifest": str(args.pro_manifest),
        },
        "schedule": {
            "phase_breaks": list(schedule.phase_breaks),
            "stageA_mix": list(schedule.stageA_mix),
            "stageB_mix": list(schedule.stageB_mix),
            "stageC_mix": list(schedule.stageC_mix),
            "stageD_mix": list(schedule.stageD_mix),
            "lambda_attr_start": schedule.lambda_attr_start,
            "lambda_attr_max": schedule.lambda_attr_max,
        },
    }


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
    table = wandb_mod.Table(columns=["step"] + list(curves.keys()))
    for i, s in enumerate(steps):
        row = [int(s)] + [curves[k][i] for k in curves.keys()]
        table.add_data(*row)
    wandb_mod.log({"curriculum/preview": table})


def wandb_dataset_quickstats(wandb_mod, train_ds, sample_n: int = 512):
    try:
        pos_counts: List[int] = []
        if hasattr(train_ds, "protein_ids") and hasattr(train_ds, "pid2pos"):
            for pid in train_ds.protein_ids[:sample_n]:
                pos = train_ds.pid2pos.get(pid, [])
                pos_counts.append(len(pos))
        if pos_counts:
            wandb_mod.log({"data/positives_per_protein": wandb_mod.Histogram(np.array(pos_counts))})
            wandb_mod.summary["data/positives_per_protein_mean"] = float(np.mean(pos_counts))
            wandb_mod.summary["data/positives_per_protein_p95"] = float(np.percentile(pos_counts, 95))

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
            wandb_mod.log({"data/lengths": wandb_mod.Histogram(np.array(lengths))})
            wandb_mod.summary["data/length_mean"] = float(np.mean(lengths))
            wandb_mod.summary["data/length_p95"] = float(np.percentile(lengths, 95))
    except Exception as e:
        logging.getLogger("wandb").warning("Dataset quickstats failed: %r", e)


# ============== Runner ==============
def run_training(args, schedule: TrainSchedule):
    signal.signal(signal.SIGINT, _sigint_handler)
    logger = logging.getLogger("main")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info("Device: %s", device)
    wandb.login()

    # Phase 0 resources
    phase0 = 0
    go_cache_path = schedule.resolve_go_cache_path(phase0)

    go_cache = build_go_cache(go_cache_path)
    dag_parents = load_go_parents()
    dag_children = load_go_children()

    # GO text dict per phase
    total_phases = (len(schedule.phase_breaks) + 1) if hasattr(schedule, "phase_breaks") else 1
    go_id_to_text: Dict[int, Dict[int, str]] = {}
    for ph in range(total_phases):
        go_id_to_text[ph] = load_go_texts_by_phase(args.go_text_folder, phase=ph)

    store = build_store(args)
    datasets = build_datasets(args, store, go_cache)
    n_spe = steps_per_epoch(len(datasets["train"]), args.batch_size)

    # Text encoder (GO)
    lora_params = LoRAParameters(adapter_name="go_encoder")
    go_encoder = BioMedBERTEncoder(
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        device=device,
        max_length=512,
        use_attention_pool=True,
        attn_hidden=128,
        attn_dropout=0.1,
        special_token_weights=None,
        enable_lora=True,
        lora_parameters=lora_params,
    )

    # GoTextStore + dataloaders
    go_text_store = GoTextStore(go_id_to_text, go_encoder.tokenizer, phase=phase0, lazy=True)
    go_text_store.materialize_tokens_once(batch_size=512, show_progress=True)
    train_loader, val_loader, collate = build_dataloaders(datasets, args, go_cache, go_text_store)

    # Memory bank (initial GO cache)
    memory_bank = None
    if args.use_go_memory_bank:
        logger.info("Using GoMemoryBank for training (CPU fp16).")
        embs_cpu16 = go_cache.embs.half().cpu()
        memory_bank = GoMemoryBank(
            init_embs=embs_cpu16,
            row2id=getattr(go_cache, 'row2id', None)
        )
   #     logger.info("Using GoMemoryBank for training.")
   #     memory_bank = GoMemoryBank(init_embs=go_cache.embs, row2id=getattr(go_cache, 'row2id', None))

    seen_go_ids_prev: set = set()

    if args.wandb:
        try:
            wandb.config.update({'go_encoder_enabled': True}, allow_val_change=True)
        except Exception:
            pass

    cur_cfg = build_scheduler_cfg(args, n_spe)
    scheduler = CurriculumScheduler(cur_cfg)

    # Vector resources — FAISS YOK: sadece bank embs ile
    vres = VectorResources(faiss_index=None, go_embs=go_cache.embs)

    out_dir = Path(args.output_dir)

    # Lightweight runtime context
    training_context = TrainingContext(
        device=device,
        schedule=schedule,
        go_cache=go_cache,
        faiss_index=None,              # FAISS kaldırıldı
        vres=vres,
        memory_bank=memory_bank,
        current_phase=None,
        last_refresh_epoch=None,
        last_refresh_reason=None,
        batch_builder=None,            # FAISS akışı yok
        maybe_refresh_phase_resources=None,
        dag_parents=dag_parents,
        dag_children=dag_children,
        scheduler=scheduler,
        go_text_store=go_text_store,
        use_queue_miner=True
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

    # VectorResources backend’i MemoryBank ile hizala (FAISS yok)
  #  try:
  #      training_context.vres.set_backends(None, training_context.memory_bank.embs)
  #  except Exception:
  #      pass

    # VectorResources backend seçimi (FAISS yok)
    if training_context.memory_bank is not None:
        training_context.vres.set_backends(None, training_context.memory_bank.embs)  # CPU bank
    else:
        training_context.vres.set_backends(None, training_context.go_cache.embs)  # fallback: cache embs

    def maybe_refresh_phase_resources(current_epoch: int, *, force: bool = False):
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

            # MemoryBank’i yeni cache ile yenile (CPU + fp16)
            if args.use_go_memory_bank:
                try:
                    embs_cpu16 = new_go_cache.embs.half().cpu()
                    training_context.memory_bank = GoMemoryBank(
                        init_embs=embs_cpu16,
                        row2id=getattr(new_go_cache, 'row2id', None)
                    )
                except Exception:
                    training_context.memory_bank = None
            else:
                training_context.memory_bank = None

            # Backend seçim
            if training_context.memory_bank is not None:
                training_context.vres.set_backends(None, training_context.memory_bank.embs)
            else:
                training_context.vres.set_backends(None, training_context.go_cache.embs)

            # GoTextStore fazı
            training_context.go_text_store.update_phase_and_tokenize(new_phase)

            # Dataloader’ları yeniden kur
            nonlocal train_loader, val_loader, collate
            train_loader, val_loader, collate = build_dataloaders(
                datasets, args, training_context.go_cache, training_context.go_text_store)

            training_context.current_phase = new_phase
            training_context.last_refresh_epoch = current_epoch
            training_context.last_refresh_reason = "phase_change" if not force else "force"

    # expose refresher
    training_context.maybe_refresh_phase_resources = maybe_refresh_phase_resources

    # W&B
  #  wandb_run = None
  #  wandb_logger = None
  #  if args.wandb:
  #      os.environ.setdefault("WANDB_MODE", args.wandb_mode)
  #      cfg = build_wandb_config(args, schedule)
  #      wandb_run = wandb.init(project=args.wandb_project,
  #                             entity=args.wandb_entity or None,
  #                             name=args.wandb_run_name or None,
    #                           config=cfg,
    #                           dir=args.output_dir)
    #    wandb.define_metric("global_step")
    #    wandb.define_metric("*", step_metric="global_step")
        # İstersen aç:
        # wandb_preview_curriculum(wandb, args, total_steps=n_spe * args.epochs)
        # wandb_dataset_quickstats(wandb, datasets["train"], sample_n=512)

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
    trainer = OppTrainer(cfg=trainer_cfg, attr=attr_cfg, ctx=training_context, go_encoder=go_encoder)

    # -------------------------   Training loop -------------------------
    logger.info("Start training for %d epochs", args.epochs)
    training_context.maybe_refresh_phase_resources(current_epoch=0, force=False)

    best_val = None
    global_step = 0

    # (opsiyonel) dataloader hız ping
    t0 = time.time()
    for i, b in enumerate(itertools.islice(train_loader, 5)):
        dt = time.time() - t0
        print(f"[DL] batch {i} hazırlandı: {dt:.2f}s")
        t0 = time.time()

    for epoch in range(args.epochs):
        # ---- Partial MemoryBank refresh using GO ids seen in the previous epoch ----
        try:
            if wandb_run is not None and global_step == 1:
                try:
                    wandb_preview_curriculum(wandb, args, total_steps=n_spe * args.epochs)
                    wandb_dataset_quickstats(wandb, datasets["train"], sample_n=256)
                except Exception as e:
                    logging.getLogger("wandb").warning("deferred previews failed: %r", e)
            if len(seen_go_ids_prev) > 0:
                ids_to_update = sorted(set(int(i) for i in seen_go_ids_prev))
                toks = go_text_store.batch(ids_to_update)
                with torch.no_grad():
                    new_embs = go_encoder(input_ids=toks['input_ids'].to(device),
                                          attention_mask=toks['attention_mask'].to(device)).detach().cpu()
                new_embs = F.normalize(new_embs, p=2, dim=1).cpu()
                training_context.memory_bank.update(ids_to_update, new_embs)
                # Backend’i güncelle (sadece bank embs)
                try:
                    training_context.vres.set_backends(None, training_context.memory_bank.embs)
                except Exception:
                    pass
            seen_go_ids = set()
        except Exception as _e:
            logging.getLogger('bank').warning('Partial refresh failed: %r', _e)

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
            losses = trainer.step_losses(batch, epoch)
            loss = losses["total"]

            # backward
            trainer.opt.zero_grad(set_to_none=True)
            loss.backward()

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
                path = save_checkpoint(out_dir, tag=f"step{global_step}",
                                       trainer=trainer, args=args,
                                       epoch=epoch, step=global_step)
                cleanup_old_checkpoints(out_dir, keep_last_n=args.keep_last_n)
                # (opsiyonel) wandb artifact

        # epoch finished - update ema
        trainer.update_ema(trainer.index_projector, trainer.model.proj_p, m=0.995)

        # validation
        if val_loader is not None:
            val_logs = trainer.eval_epoch(val_loader, epoch)
            msg = " | ".join([f"{k}: {val_logs[k]:.4f}" for k in val_logs])
            logger.info(f"[val]   epoch {epoch} :: {msg}")

        # Persist seen GO ids for next epoch's partial refresh
        try:
            seen_go_ids_prev = set(seen_go_ids)
        except Exception:
            pass

        # epoch checkpoint
        try:
            path = save_checkpoint(out_dir, tag=f"epoch{epoch + 1}", trainer=trainer,
                                   args=args, epoch=epoch, step=global_step)
            cleanup_old_checkpoints(out_dir, keep_last_n=args.keep_last_n)
        except Exception:
            pass

    # final checkpoint
    try:
        final_path = save_checkpoint(out_dir, tag="final", trainer=trainer,
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
from src.configs.paths import TRAINING_CONFIG as _TRAINING_CONFIG_DEFAULT
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

    args = types.SimpleNamespace(
        # general
        use_queue_miner = bool(general.get("use_queue_miner", True)),
        use_go_memory_bank = bool(general.get("use_go_memory_bank", False)),
        use_faiss = bool(general.get("use_faiss", False)),
        # paths / store
        train_ids=Path(stores.get("train_ids_path", PROTEIN_TRAIN_IDS)),
        pid2pos=Path(stores.get("pid2pos_path", PID_TO_POSITIVES)),
        val_ids=Path(stores.get("val_ids_path", PROTEIN_VAL_IDS)),
        embed_dir=Path(stores.get("embed_dir", GOOGLE_DRIVE_ESM3B_EMBEDDINGS)),
        seq_len_lookup=Path(stores.get("seq_len_lookup_dir", P_SEQ_LEN_LOOKUP)),
        pro_manifest=Path(stores.get("protein_manifest_file", GOOGLE_DRIVE_MANIFEST_CACHE)),
        go_text_folder=Path(stores.get("go_text_folder")) if stores.get("go_text_folder") else None,

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
        fp16=bool(training.get("fp16", False)),
        cpu=bool(training.get("cpu", False)),
        seed=int(training.get("seed", 42)),
        output_dir=training.get("output_dir", "outputs/run1"),
        save_every=int(training.get("save_every", 1000)),
        keep_last_n=int(training.get("keep_last_n", 3)),
        resume=training.get("resume"),
        log_every=int(training.get("log_every", 50)),
        log_level=training.get("log_level", "INFO"),

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


        # wandb
        wandb=bool(wandb_block.get("enabled", False)),
        wandb_project=wandb_block.get("project", "protein-go-align"),
        wandb_entity=wandb_block.get("entity"),
        wandb_run_name=wandb_block.get("run_name"),
        wandb_mode=wandb_block.get("mode", "online"),

        # misc
        n_go=cfg.get("n_go", None),
    )

    # ---- TrainSchedule from 'schedule' block ----
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


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args, schedule = load_structured_cfg(TRAINING_CONFIG)
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
