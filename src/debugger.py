"""
Fast smoke test and robustness checks for the training pipeline.

Purpose
-------
Quickly validate the simplified "trainer owns BatchBuilder" flow across the chain:
data → index → negative mining → candidate assembly → forward pass (losses),
without running a full training session.

What it does (summary)
----------------------
1) GO Cache: Loads embeddings and asserts L2-normalization.
2) FAISS: Builds index and checks that `ntotal == G` (if available).
3) Store/Datasets: Prepares datasets; optionally downsamples train for speed.
4) GO Text Encoder + GoTextStore: Initializes tokenizer-compatible store.
5) DataLoader: Verifies collate invariants (shapes, dtypes, required keys).
6) Curriculum + VectorResources: Sets up schedule/resources.
7) Trainer (one batch):
   - Attach EMA projector (frozen copy of trainer.model.proj_p) to VectorResources.
   - ANN probe after projector attach.
   - Negative mining outputs: shapes, padding ratio (-1 pads).
   - Positive leakage and ZS filter checks.
   - Loss finiteness (no NaN/Inf).
8) MemoryBank refresh: Update a small slice and rebuild FAISS as a smoke test.

CLI arguments
-------------
--config       : Path to YAML training config (default: TRAINING_CONFIG)
--cpu          : Force CPU mode (faster for debug)
--max-train    : Upper limit on train size for quick runs (default: 32)
--log-level    : "INFO" | "DEBUG" (default: INFO)

Exit codes
----------
- Non-zero on assertion failure.
- Zero on success, ends with “Debugger finished OK”.
"""

import argparse
import logging
import time
import gc
from pathlib import Path
from types import SimpleNamespace
import random
import torch
import torch.nn.functional as F

from encoders import BioMedBERTEncoder
from main import (
    TRAINING_CONFIG,
    setup_logging,
    set_seed,
    steps_per_epoch,
    rebuild_faiss_from_bank,
    load_go_texts_by_phase,
    build_go_cache,
    build_faiss,
    build_store,
    build_datasets,
    build_dataloaders,
    build_scheduler_cfg,
    load_structured_cfg,
    TrainSchedule,
    TrainerConfig,
    AttrConfig,
    CurriculumScheduler,
    OppTrainer,
    GoTextStore,
    VectorResources,
    GoMemoryBank,
)


def human_bytes(n: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{x:.1f} EB"


def cuda_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    return 0, 0


class Timer:
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger
        self.t0 = None
        self.cuda0 = cuda_mem()

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        a1, r1 = cuda_mem()
        a0, r0 = self.cuda0
        if a1 or r1:
            self.logger.info(
                f"[time] {self.name}: {dt:.3f}s, cuda Δalloc={human_bytes(a1 - a0)}, Δres={human_bytes(r1 - r0)}"
            )
        else:
            self.logger.info(f"[time] {self.name}: {dt:.3f}s")


def build_light_args(args, schedule: TrainSchedule):
    """Keep runs tiny & CPU-first for quick debug."""
    args.cpu = True
    args.batch_size = max(1, min(2, int(getattr(args, "batch_size", 4) // 2)))
    args.eval_batch_size = args.batch_size
    args.num_workers = 0
    args.log_every = 1
    args.epochs = 1
    # curriculum squeeze
    args.curriculum_epochs = max(1, int(getattr(args, "curriculum_epochs", 4)))
    args.shortlist_M_start = min(64, getattr(args, "shortlist_M_start", 256))
    args.shortlist_M_end = min(64, getattr(args, "shortlist_M_end", 256))
    args.k_hard_start = min(4, getattr(args, "k_hard_start", 16))
    args.k_hard_end = min(4, getattr(args, "k_hard_end", 16))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args


def assert_normed(v: torch.Tensor, eps: float = 1e-3, msg: str = "vectors not L2-normalized"):
    with torch.no_grad():
        n = v[0].norm(p=2).item()
        assert abs(n - 1.0) < eps, f"{msg}: {n:.6f}"


def check_lora_attached(encoder, logger: logging.Logger, name="encoder",
                        expect_attached: bool = True, strict: bool = False,
                        allow_nonlora=("bias", "norm", "layernorm")):
    """Verify LoRA adapters are present and (optionally) only LoRA weights are trainable."""
    try:
        base = getattr(encoder, "model", encoder)
    except Exception:
        base = encoder

    # Try PEFT signals
    peft_present = False
    adapters = []
    try:
        from peft import PeftModel  # type: ignore

        peft_present = isinstance(base, PeftModel) or hasattr(base, "peft_config")
        if hasattr(base, "peft_config") and isinstance(base.peft_config, dict):
            adapters = list(base.peft_config.keys())
    except Exception:
        peft_present = hasattr(base, "peft_config")

    # Name-based LoRA detection
    has_lora_params = any(("lora_" in n.lower()) or (".lora" in n.lower()) for n, _ in base.named_parameters())
    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    trainable_lora = sum(
        p.numel() for n, p in base.named_parameters()
        if p.requires_grad and (("lora_" in n.lower()) or (".lora" in n.lower()))
    )
    trainable_nonlora_names = [
        n for n, p in base.named_parameters()
        if p.requires_grad and ("lora_" not in n.lower()) and (".lora" not in n.lower())
        and not any(tok in n.lower() for tok in allow_nonlora)
    ]
    logger.info(f"[LoRA] {name}: peft_present={peft_present}, adapters={adapters if adapters else '-'}")
    pct = (100.0 * trainable_lora / max(1, trainable))
    logger.info(f"[LoRA] {name}: params total/trainable = {total:,}/{trainable:,} | trainable_lora = {trainable_lora:,} ({pct:.1f}%)")

    if expect_attached:
        assert has_lora_params or peft_present, f"[LoRA] {name}: No LoRA modules/adapters detected"
        assert trainable_lora > 0, f"[LoRA] {name}: No trainable LoRA params"
    if strict:
        assert len(trainable_nonlora_names) == 0, f"[LoRA] {name}: Non-LoRA trainables found: {trainable_nonlora_names[:10]}"
    else:
        if trainable_nonlora_names:
            logger.warning(f"[LoRA] {name}: Non-LoRA trainables present (allowed or intentional?): sample={trainable_nonlora_names[:6]}")

def dbg_check_bijection(go_cache, mem_bank) -> bool:
    """
    GO evreni aynı mı? (global id kümeleri birebir mi)
    """
    logger = logging.getLogger("debugger")
    cache_ids = set(int(g) for g in go_cache.row2id.tolist())
    bank_ids  = set(int(g) for g in mem_bank.id2row.keys())

    only_cache = cache_ids - bank_ids
    only_bank  = bank_ids  - cache_ids

    logger.info("[bijection] cache=%d, bank=%d, only_cache=%d, only_bank=%d",
                len(cache_ids), len(bank_ids), len(only_cache), len(only_bank))

    if only_cache:
        logger.warning("[bijection] Missing in MemoryBank (sample): %s",
                       list(sorted(only_cache))[:10])
    if only_bank:
        logger.warning("[bijection] Extra in MemoryBank (sample): %s",
                       list(sorted(only_bank))[:10])

    ok = (len(only_cache) == 0 and len(only_bank) == 0)
    if not ok:
        logger.error("[bijection] Universe mismatch! Fix row2id/id2row sources.")
    return ok


def dbg_check_order_alignment(go_cache, mem_bank, sample: int = 1000) -> bool:
    """
    Sıra hizası (local row) aynı mı? (zorunlu değil; ama uyarı verir)
    """
    logger = logging.getLogger("debugger")
    row2id = go_cache.row2id.tolist()
    if not row2id:
        logger.error("[order] go_cache.row2id empty!")
        return False

    N = len(row2id)
    picks = random.sample(range(N), min(sample, N))
    mismatches = 0
    for r in picks:
        g = int(row2id[r])
        r_bank = mem_bank.id2row.get(g, -1)
        if r_bank != r:
            mismatches += 1
            if mismatches <= 10:
                logger.warning("[order] row mismatch: g=%d cache_row=%d bank_row=%d", g, r, r_bank)
    if mismatches == 0:
        logger.info("[order] OK (no mismatches in %d samples)", len(picks))
        return True
    else:
        ratio = mismatches / len(picks)
        logger.warning("[order] %d/%d mismatches (%.1f%%). Not fatal unless you assume same row ordering.",
                       mismatches, len(picks), 100*ratio)
        return False


def dbg_spot_check_vectors(go_cache, mem_bank, n: int = 10):
    """
    Aynı ID için cache vs bank vektör benzerliği (EMA/güncelleme var mı?)
    """
    logger = logging.getLogger("debugger")
    ids_all = go_cache.row2id.tolist()
    if len(ids_all) == 0:
        logger.error("[vec-check] empty row2id")
        return
    picks = random.sample(ids_all, min(n, len(ids_all)))
    try:
        cache_vecs = go_cache(picks).cpu()                 # [n, d]
        bank_vecs  = mem_bank.lookup(picks)                # [n, d]  (MemoryBank CPU)
        if cache_vecs.numel() == 0 or bank_vecs.numel() == 0:
            logger.warning("[vec-check] empty vecs; picks=%d", len(picks))
            return
        cache_vecs = F.normalize(cache_vecs, dim=1)
        bank_vecs  = F.normalize(bank_vecs,  dim=1)
        cos = F.cosine_similarity(cache_vecs, bank_vecs, dim=1)
        logger.info("[vec-check] mean cos=%.4f  min=%.4f  max=%.4f",
                    float(cos.mean()), float(cos.min()), float(cos.max()))
    except KeyError as e:
        logger.error("[vec-check] KeyError (MemoryBank missing id). Run bijection check first. err=%s", e)

def dbg_check_faiss_ids_against_universe(go_cache, faiss_ids: torch.Tensor):
    """
    FAISS’te kayıtlı id’ler (global) go_cache evreninde mi?
    """
    logger = logging.getLogger("debugger")
    cache_ids = set(int(g) for g in go_cache.row2id.tolist())
    faiss_ids = [int(x) for x in faiss_ids.tolist()]
    missing = [g for g in faiss_ids if g not in cache_ids]
    if missing:
        logger.error("[faiss] %d ids not in go_cache universe (sample: %s)",
                     len(missing), missing[:10])
    else:
        logger.info("[faiss] all ids are in go_cache universe (%d)", len(faiss_ids))






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=TRAINING_CONFIG)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-train", type=int, default=32)
    ap.add_argument("--log-level", type=str, default="INFO")
    args_cli = ap.parse_args()

    # Config & logging
    args, schedule = load_structured_cfg(args_cli.config)
    out = Path(getattr(args, "output_dir", "outputs/debug"))
    setup_logging(out, level=args_cli.log_level)
    log = logging.getLogger("dbg")
    set_seed(getattr(args, "seed", 42))

    # Light overrides
    args = build_light_args(args, schedule)
    if args_cli.cpu:
        args.cpu = True
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    log.info("Device: %s", device)

    # 1) GO cache
    with Timer("build_go_cache", log):
        phase0 = 0
        gc_path = schedule.resolve_go_cache_path(phase0)
        go_cache = build_go_cache(gc_path)
        log.info("go_cache.embs: %s", tuple(go_cache.embs.shape))
        assert_normed(go_cache.embs, msg="GO cache embs not L2")

    # 2) FAISS index
    with Timer("build_faiss", log):
        faiss_index = build_faiss(phase=phase0)
        ntotal = getattr(faiss_index, "ntotal", None)
        log.info("faiss.ntotal=%s", str(ntotal))
        if ntotal is not None:
            assert ntotal == go_cache.embs.shape[0], f"FAISS ntotal {ntotal} != G {go_cache.embs.shape[0]}"

    # 3) Store & datasets
    with Timer("build_store_datasets", log):
        store = build_store(args)
        datasets = build_datasets(args, store, go_cache)
        ntr = len(datasets["train"])
        if ntr > args_cli.max_train and hasattr(datasets["train"], "subset"):
            datasets["train"] = datasets["train"].subset(range(args_cli.max_train))
            log.info("train subset -> %d", len(datasets["train"]))
        val_info = f", val={len(datasets['val'])}" if datasets.get("val") else ""
        log.info("datasets: train=%d%s", len(datasets["train"]), val_info)

    # 4) GO text encoder + store
    with Timer("go_text_encoder_store", log):
        from configs.data_classes import LoRAParameters  # type: ignore

        lora = LoRAParameters(adapter_name="go_encoder")
        go_encoder = BioMedBERTEncoder(
            model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            device=device,
            max_length=512,
            use_attention_pool=True,
            attn_hidden=128,
            attn_dropout=0.1,
            special_token_weights=None,
            enable_lora=True,
            lora_parameters=lora,
        )
        total_phases = (len(schedule.phase_breaks) + 1) if hasattr(schedule, "phase_breaks") else 1
        go_id_to_text = {ph: load_go_texts_by_phase(str(getattr(args, "go_text_folder", "")), phase=ph) for ph in range(total_phases)}
        go_text_store = GoTextStore(go_id_to_text, go_encoder.tokenizer, phase=phase0)

        # LoRA check (optional: set strict=True if only LoRA should be trainable)
        try:
            check_lora_attached(go_encoder, log, name="go_encoder", expect_attached=True, strict=False)
        except Exception as e:
            log.warning("[LoRA] go_encoder check skipped/failed: %r", e)

    # 5) DataLoaders (+ collate invariants)
    with Timer("build_dataloaders", log):
        train_loader, val_loader, collate = build_dataloaders(datasets, args, go_cache, go_text_store)
        log.info("loader: batch_size=%d, workers=%d", args.batch_size, args.num_workers)
        tmp_batch = next(iter(train_loader))
        H = tmp_batch["prot_emb_pad"]
        Mv = tmp_batch["prot_attn_mask"]
        assert H.shape[:2] == Mv.shape, f"pad vs mask mismatch: {H.shape} vs {Mv.shape}"
        assert H.ndim == 3 and Mv.dtype == torch.bool, "H must be [B,L,D], mask torch.bool"
        assert "uniq_go_ids" in tmp_batch and isinstance(tmp_batch["pos_go_local"], list), "uniq_go_ids/pos_go_local missing"
        if "zs_mask" in tmp_batch:
            zs = tmp_batch["zs_mask"]
            assert zs.ndim in (1, 2), f"zs_mask shape bad: {zs.shape}"

    # 6) Scheduler + runtime resources
    with Timer("scheduler_vec_resources", log):
        n_spe = steps_per_epoch(len(datasets["train"]), args.batch_size)
        cur_cfg = build_scheduler_cfg(args, n_spe)
        scheduler = CurriculumScheduler(cur_cfg)
        memory_bank = GoMemoryBank(init_embs=go_cache.embs, row2id=getattr(go_cache, "row2id", None))
        vres = VectorResources(faiss_index, go_cache.embs)
        try:
            vres.set_backends(faiss_index, memory_bank.embs)
        except Exception:
            vres.set_backends(faiss_index, go_cache.embs)
        log.info("curriculum: steps/epoch=%d, shortlist_M=%s, k_hard=%s", n_spe, cur_cfg.shortlist_M, cur_cfg.k_hard)

    # 7) Trainer + one batch step (attach EMA projector, probe ANN, then losses)
    with Timer("trainer_step_with_checks", log):
        sample_item = datasets["train"][0]
        d_h = int(sample_item["prot_emb"].shape[1])
        d_g = int(go_cache.embs.shape[1])
        d_z = d_g

        trainer_cfg = TrainerConfig(d_h=d_h, d_g=d_g, d_z=d_z, device=str(device), lr=getattr(args, "lr", 3e-4), max_epochs=1)
        attr_cfg = AttrConfig(
            lambda_attr=getattr(args, "lambda_attr", 0.1),
            lambda_entropy_alpha=getattr(args, "lambda_entropy_alpha", 0.05),
            lambda_entropy_window=getattr(args, "lambda_entropy_window", 0.01),
            topk_per_window=int(getattr(args, "topk_per_window", 64)),
            curriculum_epochs=int(getattr(args, "curriculum_epochs", 4)),
            temperature=float(getattr(args, "temperature", 0.07)),
        )

        ctx = SimpleNamespace(
            device=device,
            schedule=schedule,
            go_cache=go_cache,
            faiss_index=faiss_index,
            vres=vres,
            memory_bank=memory_bank,
            current_phase=0,
            last_refresh_epoch=0,
            last_refresh_reason="init",
            batch_builder=None,  # trainer owns builder internally
            maybe_refresh_phase_resources=lambda *a, **k: None,
            dag_parents=None,
            dag_children=None,
            scheduler=scheduler,
            align_dim=getattr(vres, "align_dim", d_g),
        )

        trainer = OppTrainer(
            cfg=trainer_cfg,
            attr=attr_cfg,
            ctx=ctx,
            go_encoder=BioMedBERTEncoder(
                model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                device=device,
                max_length=512,
                use_attention_pool=True,
                attn_hidden=128,
                attn_dropout=0.1,
                special_token_weights=None,
                enable_lora=False,
                lora_parameters=None,
            ),
        )
        trainer.model.to(device)

        # === Attach EMA projector to VectorResources and run ANN probe ===
        import copy

        if hasattr(trainer.model, "proj_p"):
            ema_proj = copy.deepcopy(trainer.model.proj_p).eval()
            for p in ema_proj.parameters():
                p.requires_grad = False
            vres.query_projector = ema_proj.to(device)
            log.info(
                "[EMA] Attached EMA projector on %s",
                str(next(ema_proj.parameters()).device),
            )
        else:
            log.warning("[EMA] trainer.model.proj_p not found; skipping EMA projector attach")

        # ANN probe after projector attach
        try:
            item0 = datasets["train"][0]
            H0 = item0["prot_emb"].unsqueeze(0)  # [1, L, D]
            m0 = torch.ones(H0.size(1), dtype=torch.bool).unsqueeze(0)
            q_raw = vres.coarse_prot_vecs(H0, m0)
            q = vres.project_queries_to_index(q_raw)
            assert q.shape[1] == vres.align_dim, "projection mismatch (after EMA attach)"
            if hasattr(vres, "query"):
                I = vres.query(q, topM=8)
            else:
                _, I = vres.coarse_search(q, topM=8)
            assert I.ndim == 2 and I.size(1) == 8, "ANN probe failed after EMA attach"
            log.info("ANN probe ok after EMA attach: top-idx shape=%s", tuple(I.shape))
        except Exception as e:
            log.warning("ANN probe after EMA attach failed: %r", e)

        # Pull a batch and move to device
        batch = next(iter(train_loader))

        def to_dev(x):
            return x.to(device, non_blocking=False) if isinstance(x, torch.Tensor) else x

        for k in list(batch.keys()):
            try:
                batch[k] = to_dev(batch[k])
            except Exception:
                pass

        # Negative mining outputs (via trainer's internal builder)
        if hasattr(trainer, "_build_negatives"):
            mined = trainer._build_negatives(batch)
            neg = mined["neg_go_ids"]
            cand = mined["cand_ids"]
            assert neg.ndim == 2 and cand.ndim == 2, "mined shapes bad"
            pad_ratio = (neg == -1).float().mean().item()
            log.info("mined: neg.shape=%s, cand.shape=%s, pad_ratio=%.3f", tuple(neg.shape), tuple(cand.shape), pad_ratio)

            # Positive leakage check
            if "uniq_go_ids" in batch and isinstance(batch.get("pos_go_local"), list):
                uniq = batch["uniq_go_ids"]
                pos_local = batch["pos_go_local"]
                b = 0 if neg.size(0) > 0 else None
                if b is not None and pos_local and len(pos_local) > b and pos_local[b].numel() > 0:
                    g_pos = uniq.index_select(0, pos_local[b].to(uniq.device))
                    leaked = set(int(x) for x in g_pos.tolist()) & set(int(x) for x in neg[b][neg[b] >= 0].tolist())
                    assert not leaked, f"positives leaked into negatives: {len(leaked)}"

            # ZS filter check
            if "zs_mask" in batch:
                zs_b = batch["zs_mask"][0] if batch["zs_mask"].ndim == 2 else batch["zs_mask"]
                bad = [int(x) for x in neg[0].tolist() if x >= 0 and bool(zs_b[int(x)])]
                assert len(bad) == 0, f"ZS terms appeared in negatives: {len(bad)}"

        # Forward + losses finiteness
        a0, r0 = cuda_mem()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        losses = trainer.step_losses(batch, epoch_idx=0)
        loss = losses["total"]
        trainer.opt.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_([p for p in trainer.model.parameters() if p.requires_grad], max_norm=1.0)
        trainer.opt.step()
        for k, v in losses.items():
            if v is not None:
                assert torch.isfinite(v.detach()), f"non-finite loss {k}"
        loss_str = " | ".join([f"{k}:{float(v.item()) if hasattr(v, 'item') else float(v):.4f}" for k, v in losses.items() if v is not None])
        log.info("losses: %s", loss_str)
        a1, r1 = cuda_mem()
        if a1 or r1:
            log.info("cuda mem Δ: alloc=%s, res=%s", human_bytes(a1 - a0), human_bytes(r1 - r0))

    # 8) MemoryBank quick rebuild
    with Timer("memory_bank_refresh_faiss_rebuild", log):
        try:
            ok_uni = dbg_check_bijection(go_cache, memory_bank)
            ok_order = dbg_check_order_alignment(go_cache, memory_bank, sample=2000)
            dbg_spot_check_vectors(go_cache, memory_bank, n=16)

            # (opsiyonel) FAISS id evren kontrolü
            if hasattr(vres, "ids_tensor"):   # örnek; sende nasıl saklandıysa
                 dbg_check_faiss_ids_against_universe(go_cache, vres.ids_tensor)

            if not ok_uni:
                log.error("[debugger] STOP: Universe mismatch detected. Fix sources before training.")
            if not ok_order:
                log.warning("[debugger] Universe order mismatch detected. OK if you don't assume same ordering.")

        except Exception as e:
            log.exception("[debugger] consistency checks failed: %s", e)
            return

        try:
            idxs = list(range(min(8, memory_bank.embs.shape[0])))
            sub = F.normalize(memory_bank.embs[idxs] + 1e-3, p=2, dim=1).cpu()
            memory_bank.update(idxs, sub)
            faiss_re = rebuild_faiss_from_bank(memory_bank, metric="ip")
            VectorResources(faiss_re, memory_bank.embs)  # smoke
            log.info("bank refresh ok, rebuilt ntotal=%s", getattr(faiss_re, "ntotal", "unknown"))
        except Exception as e:
            log.warning("bank refresh test failed: %r", e)

    log.info("Debugger finished OK")


if __name__ == "__main__":
    main()
