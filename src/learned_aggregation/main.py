"""One training entrypoint for BP/CC/MF and fixed/learned score aggregation."""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .checkpoints import (
    atomic_alias, atomic_json, capture_rng, epoch_checkpoints, initial_selection, preserve_best_aliases,
    read_checkpoint, restore_rng, resume, save_epoch, update_selection,
)
from .config import get_args, json_config
from .data import file_sha256, prepare_data


@contextmanager
def isolated_logging_rng():
    """Media/SDK randomness must not perturb dropout or epoch-boundary resume."""
    state = capture_rng({})
    try:
        yield
    finally:
        restore_rng(state, {})


def setup_logging(directory, level):
    Path(directory).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(Path(directory) / "train.log", encoding="utf-8")],
        force=True,
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_trainer(a, data, device):
    from src.configs.data_classes import LoggingConfig, TrainerConfig, TrainingContext
    from .trainer import OppTrainer

    context = TrainingContext(
        go_cache=data.go_cache, device=device, go_text_store=data.text_store,
        run_name=a.wandb_run_name, fp16_enabled=a.fp16,
        protein_pooling_strategy=a.protein_pooling_strategy, go_pool_type=a.go_pool_type,
        go_encoder_output_mode=a.go_encoder_output_mode,
        go_segment_representation_mode=a.go_segment_representation_mode,
        eval_id_list=list(data.active_ids), eval_seen_go_ids=data.seen,
        eval_unseen_ids=data.zero, eval_rare_go_ids=data.rare,
        logger=logging.getLogger("main"), logging=LoggingConfig(log_every=a.log_every),
    )
    cfg = TrainerConfig(
        d_h=data.d_h, d_g=data.d_g, d_z=a.align_dim, device=str(device),
        lr=a.lr, weight_decay=a.weight_decay, grad_clip=a.grad_clip, max_epochs=a.epochs,
        batch_size=a.batch_size, eval_batch_size=a.eval_batch_size, fp16=a.fp16,
        warmstart_path=None, monitor_metric=a.monitor_metric,
        secondary_monitor_metric=a.secondary_monitor_metric, monitor_mode=a.monitor_mode,
        protein_pooling_strategy=a.protein_pooling_strategy, use_lora=False,
        go_pooling="none", go_pool_type=a.go_pool_type,
        go_encoder_output_mode=a.go_encoder_output_mode,
        go_segment_representation_mode=a.go_segment_representation_mode,
        eval_go_bs=a.eval_go_bs, temperature=a.temperature, lambda_con=a.lambda_con,
        pbr_lambda=a.pbr_lambda, pbr_margin=a.pbr_margin, pbr_tau=a.pbr_tau,
        retrieval_eval_ks=a.retrieval_eval_ks, positive_rank_quantiles=a.positive_rank_quantiles,
        log_gradient_norms=a.log_gradient_norms, log_positive_rank_cdf=a.log_positive_rank_cdf,
        log_cardinality_metrics=a.log_cardinality_metrics, log_go_segment_weights=a.log_go_segment_weights,
    )
    extras = (
        "protein_mode", "segment_aggregation", "window_length", "window_stride", "mil_tau",
        "parent_strength", "go_chunk_size", "pbr_mode", "pbr_local_k", "frequency_aware_local",
        "mzsgo_frozen_bank", "validation_fixed_threshold", "log_go_term_metrics",
        "diagnostics_log_every_steps", "diagnostics_log_every_epochs", "diagnostics_max_table_rows",
        "diagnostics_heatmap_rows", "save_diagnostics_local", "output_dir",
    )
    for name in extras:
        setattr(cfg, name, getattr(a, name))
    cfg.canonical_go_jsonl = a.go_text_path
    cfg.train_frequency_by_go = data.train_counts
    trainer = OppTrainer(cfg=cfg, ctx=context, go_encoder=data.encoder, wandb_run=None)
    if a.frequency_aware_local:
        # Preserve the old Local PBR frequency definition, using TRAIN only.
        frequency = torch.tensor([data.train_counts.get(g, 0) for g in data.active_ids], dtype=torch.float32)
        observed = frequency > 0
        if not observed.any():
            raise RuntimeError("No positive TRAIN term frequencies")
        weights = torch.ones_like(frequency)
        weights[observed] = frequency[observed].pow(-a.frequency_weight_power)
        weights[observed] /= weights[observed].mean()
        weights[observed] = weights[observed].clamp(a.frequency_weight_min, a.frequency_weight_max)
        trainer._frequency_weights_gpu = weights.to(device)
        logging.info("[FREQ-LOCAL-PBR] observed_terms=%d weight_min=%.4f weight_mean=%.4f weight_max=%.4f",
                     int(observed.sum()), float(weights[observed].min()),
                     float(weights[observed].mean()), float(weights[observed].max()))
    return trainer, cfg


def start_logger(a, trainer, cfg, data):
    from src.training.wandb_helper import RetrieverWandbLogger
    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project=a.wandb_project, entity=a.wandb_entity,
                         name=a.wandb_run_name, config=json_config(a), mode=a.wandb_mode,
                         dir=str(a.output_dir))
    wb = RetrieverWandbLogger(run=run, cfg=cfg, go_ids=data.active_ids,
                              segment_names=trainer.scoring_segment_names, output_dir=a.output_dir)
    objective = "full_go_infonce"
    if a.pbr_lambda > 0:
        objective += f"+{a.pbr_mode}_pbr_lambda{a.pbr_lambda:g}"
        if a.frequency_aware_local:
            objective += "_frequency_aware"
    wb.log_run_metadata(model=trainer.model, active_go_ids=data.active_ids,
                        protein_pooling=a.protein_pooling_strategy,
                        go_segments=a.enabled_segments, objective=objective)
    if run is not None:
        for key, value in data.identity.items():
            if isinstance(value, (int, float, str)):
                run.summary[f"data/{key}"] = value
        counts = [len(data.train_ds.pid2pos[p]) for p in data.train_ds.pids]
        run.summary["data/positives_per_protein_mean"] = float(np.mean(counts))
        run.summary["data/positives_per_protein_p95"] = float(np.percentile(counts, 95))
        run.summary["selection/primary"] = a.monitor_metric
        run.summary["selection/early_stop_rule"] = "reset on primary OR secondary improvement"
        run.summary["selection/evaluation_split"] = "validation"
        run.summary["resume/source"] = str(a.resume) if a.resume else "from_scratch"
    return wb, run


def train_epoch(trainer, loader, wb, a, epoch, device):
    trainer.model.train()  # aligner keeps the frozen GO encoder in eval mode
    running = {"total": 0.0, "contrastive": 0.0, "ranking_loss": 0.0}
    n_batches = 0
    use_bf16 = bool(a.fp16 and device.type == "cuda")
    params = [p for p in trainer.model.parameters() if p.requires_grad]
    for batch in loader:
        trainer.opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            losses = trainer.step_losses(batch, epoch)
            loss = losses["total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at epoch={epoch} step={trainer._global_step}")
        loss.backward()
        step = int(trainer._global_step)
        if a.log_gradient_norms and (step == 1 or step % a.diagnostics_log_every_steps == 0):
            with isolated_logging_rng():
                wb.log_gradients(model=trainer.model, step=step)
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, a.grad_clip, error_if_nonfinite=True)
        trainer.opt.step()
        n_batches += 1
        for key in running:
            value = losses[key]
            running[key] += float(value.detach().item() if torch.is_tensor(value) else value)
        if step % a.log_every == 0:
            averaged = {key: value / n_batches for key, value in running.items()}
            lr = float(trainer.opt.param_groups[0]["lr"])
            logging.getLogger("main").info("[train] epoch %d step %d :: %s | lr:%.2e", epoch, step,
                                           " | ".join(f"{key}:{value:.5f}" for key, value in averaged.items()), lr)
            with isolated_logging_rng():
                wb.log_train(losses={**losses, **averaged}, lr=lr, step=step)
    if not n_batches:
        raise RuntimeError("Training DataLoader produced no batches")
    return {key: value / n_batches for key, value in running.items()}


def validate(trainer, loader, a, epoch, device):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=bool(a.fp16 and device.type == "cuda")):
        metrics = trainer.eval_epoch(loader, epoch)
    result = {key: float(value) for key, value in metrics.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise RuntimeError("Validation produced a nonfinite scalar metric")
    logging.getLogger("main").info("[val] epoch %d :: %s", epoch,
                                   " | ".join(f"{key}:{value:.5f}" for key, value in result.items()))
    return result


def run_training(a):
    device = torch.device(a.device)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("This runner supports cuda or cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for a CPU run")
    torch.set_num_threads(a.torch_threads)
    seed_all(a.seed)
    existing = epoch_checkpoints(a.output_dir)
    if not a.resume and (existing or (a.output_dir / "last.pt").exists()):
        raise RuntimeError("Output directory already has checkpoints. Use --resume last.pt or a new --output-dir")
    if a.resume and not a.eval_only:
        meta = read_checkpoint(a.resume)["meta"]
        if existing and existing[-1][0] > meta["epoch"] + 1:
            raise RuntimeError("Output contains later epochs than resume source; use a new --output-dir to branch the run")
    setup_logging(a.output_dir, a.log_level)
    log = logging.getLogger("main")
    log.info("[CONFIG] branch=%s protein=%s aggregation=%s InfoNCE=%g PBR=%g device=%s",
             a.branch, a.protein_mode, a.segment_aggregation, a.lambda_con, a.pbr_lambda, device)
    data = prepare_data(a, device)
    trainer, cfg = build_trainer(a, data, device)
    resolved = json_config(a)
    sources = {path.name: file_sha256(path) for path in Path(__file__).parent.glob("*.py")}
    helper_path = Path(__file__).resolve().parents[1] / "training" / "wandb_helper.py"
    sources["shared/wandb_helper.py"] = file_sha256(helper_path)
    manifest = {"config": resolved, "architecture": trainer.model.architecture_metadata(),
                "data": data.identity, "source_sha256": sources,
                "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": str(device)}}
    start_epoch, selection = 0, initial_selection()
    if a.resume:
        start_epoch, selection = resume(trainer, a.resume, config=resolved,
                                        data_identity=data.identity, generators=data.generators, eval_only=a.eval_only)
        if not a.eval_only:
            preserve_best_aliases(a.resume, a.output_dir)
    # Keep original manifest and write invocation-specific manifests on resume/eval.
    name = "eval_manifest.json" if a.eval_only else (f"resume_epoch{start_epoch}_manifest.json" if a.resume else "run_manifest.json")
    atomic_json(a.output_dir / name, manifest)
    with isolated_logging_rng():
        wb, run = start_logger(a, trainer, cfg, data)
    try:
        if a.eval_only:
            epoch = start_epoch - 1
            metrics = validate(trainer, data.val_loader, a, epoch, device)
            with isolated_logging_rng():
                wb.log_validation(metrics=metrics, diagnostics=trainer.last_eval_diagnostics,
                                  step=trainer._global_step, epoch=epoch)
            atomic_json(a.output_dir / "validation_eval_only.json", metrics)
            return
        if selection["early_stopped"]:
            log.info("Checkpoint already met its early-stopping rule. No further epochs run.")
            return
        if start_epoch >= a.epochs:
            log.info("Already completed %d epochs; configured total is %d.", start_epoch, a.epochs)
            return
        log.info("Start training: epochs %d..%d (zero-based), bf16=%s", start_epoch, a.epochs - 1,
                 bool(a.fp16 and device.type == "cuda"))
        for epoch in range(start_epoch, a.epochs):
            began = time.perf_counter()
            losses = train_epoch(trainer, data.train_loader, wb, a, epoch, device)
            metrics = validate(trainer, data.val_loader, a, epoch, device)
            selection, improved, macro_improved = update_selection(
                selection, metrics, primary=a.monitor_metric, secondary=a.secondary_monitor_metric,
                epoch=epoch, step=trainer._global_step, patience=a.early_stop_patience,
            )
            # Save the resumable epoch before optional dashboard/media processing.
            saved = save_epoch(trainer, directory=a.output_dir, epoch=epoch, config=resolved,
                               data_identity=data.identity, selection=selection, generators=data.generators,
                               metrics=metrics, primary_improved=improved, secondary_improved=macro_improved,
                               keep_last_n=a.keep_last_n)
            atomic_json(a.output_dir / "validation_history" / f"epoch_{epoch + 1:03d}.json",
                        {"epoch": epoch, "completed_epochs": epoch + 1, "step": trainer._global_step,
                         "train": losses, "validation": metrics, "selection": selection,
                         "epoch_seconds": time.perf_counter() - began})
            with isolated_logging_rng():
                wb.log_validation(metrics=metrics, diagnostics=trainer.last_eval_diagnostics,
                                  step=trainer._global_step, epoch=epoch)
                if run is not None:
                    run.summary[f"best/{a.monitor_metric}"] = selection["best_primary"]
                    run.summary[f"best/{a.secondary_monitor_metric}"] = selection["best_secondary"]
                    run.summary["best/epoch"] = selection["best_primary_epoch"]
                    run.summary["best/step"] = selection["best_primary_step"]
                    run.summary["selection/no_improve_epochs"] = selection["no_improve_epochs"]
            if selection["early_stopped"]:
                log.info("[early-stop] neither monitor improved for %d epochs", a.early_stop_patience)
                break
        atomic_alias(saved, a.output_dir / "checkpoint_final.pt")
        atomic_json(a.output_dir / "selection.json", selection)
        log.info("Training finished. Selected validation checkpoint: %s (epoch %s, %s=%.8f)",
                 a.output_dir / "best_primary.pt", selection["best_primary_epoch"],
                 a.monitor_metric, selection["best_primary"])
    finally:
        best = {}
        if selection["best_primary"] is not None:
            best = {a.monitor_metric: selection["best_primary"],
                    a.secondary_monitor_metric: selection["best_secondary"],
                    "epoch": selection["best_primary_epoch"], "step": selection["best_primary_step"]}
        with isolated_logging_rng():
            wb.finalize(best_metrics=best)


def main(argv=None):
    a = get_args(argv)
    if a.print_config:
        print(json.dumps(json_config(a), indent=2))
        return
    # Dataset workers use the same explicit spawn policy as the supplied main.
    if a.num_workers:
        torch.multiprocessing.set_start_method("spawn", force=True)
    began = time.perf_counter()
    try:
        run_training(a)
    except KeyboardInterrupt:
        logging.warning("Interrupted. Resume from the latest complete last.pt checkpoint.")
        raise
    except Exception:
        logging.exception("Training failed; completed checkpoints were not overwritten")
        raise
    finally:
        logging.info("Total runtime: %.1f min", (time.perf_counter() - began) / 60)


if __name__ == "__main__":
    main()
