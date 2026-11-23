"""Atomic epoch checkpoints with explicit scorer identity and complete resume state."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

FORMAT = "mzsgo_mf_ablation_checkpoint_v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_alias(source, destination):
    """Immutable checkpoint aliases share storage where the filesystem supports it."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    os.unlink(tmp)
    try:
        try:
            os.link(source, tmp)
        except OSError:
            shutil.copyfile(source, tmp)
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def capture_rng(generators):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": {name: gen.get_state() for name, gen in generators.items()},
    }


def restore_rng(state, generators):
    # Reject incompatible state before changing any of the live generators.
    if set(generators) != set(state["loaders"]):
        raise RuntimeError("DataLoader generator names differ from checkpoint")
    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state["cuda"]) != cuda_devices:
        raise RuntimeError("CUDA device count changed since checkpoint; exact RNG resume is unavailable")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    for name, gen in generators.items():
        gen.set_state(state["loaders"][name].cpu())


def training_contract(config):
    """Settings that may alter the experiment cannot drift silently on resume."""
    names = (
        "seed", "device", "batch_size", "eval_batch_size", "num_workers", "fp16", "max_len", "overlap",
        "lr", "weight_decay", "grad_clip", "temperature", "lambda_con", "pbr_lambda",
        "pbr_mode", "pbr_local_k", "pbr_margin", "pbr_tau", "frequency_aware_local",
        "frequency_weight_power", "frequency_weight_min", "frequency_weight_max",
        "monitor_metric", "secondary_monitor_metric", "monitor_mode", "early_stop_patience",
        "encoder_name", "go_encoder_inner_pooling", "go_segment_max_len", "rare_lt",
        "retrieval_eval_ks", "ablation_id", "go_representation", "go_single_text_max_len",
    )
    return {name: config.get(name) for name in names}


def initial_selection():
    return {"best_primary": None, "best_primary_epoch": None, "best_primary_step": None,
            "best_secondary": None, "best_secondary_epoch": None, "no_improve_epochs": 0,
            "early_stopped": False}


def update_selection(selection, metrics, *, primary, secondary, epoch, step, patience):
    state = dict(selection)
    score, aux = float(metrics[primary]), float(metrics[secondary])
    if not np.isfinite(score) or not np.isfinite(aux):
        raise RuntimeError("Nonfinite validation checkpoint metric")
    primary_improved = state["best_primary"] is None or score > state["best_primary"] + 1e-6
    secondary_improved = state["best_secondary"] is None or aux > state["best_secondary"] + 1e-6
    if primary_improved:
        state.update(best_primary=score, best_primary_epoch=epoch, best_primary_step=step)
    if secondary_improved:
        state.update(best_secondary=aux, best_secondary_epoch=epoch)
    # Preserve the source main's early-stop rule; primary still selects the model.
    state["no_improve_epochs"] = 0 if primary_improved or secondary_improved else state["no_improve_epochs"] + 1
    state["early_stopped"] = bool(patience > 0 and state["no_improve_epochs"] >= patience)
    return state, primary_improved, secondary_improved


def epoch_checkpoints(directory):
    items = []
    for path in Path(directory).glob("checkpoint_epoch*.pt"):
        match = re.fullmatch(r"checkpoint_epoch(\d+)\.pt", path.name)
        if match:
            items.append((int(match.group(1)), path))
    return sorted(items)


def save_epoch(trainer, *, directory, epoch, config, data_identity, selection,
               generators, metrics, primary_improved, secondary_improved, keep_last_n):
    if keep_last_n < 1:
        raise ValueError("keep_last_n must be at least one")
    directory = Path(directory)
    path = directory / f"checkpoint_epoch{epoch + 1}.pt"
    payload = {
        "format": FORMAT,
        "model": trainer.model.state_dict(), "optimizer": trainer.opt.state_dict(),
        "meta": {
            "epoch": epoch, "global_step": int(trainer._global_step), "step": int(trainer._global_step),
            "epoch_complete": True, "architecture": trainer.model.architecture_metadata(),
            "config": config, "data_identity": data_identity, "training_contract": training_contract(config),
            "selection": selection, "validation_metrics": metrics,
            "inference_entrypoint": "src.ablations.b3_1_local_mean.inference.load_aligner",
        },
        "rng": capture_rng(generators),
    }
    atomic_torch_save(payload, path)
    if primary_improved:
        atomic_alias(path, directory / "best_primary.pt")
    if secondary_improved:
        atomic_alias(path, directory / "best_macro.pt")
    atomic_alias(path, directory / "last.pt")
    for _, old in epoch_checkpoints(directory)[:-keep_last_n]:
        old.unlink()
    logging.getLogger("checkpoint").info("Saved %s", path)
    return path


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise RuntimeError(
            "This resume loader accepts this package's checkpoints only. Legacy fixed/local/global "
            "weights do not identify their scoring architecture reliably. Start the learned ablation "
            "from scratch; keep legacy runs in their original evaluator."
        )
    if not checkpoint.get("meta", {}).get("epoch_complete"):
        raise RuntimeError("Checkpoint is not a completed epoch")
    return checkpoint


def resume(trainer, path, *, config, data_identity, generators, eval_only=False):
    checkpoint = read_checkpoint(path)
    meta = checkpoint["meta"]
    if meta["architecture"] != trainer.model.architecture_metadata():
        raise RuntimeError("Checkpoint architecture differs from resolved YAML: refusing scorer mismatch")
    if meta["data_identity"] != data_identity:
        raise RuntimeError("Checkpoint split, annotations or GO bank differs from the current inputs")
    if not eval_only and meta["training_contract"] != training_contract(config):
        old, new = meta["training_contract"], training_contract(config)
        changed = [key for key in new if old.get(key) != new[key]]
        raise RuntimeError(f"Training settings changed on resume: {changed}")
    trainer.model.load_state_dict(checkpoint["model"], strict=True)
    trainer._global_step = int(meta["global_step"])
    if not eval_only:
        trainer.opt.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"], generators)
    logging.getLogger("checkpoint").info(
        "Restored epoch=%d (completed epochs=%d), step=%d, primary=%s, patience=%d",
        meta["epoch"], meta["epoch"] + 1, trainer._global_step,
        meta["selection"]["best_primary"], meta["selection"]["no_improve_epochs"],
    )
    return int(meta["epoch"]) + 1, dict(meta["selection"])


def preserve_best_aliases(resume_path, directory):
    """Recover exactly the historical winners named by the resumed selection.

    Saving an epoch is atomic, but updating its three aliases is not a single
    transaction. A disk failure between those operations can leave an existing
    alias stale or corrupt. Existence alone therefore never establishes that an
    alias is the selected winner.
    """
    checkpoint = read_checkpoint(resume_path)
    resume_meta = checkpoint["meta"]
    selection = resume_meta["selection"]
    source_dir, directory = Path(resume_path).parent, Path(directory)

    def verified(path, expected_epoch, metric_name, expected_value, expected_step):
        try:
            meta = read_checkpoint(path)["meta"]
            if meta["epoch"] != expected_epoch:
                return False
            if any(meta.get(key) != resume_meta.get(key)
                   for key in ("architecture", "data_identity", "training_contract")):
                return False
            if expected_step is not None and meta["global_step"] != expected_step:
                return False
            actual_value = meta.get("validation_metrics", {}).get(metric_name)
            return (actual_value is not None and expected_value is not None
                    and np.isfinite(float(actual_value))
                    and np.isclose(float(actual_value), float(expected_value), rtol=0.0, atol=1e-12))
        except Exception as exc:
            logging.getLogger("checkpoint").debug("Unusable alias candidate %s: %s", path, exc)
            return False

    for filename, key, score_key, config_key in (
        ("best_primary.pt", "best_primary_epoch", "best_primary", "monitor_metric"),
        ("best_macro.pt", "best_secondary_epoch", "best_secondary", "secondary_monitor_metric"),
    ):
        target = directory / filename
        expected_epoch = selection[key]
        if expected_epoch is None:
            if target.exists():
                raise RuntimeError(f"Unexpected {target}: resumed selection has no historical winner")
            continue
        metric_name = resume_meta["config"][config_key]
        expected_step = selection["best_primary_step"] if score_key == "best_primary" else None
        epoch_name = f"checkpoint_epoch{int(expected_epoch) + 1}.pt"
        candidates = [target, Path(resume_path), source_dir / filename,
                      source_dir / epoch_name, directory / epoch_name]
        seen = set()
        for source in candidates:
            resolved = source.resolve()
            if resolved in seen or not source.is_file():
                continue
            seen.add(resolved)
            if verified(source, expected_epoch, metric_name, selection[score_key], expected_step):
                atomic_alias(source, target)
                break
        else:
            raise RuntimeError(
                f"Cannot restore verified {filename} for epoch={expected_epoch} from {source_dir}. "
                "The alias is missing, stale, corrupt, or belongs to different inputs. "
                f"Restore the original winner or {epoch_name} before resuming."
            )
