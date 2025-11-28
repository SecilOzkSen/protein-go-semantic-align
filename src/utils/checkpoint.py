import torch
import os
from pathlib import Path

def save_checkpoint(out_dir: str,
                    tag: str,
                    trainer,
                    args=None,
                    epoch: int = 0,
                    step: int = 0) -> str:
    """
    Save a unified training checkpoint.
    Handles non-nn.Module trainers by extracting model and optimizer states manually.

    Returns the full checkpoint path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"checkpoint_{tag}.pt"

    # --- 1-Core model state ---
    model_state = {}
    if hasattr(trainer, "model") and isinstance(trainer.model, torch.nn.Module):
        model_state["model"] = trainer.model.state_dict()

    # --- 2-EMA / teacher modules ---
    ema_state = {}
    if hasattr(trainer, "index_projector"):
        ema_state["index_projector"] = trainer.index_projector.state_dict()
    if hasattr(trainer, "go_encoder_k"):
        ema_state["go_encoder_k"] = trainer.go_encoder_k.state_dict()

    # --- 3- Optimizer & Scheduler ---
    opt_state = {}
    if hasattr(trainer, "opt"):
        opt_state["optimizer"] = trainer.opt.state_dict()
    if hasattr(trainer, "scheduler"):
        opt_state["scheduler"] = trainer.scheduler.state_dict() if hasattr(trainer.scheduler, "state_dict") else {}

    # --- 4- Training metadata ---
    meta = dict(
        epoch=epoch,
        step=step,
        global_step=getattr(trainer, "_global_step", step),
        m_ema=getattr(trainer, "m_ema", None),
        config=getattr(args, "__dict__", {}),
    )

    # --- 5- Aggregate checkpoint ---
    ckpt = dict(
        model=model_state,
        ema=ema_state,
        optimizer=opt_state,
        meta=meta,
    )

    torch.save(ckpt, ckpt_path)
    print(f"[checkpoint] Saved → {ckpt_path}")
    return str(ckpt_path)

def load_checkpoint(trainer, path: str, map_location="cuda"):
    ckpt = torch.load(path, map_location=map_location)
    model_state = ckpt.get("model", {})
    if "model" in model_state and hasattr(trainer, "model"):
        trainer.model.load_state_dict(model_state["model"], strict=False)

    ema_state = ckpt.get("ema", {})
    if "index_projector" in ema_state and hasattr(trainer, "index_projector"):
        trainer.index_projector.load_state_dict(ema_state["index_projector"], strict=False)
    if "go_encoder_k" in ema_state and hasattr(trainer, "go_encoder_k"):
        trainer.go_encoder_k.load_state_dict(ema_state["go_encoder_k"], strict=False)

    if hasattr(trainer, "opt") and "optimizer" in ckpt.get("optimizer", {}):
        trainer.opt.load_state_dict(ckpt["optimizer"]["optimizer"])
    print(f"[checkpoint] Loaded from {path}")