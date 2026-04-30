import torch
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
    if hasattr(trainer, "go_encoder_k") and trainer.go_encoder_k is not None:
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
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if "model" not in ckpt:
        raise RuntimeError(f"No 'model' key in checkpoint. Keys: {ckpt.keys()}")

    missing, unexpected = trainer.model.load_state_dict(ckpt["model"], strict=False)
    print(f"[checkpoint] Model loaded. missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[checkpoint][WARN] missing model keys example:", missing[:10])
    if unexpected:
        print("[checkpoint][WARN] unexpected model keys example:", unexpected[:10])

    ema_state = ckpt.get("ema", {})
    if "go_encoder_k" in ema_state and getattr(trainer, "go_encoder_k", None) is not None:
        missing_ema, unexpected_ema = trainer.go_encoder_k.load_state_dict(
            ema_state["go_encoder_k"],
            strict=False,
        )
        print(f"[checkpoint] EMA go_encoder_k loaded. missing={len(missing_ema)} unexpected={len(unexpected_ema)}")

    opt_blob = ckpt.get("optimizer", None)
    if opt_blob is not None and hasattr(trainer, "opt"):
        try:
            if isinstance(opt_blob, dict) and "param_groups" in opt_blob:
                trainer.opt.load_state_dict(opt_blob)
                print("[checkpoint] Optimizer loaded.")

            elif isinstance(opt_blob, dict) and "optimizer" in opt_blob:
                trainer.opt.load_state_dict(opt_blob["optimizer"])
                print("[checkpoint] Optimizer loaded from nested key.")

            else:
                keys = list(opt_blob.keys()) if isinstance(opt_blob, dict) else None
                print(f"[checkpoint][WARN] Optimizer skipped. type={type(opt_blob)} keys={keys}")

        except Exception as e:
            print(f"[checkpoint][WARN] Optimizer load failed, continuing with fresh optimizer: {repr(e)}")

    else:
        print("[checkpoint][WARN] No optimizer found in checkpoint, continuing with fresh optimizer.")

    meta = ckpt.get("meta", {})

    trainer._global_step = int(meta.get("global_step", meta.get("step", 0)))
    start_epoch = int(meta.get("epoch", -1)) + 1

    if trainer._global_step <= 0:
        print("[checkpoint][WARN] global_step is 0 after loading.")
    if start_epoch < 0:
        raise RuntimeError(f"Invalid start_epoch={start_epoch}")

    print(f"[checkpoint] Loaded from {path}")
    print(f"[checkpoint] meta keys={list(meta.keys())}")
    print(f"[checkpoint] Resume from epoch={start_epoch}, step={trainer._global_step}")

    return start_epoch