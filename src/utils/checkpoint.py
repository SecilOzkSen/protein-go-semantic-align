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

def _looks_like_state_dict(d):
    if not isinstance(d, dict):
        return False

    n_tensor = 0
    for v in d.values():
        if torch.is_tensor(v):
            n_tensor += 1
            if n_tensor >= 5:
                return True
    return False


def _unwrap_state_dict(blob, *, name: str):
    """
    Handles:
      state_dict
      {"model": state_dict}
      {"state_dict": state_dict}
      {"model_state_dict": state_dict}
      {"module": state_dict}
      repeated nesting
    """
    if not isinstance(blob, dict):
        raise RuntimeError(f"[checkpoint] {name} blob is not dict: {type(blob)}")

    state = blob
    unwrap_keys = ["model", "state_dict", "model_state_dict", "module", "net"]

    for depth in range(10):
        if _looks_like_state_dict(state):
            if depth > 0:
                print(f"[checkpoint] unwrapped {name} state_dict at depth={depth}")
            return state

        if not isinstance(state, dict):
            raise RuntimeError(
                f"[checkpoint] {name} became non-dict at depth={depth}: {type(state)}"
            )

        for key in unwrap_keys:
            if key in state and isinstance(state[key], dict):
                print(f"[checkpoint] unwrap {name} depth={depth}: state = state['{key}']")
                state = state[key]
                break
        else:
            raise RuntimeError(
                f"[checkpoint] Could not unwrap {name} state_dict. "
                f"Current keys={list(state.keys())[:30]}"
            )

    raise RuntimeError(f"[checkpoint] Exceeded max unwrap depth for {name}.")


def _clean_state_keys(state):
    """
    Clean common wrappers from keys if needed.
    """
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


def load_checkpoint(trainer, path: str, map_location="cuda"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"[checkpoint] Expected dict checkpoint, got {type(ckpt)}")

    if "model" not in ckpt:
        raise RuntimeError(f"No 'model' key in checkpoint. Keys: {ckpt.keys()}")

    print(f"[checkpoint] Loading from {path}")
    print("[checkpoint] top-level keys:", list(ckpt.keys()))

    # -------------------------
    # Model
    # -------------------------
    model_state = _unwrap_state_dict(ckpt["model"], name="model")
    model_state = _clean_state_keys(model_state)

    current = trainer.model.state_dict()

    # Optional strict shape filter, useful if some accidental incompatible keys exist.
    loadable = {}
    skipped_missing = []
    skipped_shape = []

    for k, v in model_state.items():
        if k not in current:
            skipped_missing.append(k)
            continue

        if tuple(current[k].shape) != tuple(v.shape):
            skipped_shape.append((k, tuple(v.shape), tuple(current[k].shape)))
            continue

        loadable[k] = v

    if len(loadable) == 0:
        print("[checkpoint][DEBUG] model_state sample:", list(model_state.keys())[:50])
        print("[checkpoint][DEBUG] current sample:", list(current.keys())[:50])
        raise RuntimeError("[checkpoint] Loaded 0 compatible model keys.")

    missing, unexpected = trainer.model.load_state_dict(loadable, strict=False)

    print(f"[checkpoint] Model loaded. compatible={len(loadable)} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[checkpoint][WARN] missing model keys example:", missing[:20])
    if unexpected:
        print("[checkpoint][WARN] unexpected model keys example:", unexpected[:20])
    if skipped_missing:
        print("[checkpoint][WARN] skipped_missing example:", skipped_missing[:20])
    if skipped_shape:
        print("[checkpoint][WARN] skipped_shape example:", skipped_shape[:10])

    # For true resume, missing hundreds of keys is not acceptable.
    # Same architecture resume should be basically exact.
    if len(missing) > 20:
        raise RuntimeError(
            f"[checkpoint] Too many missing model keys after resume: {len(missing)}. "
            "Checkpoint likely did not load correctly."
        )

    if unexpected:
        raise RuntimeError(
            f"[checkpoint] Unexpected model keys after resume: {unexpected[:20]}"
        )

    # Important sanity checks for this P2b run
    important_prefixes = [
        "protein_local_evidence_pool.",
        "protein_ln.",
        "proj_p.",
        "go_ln.",
        "proj_g.",
    ]

    for pref in important_prefixes:
        n_loaded = sum(k.startswith(pref) for k in loadable)
        print(f"[checkpoint][CHECK] {pref} loaded={n_loaded}")

    # -------------------------
    # EMA
    # -------------------------
    ema_state = ckpt.get("ema", {})
    if "go_encoder_k" in ema_state and getattr(trainer, "go_encoder_k", None) is not None:
        missing_ema, unexpected_ema = trainer.go_encoder_k.load_state_dict(
            ema_state["go_encoder_k"],
            strict=False,
        )
        print(
            f"[checkpoint] EMA go_encoder_k loaded. "
            f"missing={len(missing_ema)} unexpected={len(unexpected_ema)}"
        )

    # -------------------------
    # Optimizer
    # -------------------------
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

    # -------------------------
    # Meta
    # -------------------------
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