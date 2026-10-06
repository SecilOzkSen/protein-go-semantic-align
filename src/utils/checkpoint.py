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


def load_checkpoint(
        trainer,
        path: str,
        map_location="cuda",
):
    """
    Resume a Retriever v2 training run.

    Restores:
      - model weights
      - optimizer state
      - global step
      - next epoch

    Retriever v2 intentionally has no:
      - EMA GO encoder
      - local protein expert
      - slot state
      - queue state
      - logit-scale state
    """
    ckpt = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"[checkpoint] Expected dict checkpoint, got {type(ckpt)}"
        )

    if "model" not in ckpt:
        raise RuntimeError(
            f"[checkpoint] No 'model' key. Keys: {list(ckpt.keys())}"
        )

    print(f"[checkpoint] Loading from {path}")
    print(
        "[checkpoint] top-level keys:",
        list(ckpt.keys()),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model_state = _unwrap_state_dict(
        ckpt["model"],
        name="model",
    )
    model_state = _clean_state_keys(model_state)

    current_state = trainer.model.state_dict()

    loadable = {}
    skipped_missing = []
    skipped_shape = []

    for key, value in model_state.items():
        if key not in current_state:
            skipped_missing.append(key)
            continue

        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped_shape.append(
                (
                    key,
                    tuple(value.shape),
                    tuple(current_state[key].shape),
                )
            )
            continue

        loadable[key] = value

    if not loadable:
        print(
            "[checkpoint][DEBUG] checkpoint model sample:",
            list(model_state.keys())[:50],
        )
        print(
            "[checkpoint][DEBUG] current model sample:",
            list(current_state.keys())[:50],
        )
        raise RuntimeError(
            "[checkpoint] Loaded 0 compatible model parameters."
        )

    missing, unexpected = trainer.model.load_state_dict(
        loadable,
        strict=False,
    )

    print(
        "[checkpoint] Model loaded. "
        f"compatible={len(loadable)} "
        f"missing={len(missing)} "
        f"unexpected={len(unexpected)}"
    )

    if skipped_missing:
        print(
            "[checkpoint][WARN] checkpoint keys not present "
            "in current model:",
            skipped_missing[:20],
        )

    if skipped_shape:
        print(
            "[checkpoint][WARN] shape-mismatched keys:",
            skipped_shape[:10],
        )

    if missing:
        print(
            "[checkpoint][WARN] missing current-model keys:",
            missing[:20],
        )

    if unexpected:
        print(
            "[checkpoint][WARN] unexpected keys:",
            unexpected[:20],
        )

    # ------------------------------------------------------------------
    # True resume must match the Retriever v2 architecture.
    # ------------------------------------------------------------------

    if missing:
        raise RuntimeError(
            "[checkpoint] Missing model parameters during Retriever v2 "
            f"resume: {missing[:20]}"
        )

    if unexpected:
        raise RuntimeError(
            "[checkpoint] Unexpected model parameters during Retriever v2 "
            f"resume: {unexpected[:20]}"
        )

    if skipped_shape:
        raise RuntimeError(
            "[checkpoint] Shape mismatch during Retriever v2 resume: "
            f"{skipped_shape[:10]}"
        )

    # ------------------------------------------------------------------
    # Critical Retriever v2 modules
    # ------------------------------------------------------------------

    important_prefixes = [
        "protein_mean_attn_gate_pool.",
        "protein_ln.",
        "proj_p.",
        "go_segment_gate.",
        "go_ln.",
        "proj_g.",
    ]

    for prefix in important_prefixes:
        expected = sum(
            key.startswith(prefix)
            for key in current_state
        )
        loaded = sum(
            key.startswith(prefix)
            for key in loadable
        )

        print(
            f"[checkpoint][CHECK] {prefix} "
            f"loaded={loaded}/{expected}"
        )

        if expected == 0:
            raise RuntimeError(
                "[checkpoint] Expected Retriever v2 module is "
                f"missing from current model: {prefix}"
            )

        if loaded != expected:
            raise RuntimeError(
                "[checkpoint] Retriever v2 module was not fully restored: "
                f"{prefix} loaded={loaded}/{expected}"
            )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    opt_blob = ckpt.get("optimizer")

    if opt_blob is None:
        raise RuntimeError(
            "[checkpoint] No optimizer state found. "
            "True training resume requires optimizer state."
        )

    if not hasattr(trainer, "opt"):
        raise RuntimeError(
            "[checkpoint] Trainer has no optimizer."
        )

    try:
        if (
                isinstance(opt_blob, dict)
                and "param_groups" in opt_blob
        ):
            trainer.opt.load_state_dict(opt_blob)

        elif (
                isinstance(opt_blob, dict)
                and "optimizer" in opt_blob
        ):
            trainer.opt.load_state_dict(
                opt_blob["optimizer"]
            )

        else:
            raise RuntimeError(
                "Unrecognized optimizer checkpoint format. "
                f"type={type(opt_blob)}"
            )

    except Exception as exc:
        raise RuntimeError(
            "[checkpoint] Optimizer restore failed. "
            "Refusing to silently resume with a fresh optimizer."
        ) from exc

    print("[checkpoint] Optimizer loaded.")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    meta = ckpt.get("meta", {})

    if not isinstance(meta, dict):
        raise RuntimeError(
            f"[checkpoint] Expected meta dict, got {type(meta)}"
        )

    saved_epoch = int(
        meta.get("epoch", -1)
    )
    global_step = int(
        meta.get(
            "global_step",
            meta.get("step", 0),
        )
    )

    if saved_epoch < 0:
        raise RuntimeError(
            "[checkpoint] Missing or invalid epoch metadata: "
            f"{saved_epoch}"
        )

    if global_step <= 0:
        raise RuntimeError(
            "[checkpoint] Missing or invalid global_step metadata: "
            f"{global_step}"
        )

    trainer._global_step = global_step

    # Checkpoint stores the completed epoch.
    # Resume from the following epoch.
    start_epoch = saved_epoch + 1

    print(
        f"[checkpoint] saved_epoch={saved_epoch}"
    )
    print(
        f"[checkpoint] global_step={trainer._global_step}"
    )
    print(
        f"[checkpoint] Resume from epoch={start_epoch}"
    )

    return start_epoch