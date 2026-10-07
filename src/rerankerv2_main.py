from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from src.datasets.retriever_dataset import (
    RetrieverV2DumpDataset,
    validate_matching_go_banks,
)
from src.loss.asymmetric_loss import AsymmetricLoss, AsymmetricLossConfig
from src.models.rerankerv2_model import (
    PredictionHeadConfig,
    ProteinGOPredictionHead,
)
from src.training.wandb_helper import RerankerV2WandbLogger

LOGGER = logging.getLogger("rerankerv2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment B: frozen Retriever-v2 dumps + "
            "low-rank residual protein-GO prediction head"
        )
    )

    # Data
    p.add_argument("--train_dump", type=Path, required=True)
    p.add_argument("--val_dump", type=Path, required=True)

    # Runtime
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)

    # Architecture
    p.add_argument("--adapter_rank", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # ASL
    p.add_argument("--gamma_pos", type=float, default=0.0)
    p.add_argument("--gamma_neg", type=float, default=4.0)
    p.add_argument("--asl_clip", type=float, default=0.05)

    # Optimizer
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Preflight
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--preflight_steps", type=int, default=5)

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="protein-go-align-pfresgo")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    p.add_argument("--wandb_run_name", default=None)

    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
        dataset,
        *,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def move_batch(batch, device: torch.device):
    return {
        "protein_z": batch["protein_z"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "retriever_scores": batch["retriever_scores"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "labels": batch["labels"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
    }


@torch.no_grad()
def adapter_deltas(
        model: ProteinGOPredictionHead,
        protein_z: torch.Tensor,
        go_z: torch.Tensor,
):
    h_p = model.protein_adapter(protein_z)
    h_g = model.go_adapter(go_z)

    return {
        "protein_max_delta": float(
            (h_p - protein_z).abs().max().item()
        ),
        "go_max_delta": float(
            (h_g - go_z).abs().max().item()
        ),
    }


def create_wandb_run(args, train_ds, n_params: int):
    enabled = (
            args.wandb
            and args.wandb_mode != "disabled"
    )

    if not enabled:
        return None

    if wandb is None:
        raise RuntimeError(
            "--wandb was requested but wandb is not installed."
        )

    run_name = args.wandb_run_name or (
        f"RerankerV2-Preflight-"
        f"gp{args.gamma_pos:g}-"
        f"gn{args.gamma_neg:g}-"
        f"clip{args.asl_clip:g}"
    )

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode=args.wandb_mode,
        config={
            "experiment": "B",
            "architecture": (
                "frozen_retriever_v2"
                "+low_rank_residual_interaction_head"
            ),
            "train_dump": str(args.train_dump),
            "val_dump": str(args.val_dump),
            "embedding_dim": train_ds.embedding_dim,
            "go_universe_size": train_ds.num_go,
            "adapter_rank": args.adapter_rank,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "gamma_pos": args.gamma_pos,
            "gamma_neg": args.gamma_neg,
            "asl_clip": args.asl_clip,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "trainable_parameters": n_params,
        },
    )


def run_preflight(
        *,
        model: ProteinGOPredictionHead,
        criterion: AsymmetricLoss,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        go_z: torch.Tensor,
        device: torch.device,
        max_steps: int,
        grad_clip: float,
        wb: RerankerV2WandbLogger,
) -> None:
    if max_steps <= 0:
        raise ValueError("preflight_steps must be > 0")

    model.train()

    # ------------------------------------------------------------
    # 1. Zero-init invariant
    # ------------------------------------------------------------
    first_batch = move_batch(
        next(iter(train_loader)),
        device,
    )

    initial_delta = adapter_deltas(
        model,
        first_batch["protein_z"],
        go_z,
    )

    LOGGER.info(
        "[preflight] initial adapter delta | protein=%.8f | GO=%.8f",
        initial_delta["protein_max_delta"],
        initial_delta["go_max_delta"],
    )

    wb.log_adapter_state(
        step=0,
        protein_max_delta=initial_delta["protein_max_delta"],
        go_max_delta=initial_delta["go_max_delta"],
    )

    if initial_delta["protein_max_delta"] != 0.0:
        raise RuntimeError(
            "Protein adapter is not exact identity at initialization."
        )

    if initial_delta["go_max_delta"] != 0.0:
        raise RuntimeError(
            "GO adapter is not exact identity at initialization."
        )

    # ------------------------------------------------------------
    # 2. Short real optimization run
    # ------------------------------------------------------------
    last_step = 0

    for step, batch_cpu in enumerate(train_loader, start=1):
        if step > max_steps:
            break

        last_step = step
        batch = move_batch(batch_cpu, device)
        labels = batch["labels"]

        if torch.any((labels != 0) & (labels != 1)):
            raise RuntimeError(
                "Labels contain values outside {0,1}."
            )

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            protein_z=batch["protein_z"],
            go_z=go_z,
            retriever_scores=batch["retriever_scores"],
        )

        if logits.shape != labels.shape:
            raise RuntimeError(
                "logit/label shape mismatch: "
                f"{tuple(logits.shape)} vs {tuple(labels.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"Non-finite logits at step {step}"
            )

        loss, diagnostics = criterion(
            logits,
            labels,
            return_diagnostics=True,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite ASL at step {step}"
            )

        # Zero-init residual adapter invariant:
        # up must receive gradient immediately.
        # down may be zero on step 1 because up.weight starts at zero.
        loss.backward()

        grad_metrics = wb.gradient_norms(model)

        if step == 1:
            if grad_metrics.get("grad/protein_adapter_up", 0.0) <= 0.0:
                raise RuntimeError(
                    "Protein adapter UP received no first-step gradient."
                )

            if grad_metrics.get("grad/go_adapter_up", 0.0) <= 0.0:
                raise RuntimeError(
                    "GO adapter UP received no first-step gradient."
                )

            if grad_metrics.get("grad/predictor", 0.0) <= 0.0:
                raise RuntimeError(
                    "Prediction MLP received no first-step gradient."
                )

        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )

        wb.log_gradients(
            model=model,
            step=step,
            total_before_clip=total_grad_norm,
        )

        lr = float(optimizer.param_groups[0]["lr"])

        wb.log_train(
            diagnostics=diagnostics,
            lr=lr,
            step=step,
        )

        optimizer.step()

        LOGGER.info(
            "[preflight] step=%d | "
            "loss=%.6f | pos_loss=%.6f | neg_loss=%.6f | "
            "p_pos=%.4f | p_neg=%.4f | gap=%.4f | "
            "suppressed=%.4f",
            step,
            diagnostics["loss"],
            diagnostics["positive_loss_mean"],
            diagnostics["negative_loss_mean"],
            diagnostics["positive_probability_mean"],
            diagnostics["negative_probability_mean"],
            diagnostics["probability_gap"],
            diagnostics["suppressed_negative_fraction"],
        )

        LOGGER.info(
            "[preflight] gradients step=%d | "
            "p_down=%.3e | p_up=%.3e | "
            "g_down=%.3e | g_up=%.3e | "
            "predictor=%.3e | total_before_clip=%.3e",
            step,
            grad_metrics.get("grad/protein_adapter_down", 0.0),
            grad_metrics.get("grad/protein_adapter_up", 0.0),
            grad_metrics.get("grad/go_adapter_down", 0.0),
            grad_metrics.get("grad/go_adapter_up", 0.0),
            grad_metrics.get("grad/predictor", 0.0),
            float(total_grad_norm.item()),
        )

    # ------------------------------------------------------------
    # 3. Adapters must move away from exact identity after updates
    # ------------------------------------------------------------
    model.eval()

    final_delta = adapter_deltas(
        model,
        first_batch["protein_z"],
        go_z,
    )

    wb.log_adapter_state(
        step=last_step,
        protein_max_delta=final_delta["protein_max_delta"],
        go_max_delta=final_delta["go_max_delta"],
    )

    LOGGER.info(
        "[preflight] final adapter delta | protein=%.8f | GO=%.8f",
        final_delta["protein_max_delta"],
        final_delta["go_max_delta"],
    )

    if final_delta["protein_max_delta"] <= 0.0:
        raise RuntimeError(
            "Protein residual adapter did not move."
        )

    if final_delta["go_max_delta"] <= 0.0:
        raise RuntimeError(
            "GO residual adapter did not move."
        )

    LOGGER.info("[preflight] PASSED")


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    LOGGER.info("Device: %s", device)

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    train_ds = RetrieverV2DumpDataset(
        args.train_dump
    )
    val_ds = RetrieverV2DumpDataset(
        args.val_dump
    )

    validate_matching_go_banks(
        train_ds,
        val_ds,
    )

    LOGGER.info(
        "Dump contract OK | train=%d | val=%d | G=%d | D=%d",
        len(train_ds),
        len(val_ds),
        train_ds.num_go,
        train_ds.embedding_dim,
    )

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    # Shared GO bank, moved once.
    go_z = train_ds.get_go_z_tensor(
        dtype=torch.float32,
    ).to(
        device=device,
        non_blocking=True,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model_cfg = PredictionHeadConfig(
        dim=train_ds.embedding_dim,
        adapter_rank=args.adapter_rank,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    model = ProteinGOPredictionHead(
        model_cfg
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------
    loss_cfg = AsymmetricLossConfig(
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        clip=args.asl_clip,
    )

    criterion = AsymmetricLoss(
        loss_cfg
    ).to(device)

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    LOGGER.info(
        "Model | D=%d | rank=%d | hidden=%d | "
        "params=%d | ASL gp=%.2f gn=%.2f clip=%.3f",
        model_cfg.dim,
        model_cfg.adapter_rank,
        model_cfg.hidden_dim,
        n_params,
        loss_cfg.gamma_pos,
        loss_cfg.gamma_neg,
        loss_cfg.clip,
    )

    # ------------------------------------------------------------
    # W&B
    #
    # Important: main owns wandb.init().
    # Logger only formats/logs an already-created run.
    # ------------------------------------------------------------
    run = create_wandb_run(
        args,
        train_ds,
        n_params,
    )

    wb = RerankerV2WandbLogger(
        run=run,
        cfg=args,
    )

    try:
        if args.preflight:
            run_preflight(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                go_z=go_z,
                device=device,
                max_steps=args.preflight_steps,
                grad_clip=args.grad_clip,
                wb=wb,
            )
            return

        raise RuntimeError(
            "Full Experiment-B training loop is not enabled yet. "
            "Run --preflight first."
        )

    finally:
        wb.finalize()


if __name__ == "__main__":
    main()
