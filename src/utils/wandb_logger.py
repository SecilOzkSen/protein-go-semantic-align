from __future__ import annotations
import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt


def _fig_to_wandb_image(fig):
    """Convert a Matplotlib figure to a wandb.Image and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    plt.close(fig)
    return wandb.Image(buf)


class WabLogger:
    """
    Thin wrapper around Weights & Biases to standardize our logging keys
    and provide convenience helpers for histograms, images, and tables.

    Namespaces
    ----------
    - loss/*      : scalar losses
    - phase/*     : phase id/name and phase change markers
    - sched/*     : curriculum knobs (hard_frac, k_hard, shortlist_M, hops_up/down, random_k, use_inbatch_easy)
    - lora/*      : LoRA parameter stats, gradients, histograms, embedding drift
    - expl/*      : explainability media (residue/window heatmaps, tables)
    - gospec/*    : GO-specific Watt-i stats (if/when you log them)
    """

    def __init__(self, run: wandb.sdk.wandb_run.Run, project: str, config: Dict):
        self.run = run
        # Allow changing/expanding config over time (e.g., when loaded from YAML/CLI)
        wandb.config.update(config, allow_val_change=True)

    # ----------------- LoRA diagnostics -----------------
    @torch.no_grad()
    def log_lora_summaries(self, model: torch.nn.Module, step: int):
        """
        Log per-parameter norms/grad-norms for LoRA/adapter params, and global counts.
        A parameter is considered LoRA/adapter if its name contains 'lora_' or 'adapter'.
        """
        total_trainable, total_params = 0, 0
        for n, p in model.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                total_trainable += p.numel()

            is_lora = ("lora_" in n) or ("adapter" in n.lower()) or ("LoRA" in n)
            if is_lora:
                # weight norm
                try:
                    wandb.log({f"lora/weight_norm[{n}]": float(p.norm().item())}, step=step)
                except Exception:
                    pass
                # grad norm (if existing)
                if p.grad is not None:
                    try:
                        wandb.log({f"lora/grad_norm[{n}]": float(p.grad.norm().item())}, step=step)
                    except Exception:
                        pass

        if total_params > 0:
            wandb.log({
                "lora/num_trainable_params": float(total_trainable),
                "lora/num_total_params": float(total_params),
                "lora/trainable_ratio": float(total_trainable) / float(total_params)
            }, step=step)

    @torch.no_grad()
    def log_lora_histograms(self, model: torch.nn.Module, step: int):
        """
        Log histograms for LoRA/adapter parameters and their gradients.
        """
        for n, p in model.named_parameters():
            is_lora = ("lora_" in n) or ("adapter" in n.lower()) or ("LoRA" in n)
            if not is_lora:
                continue
            try:
                wandb.log({f"lora/hist[{n}]": wandb.Histogram(p.detach().cpu().view(-1).float())}, step=step)
            except Exception:
                pass
            if p.grad is not None:
                try:
                    wandb.log({f"lora/grad_hist[{n}]": wandb.Histogram(p.grad.detach().cpu().view(-1).float())}, step=step)
                except Exception:
                    pass

    @torch.no_grad()
    def log_embedding_drift(self,
                            base_encoder,
                            lora_encoder,
                            probe_batch: Dict[str, torch.Tensor],
                            step: int):
        """
        Compute mean cosine similarity between a frozen base encoder and an LoRA-adjusted encoder
        on a small probe batch. Expect transformers-style dict outputs.
        """
        def _encode(encoder):
            out = encoder(**probe_batch)
            if isinstance(out, dict):
                if "last_hidden_state" in out:
                    H = out["last_hidden_state"].mean(dim=1)  # [B, D]
                elif "pooler_output" in out:
                    H = out["pooler_output"]
                else:
                    # fallback: if model returns tensor directly
                    H = out if torch.is_tensor(out) else None
            else:
                H = out
            if H is None:
                raise RuntimeError("Encoder output not understood for drift logging.")
            return torch.nn.functional.normalize(H, p=2, dim=-1)

        try:
            H_base = _encode(base_encoder)
            H_lora = _encode(lora_encoder)
            cs = (H_base * H_lora).sum(dim=-1)  # [B]
            wandb.log({
                "lora/embedding_drift_mean_cos": float(cs.mean().item()),
                "lora/embedding_drift_p05": float(cs.quantile(0.05).item()),
                "lora/embedding_drift_p95": float(cs.quantile(0.95).item()),
                "lora/embedding_drift_hist": wandb.Histogram(cs.detach().cpu().view(-1).float())
            }, step=step)
        except Exception:
            # Non-fatal; simply skip drift if shapes/models don't align
            pass

    # ----------------- Loss & schedule -----------------
    def log_losses(self,
                   losses: Dict[str, float],
                   step: int,
                   epoch: int,
                   phase: Dict[str, 'str|int'] = None,
                   sched: Dict[str, float] = None):
        """
        Log scalar losses, epoch/step counters, phase identity, and curriculum knobs.
        """
        payload = {f"loss/{k}": float(v) for k, v in losses.items()}
        payload.update({"trainer/epoch": int(epoch), "trainer/step": int(step)})

        if phase:
            for k, v in phase.items():
                payload[f"phase/{k}"] = v

        if sched:
            for k, v in sched.items():
                if v is not None:
                    payload[f"sched/{k}"] = v

        wandb.log(payload, step=step)

    # ----------------- Explainability (heatmaps) -----------------
    def log_residue_heatmap(self,
                            protein_id: str,
                            go_id: str,
                            scores: np.ndarray,
                            step: int,
                            title: Optional[str] = None):
        """
        Log a simple 1xL (or 1xW) heatmap as an image, labeled with ids.
        """
        fig = plt.figure(figsize=(8, 1.8))
        ax = plt.gca()
        arr = scores
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        ax.imshow(arr[None, :], aspect="auto")
        ax.set_yticks([])
        ax.set_xlabel("Residue index" if arr.ndim == 1 else "Window index")
        if title:
            ax.set_title(title)
        img = _fig_to_wandb_image(fig)
        wandb.log({
            "expl/heatmap_image": img,
            "expl/protein_id": protein_id,
            "expl/go_id": go_id
        }, step=step)

    def log_heatmap_table(self,
                          rows: List[Tuple[str, str, np.ndarray]],
                          step: int,
                          name: str = "explain_table"):
        """
        Log a wandb.Table with (protein_id, go_id, heatmap image) rows.
        """
        data = []
        for pid, gid, s in rows:
            fig = plt.figure(figsize=(8, 1.8))
            ax = plt.gca()
            arr = s
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            ax.imshow(arr[None, :], aspect="auto")
            ax.set_yticks([])
            ax.set_xlabel("Residue/Window")
            img = _fig_to_wandb_image(fig)
            data.append([pid, gid, img])
        table = wandb.Table(columns=["protein_id", "go_id", "heatmap"], data=data)
        wandb.log({f"expl/{name}": table}, step=step)

    # ----------------- GO-specific Watt-i (optional) -----------------
    def log_gospec_stats(self, stats: Dict[str, float], step: int):
        """
        stats should include keys like:
          enabled (0/1), sparsity_gt_tau, topk_concentration, kl_base_to_gospec, attribution_alignment_r
        """
        if not isinstance(stats, dict):
            return
        wandb.log({f"gospec/{k}": v for k, v in stats.items()}, step=step)