from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import logging

from src.script.dump_retriever_candidates_global_local import (
    configure,
    setup_logging_simple,
    set_seed,
    build_go_cache,
    build_go_encoder_and_text_store,
    canonicalize_and_align_inputs,
    build_stores,
    build_datasets,
    build_trainer_for_dump,
    load_model_weights_only,
)

from src.script import dump_retriever_candidates_global_local as dump_base


# ============================================================
# Basic geometry
# ============================================================

def _to_float(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.detach().float())


def _effective_rank(x: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Effective rank of the centered representation matrix.

    x: [N, D]
    """
    x = _to_float(x)
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan")

    x = x - x.mean(dim=0, keepdim=True)

    s = torch.linalg.svdvals(x)

    if s.numel() == 0:
        return float("nan")

    power = s.square()
    denom = power.sum()

    if float(denom) <= eps:
        return 1.0

    p = power / denom
    p = p.clamp_min(eps)

    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


def _offdiag_cosines(
        x: torch.Tensor,
        max_items: int = 1943,
) -> torch.Tensor:
    """
    Return upper-triangle pairwise cosine similarities.

    x: [N,D]
    """
    x = _to_float(x)

    if x.ndim != 2:
        raise ValueError(f"Expected [N,D], got {tuple(x.shape)}")

    if x.shape[0] > max_items:
        x = x[:max_items]

    x = F.normalize(x, dim=-1)

    sim = x @ x.T

    n = sim.shape[0]
    mask = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=sim.device),
        diagonal=1,
    )

    return sim[mask].cpu()


def _geometry_summary(
        name: str,
        x: torch.Tensor,
) -> Dict[str, float]:
    x = _to_float(x)

    pair = _offdiag_cosines(x)

    norms = x.norm(dim=-1)

    return {
        "stage": name,
        "n": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "cos_mean": float(pair.mean()),
        "cos_std": float(pair.std()),
        "cos_median": float(pair.median()),
        "cos_p05": float(torch.quantile(pair, 0.05)),
        "cos_p95": float(torch.quantile(pair, 0.95)),
        "cos_min": float(pair.min()),
        "cos_max": float(pair.max()),
        "effective_rank": _effective_rank(x),
        "effective_rank_fraction": (
                _effective_rank(x) / min(x.shape[0], x.shape[1])
        ),
    }


def _spearman_torch(
        a: torch.Tensor,
        b: torch.Tensor,
) -> float:
    """
    Spearman correlation without scipy dependency.
    """
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()

    if a.numel() != b.numel():
        raise ValueError("Spearman vectors must have same length")

    # rank approximation via double argsort.
    # Ties are rare for floating cosine similarities.
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()

    ra = ra - ra.mean()
    rb = rb - rb.mean()

    denom = torch.sqrt(
        ra.square().sum() * rb.square().sum()
    ).clamp_min(1e-12)

    return float((ra * rb).sum() / denom)


def _stage_distortion(
        before_name: str,
        before: torch.Tensor,
        after_name: str,
        after: torch.Tensor,
) -> Dict[str, float]:
    """
    Compare pairwise geometry before and after a transformation.
    """
    pre = _offdiag_cosines(before)
    post = _offdiag_cosines(after)

    if pre.numel() != post.numel():
        raise RuntimeError(
            f"Pair count mismatch: {before_name}={pre.numel()} "
            f"{after_name}={post.numel()}"
        )

    delta = post - pre

    return {
        "from_stage": before_name,
        "to_stage": after_name,
        "pre_cos_mean": float(pre.mean()),
        "post_cos_mean": float(post.mean()),
        "mean_cos_shift": float(delta.mean()),
        "median_cos_shift": float(delta.median()),
        "mean_abs_cos_shift": float(delta.abs().mean()),
        "p95_abs_cos_shift": float(
            torch.quantile(delta.abs(), 0.95)
        ),
        "pairwise_similarity_spearman": _spearman_torch(
            pre,
            post,
        ),
    }


# ============================================================
# Segment analysis
# ============================================================

def _segment_weight_summary(
        weights: torch.Tensor,
        present: torch.Tensor,
        names: List[str],
) -> pd.DataFrame:
    weights = _to_float(weights).cpu()

    present = present.detach().bool().cpu()

    rows = []

    dominant = weights.argmax(dim=1)

    for s, name in enumerate(names):
        mask = present[:, s]

        if mask.sum() == 0:
            rows.append({
                "segment": name,
                "present_n": 0,
                "present_rate": 0.0,
                "weight_mean": float("nan"),
                "weight_std": float("nan"),
                "weight_median": float("nan"),
                "weight_min": float("nan"),
                "weight_max": float("nan"),
                "dominant_rate": 0.0,
            })
            continue

        w = weights[mask, s]

        rows.append({
            "segment": name,
            "present_n": int(mask.sum()),
            "present_rate": float(mask.float().mean()),
            "weight_mean": float(w.mean()),
            "weight_std": float(w.std()),
            "weight_median": float(w.median()),
            "weight_min": float(w.min()),
            "weight_max": float(w.max()),
            "dominant_rate": float(
                ((dominant == s) & mask).float().mean()
            ),
        })

    return pd.DataFrame(rows)


def _gate_entropy(
        weights: torch.Tensor,
        present: torch.Tensor,
) -> Dict[str, float]:
    w = _to_float(weights).cpu()
    p = present.detach().bool().cpu()

    w = torch.where(
        p,
        w,
        torch.zeros_like(w),
    )

    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    entropy = -(
            w.clamp_min(1e-12) *
            w.clamp_min(1e-12).log()
    ).sum(dim=-1)

    n_present = p.sum(dim=-1).float()

    max_entropy = n_present.clamp_min(1.0).log()

    normalized = torch.where(
        max_entropy > 0,
        entropy / max_entropy.clamp_min(1e-12),
        torch.zeros_like(entropy),
    )

    return {
        "gate_entropy_mean": float(entropy.mean()),
        "gate_entropy_std": float(entropy.std()),
        "gate_entropy_normalized_mean": float(normalized.mean()),
        "gate_max_weight_mean": float(w.max(dim=-1).values.mean()),
        "gate_max_weight_median": float(w.max(dim=-1).values.median()),
    }


def _segment_geometry(
        segment_embs: torch.Tensor,
        present: torch.Tensor,
        names: List[str],
) -> pd.DataFrame:
    rows = []

    segment_embs = _to_float(segment_embs)
    present = present.detach().bool()

    for s, name in enumerate(names):
        mask = present[:, s]

        if int(mask.sum()) < 2:
            continue

        x = segment_embs[mask, s]

        row = _geometry_summary(
            f"segment:{name}",
            x,
        )

        row["segment"] = name
        row["present_n"] = int(mask.sum())

        rows.append(row)

    return pd.DataFrame(rows)


def _within_go_segment_agreement(
        segment_embs: torch.Tensor,
        present: torch.Tensor,
        names: List[str],
) -> pd.DataFrame:
    segment_embs = F.normalize(
        _to_float(segment_embs),
        dim=-1,
    )

    present = present.detach().bool()

    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mask = present[:, i] & present[:, j]

            if int(mask.sum()) == 0:
                continue

            cos = (
                    segment_embs[mask, i] *
                    segment_embs[mask, j]
            ).sum(dim=-1)

            rows.append({
                "segment_a": names[i],
                "segment_b": names[j],
                "n": int(mask.sum()),
                "cos_mean": float(cos.mean()),
                "cos_std": float(cos.std()),
                "cos_median": float(cos.median()),
                "cos_p05": float(torch.quantile(cos, 0.05)),
                "cos_p95": float(torch.quantile(cos, 0.95)),
            })

    return pd.DataFrame(rows)


# ============================================================
# Diagnostic entry point
# ============================================================

@torch.no_grad()
def run_go_representation_diagnostic(
        trainer,
        outdir: str | Path,
        *,
        chunk_size: int = 128,
):
    """
    Run GO representation pipeline diagnostic using an already-built
    Trainer with checkpoint loaded.

    This avoids reconstructing config/model/checkpoint logic inside
    the diagnostic script.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = trainer.model
    ctx = trainer.ctx
    device = trainer.device

    if getattr(model, "go_encoder", None) is None:
        raise RuntimeError(
            "GO diagnostic requires model.go_encoder"
        )

    if getattr(ctx, "go_encoder_output_mode", None) != "segment_pooled":
        raise RuntimeError(
            "This diagnostic expects "
            "go_encoder_output_mode='segment_pooled'. "
            f"Got {getattr(ctx, 'go_encoder_output_mode', None)!r}"
        )

    if not hasattr(trainer, "eval_id_list"):
        raise RuntimeError(
            "trainer.eval_id_list is missing"
        )

    eval_ids = [int(x) for x in trainer.eval_id_list]

    segment_names = list(model.go_segment_names)

    print("=" * 110)
    print("D11: GO REPRESENTATION PIPELINE DIAGNOSTIC")
    print("=" * 110)

    print(f"GO terms: {len(eval_ids)}")
    print(
        "representation_mode:",
        model.go_segment_representation_mode,
    )
    print(
        "segment_names:",
        segment_names,
    )

    if model.go_segment_representation_mode == "mixed":
        print(
            "mix_alpha:",
            float(model.go_segment_mix_alpha),
        )
    else:
        print(
            "mix_alpha: NOT USED "
            f"(mode={model.go_segment_representation_mode})"
        )

    was_training = model.training
    model.eval()

    all_segment_embs = []
    all_segment_weights = []
    all_segment_present = []
    all_pooled = []

    try:
        for start in range(0, len(eval_ids), chunk_size):
            end = min(
                start + chunk_size,
                len(eval_ids),
            )

            ids = eval_ids[start:end]

            toks = ctx.go_text_store.batch(ids)

            seg_input_ids = toks["seg_input_ids"].to(
                device,
                non_blocking=True,
            )

            seg_attention_mask = toks[
                "seg_attention_mask"
            ].to(
                device,
                non_blocking=True,
            )

            seg_present = toks["seg_present"].to(
                device,
                non_blocking=True,
            )

            input_ids = toks["input_ids"].to(
                device,
                non_blocking=True,
            )

            attention_mask = toks["attention_mask"].to(
                device,
                non_blocking=True,
            )

            out = model.encode_go_segment_aware(
                seg_input_ids=seg_input_ids,
                seg_attention_mask=seg_attention_mask,
                seg_present=seg_present,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            all_segment_embs.append(
                out["segment_embs"]
                .detach()
                .float()
                .cpu()
            )

            all_segment_weights.append(
                out["segment_weights"]
                .detach()
                .float()
                .cpu()
            )

            all_segment_present.append(
                out["segment_present"]
                .detach()
                .bool()
                .cpu()
            )

            all_pooled.append(
                out["pooled"]
                .detach()
                .float()
                .cpu()
            )

            print(
                f"[GO-DIAG] {end}/{len(eval_ids)}"
            )

    finally:
        if was_training:
            model.train()

    segment_embs = torch.cat(
        all_segment_embs,
        dim=0,
    )

    segment_weights = torch.cat(
        all_segment_weights,
        dim=0,
    )

    segment_present = torch.cat(
        all_segment_present,
        dim=0,
    )

    seg_pooled = torch.cat(
        all_pooled,
        dim=0,
    )

    if seg_pooled.shape[0] != len(eval_ids):
        raise RuntimeError(
            "GO count mismatch"
        )

    # --------------------------------------------------------
    # Projection stages
    # --------------------------------------------------------

    model.eval()

    stage_go_ln = []
    stage_projected = []
    stage_normalized = []

    for start in range(0, seg_pooled.shape[0], chunk_size):
        end = min(
            start + chunk_size,
            seg_pooled.shape[0],
        )

        x = seg_pooled[start:end].to(
            device,
            non_blocking=True,
        )

        ln = model.go_ln(x)
        proj = model.proj_g(ln)
        norm = F.normalize(
            proj.float(),
            dim=-1,
        )

        stage_go_ln.append(
            ln.detach().float().cpu()
        )

        stage_projected.append(
            proj.detach().float().cpu()
        )

        stage_normalized.append(
            norm.detach().float().cpu()
        )

    go_ln = torch.cat(stage_go_ln)
    projected = torch.cat(stage_projected)
    normalized = torch.cat(stage_normalized)

    # ========================================================
    # D11A
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D11A: SEGMENT AVAILABILITY + LEARNED WEIGHTS")
    print("=" * 110)

    weight_df = _segment_weight_summary(
        segment_weights,
        segment_present,
        segment_names,
    )

    print(
        weight_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    entropy_stats = _gate_entropy(
        segment_weights,
        segment_present,
    )

    print("\nGate summary:")
    for k, v in entropy_stats.items():
        print(f"{k}: {v:.6f}")

    # ========================================================
    # D11B
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D11B: INDIVIDUAL SEGMENT GEOMETRY")
    print("=" * 110)

    segment_geometry_df = _segment_geometry(
        segment_embs,
        segment_present,
        segment_names,
    )

    if len(segment_geometry_df):
        cols = [
            "segment",
            "present_n",
            "cos_mean",
            "cos_std",
            "cos_median",
            "cos_p95",
            "effective_rank",
            "effective_rank_fraction",
        ]

        print(
            segment_geometry_df[
                [c for c in cols if c in segment_geometry_df.columns]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    # ========================================================
    # D11C
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D11C: GO REPRESENTATION PIPELINE GEOMETRY")
    print("=" * 110)

    stage_tensors = {
        "seg_pooled_raw": seg_pooled,
        "go_ln": go_ln,
        "projected": projected,
        "normalized": normalized,
    }

    stage_rows = []

    for name, x in stage_tensors.items():
        print(f"Analyzing {name}...")
        stage_rows.append(
            _geometry_summary(name, x)
        )

    stage_df = pd.DataFrame(stage_rows)

    display_cols = [
        "stage",
        "n",
        "dim",
        "norm_mean",
        "cos_mean",
        "cos_std",
        "cos_median",
        "cos_p95",
        "effective_rank",
        "effective_rank_fraction",
    ]

    print(
        stage_df[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # D11D
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D11D: STAGE-TO-STAGE GEOMETRY DISTORTION")
    print("=" * 110)

    distortion_pairs = [
        ("seg_pooled_raw", seg_pooled, "go_ln", go_ln),
        ("go_ln", go_ln, "projected", projected),
        ("projected", projected, "normalized", normalized),
        ("seg_pooled_raw", seg_pooled, "normalized", normalized),
    ]

    distortion_rows = []

    for before_name, before, after_name, after in distortion_pairs:
        print(f"Analyzing {before_name} -> {after_name}...")

        row = _stage_distortion(
            before_name,
            before,
            after_name,
            after,
        )

        distortion_rows.append(row)

    distortion_df = pd.DataFrame(distortion_rows)

    print(
        distortion_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    # ========================================================
    # D11G: segment-specific projection geometry
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D11G: SEGMENT-SPECIFIC PROJECTION GEOMETRY")
    print("=" * 110)

    segment_projection_rows = []

    for s, segment_name in enumerate(segment_names):

        mask = segment_present[:, s]

        if int(mask.sum()) < 2:
            continue

        raw = segment_embs[mask, s].float()

        projected_chunks = []

        for start in range(0, raw.shape[0], chunk_size):
            end = min(
                start + chunk_size,
                raw.shape[0],
            )

            x = raw[start:end].to(
                device,
                non_blocking=True,
            )

            # Use exactly the same GO-side projection path
            # as the normal retrieval representation.
            x_ln = model.go_ln(x)
            x_proj = model.proj_g(x_ln)
            x_norm = F.normalize(
                x_proj.float(),
                dim=-1,
            )

            projected_chunks.append(
                x_norm.detach().cpu()
            )

        projected_segment = torch.cat(
            projected_chunks,
            dim=0,
        )

        raw_stats = _geometry_summary(
            f"{segment_name}:raw",
            raw,
        )

        proj_stats = _geometry_summary(
            f"{segment_name}:projected",
            projected_segment,
        )

        raw_pairs = _offdiag_cosines(raw)
        proj_pairs = _offdiag_cosines(
            projected_segment
        )

        pair_spearman = _spearman_torch(
            raw_pairs,
            proj_pairs,
        )

        segment_projection_rows.append({
            "segment": segment_name,
            "n": int(raw.shape[0]),

            "raw_cos_mean":
                raw_stats["cos_mean"],

            "raw_cos_std":
                raw_stats["cos_std"],

            "raw_effective_rank":
                raw_stats["effective_rank"],

            "projected_cos_mean":
                proj_stats["cos_mean"],

            "projected_cos_std":
                proj_stats["cos_std"],

            "projected_effective_rank":
                proj_stats["effective_rank"],

            "cosine_shift":
                proj_stats["cos_mean"]
                - raw_stats["cos_mean"],

            "pairwise_similarity_spearman":
                pair_spearman,
        })

    segment_projection_df = pd.DataFrame(
        segment_projection_rows
    )

    print(
        segment_projection_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    segment_projection_df.to_csv(
        outdir / "go_segment_projection_geometry.csv",
        index=False,
    )

    # ========================================================
    # D11E
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D11E: WITHIN-GO SEGMENT AGREEMENT")
    print("=" * 110)

    agreement_df = _within_go_segment_agreement(
        segment_embs,
        segment_present,
        segment_names,
    )

    if len(agreement_df):
        print(
            agreement_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
    else:
        print("No segment pairs with sufficient overlap.")

    # ========================================================
    # D11F: automatic interpretation
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D11F: AUTOMATIC GEOMETRY DIAGNOSIS")
    print("=" * 110)

    stage_lookup = {
        row["stage"]: row
        for row in stage_rows
    }

    raw_cos = stage_lookup["seg_pooled_raw"]["cos_mean"]
    ln_cos = stage_lookup["go_ln"]["cos_mean"]
    proj_cos = stage_lookup["projected"]["cos_mean"]
    norm_cos = stage_lookup["normalized"]["cos_mean"]

    raw_rank = stage_lookup["seg_pooled_raw"]["effective_rank"]
    proj_rank = stage_lookup["projected"]["effective_rank"]
    norm_rank = stage_lookup["normalized"]["effective_rank"]

    projection_shift = proj_cos - ln_cos

    proj_distortion_row = next(
        r
        for r in distortion_rows
        if r["from_stage"] == "go_ln"
        and r["to_stage"] == "projected"
    )

    proj_spearman = proj_distortion_row[
        "pairwise_similarity_spearman"
    ]

    print(f"seg_pooled mean cosine : {raw_cos:.6f}")
    print(f"go_ln mean cosine      : {ln_cos:.6f}")
    print(f"projected mean cosine  : {proj_cos:.6f}")
    print(f"normalized mean cosine : {norm_cos:.6f}")
    print()
    print(f"seg_pooled eff rank    : {raw_rank:.4f}")
    print(f"projected eff rank     : {proj_rank:.4f}")
    print(f"normalized eff rank    : {norm_rank:.4f}")
    print()
    print(
        "projection cosine shift:",
        f"{projection_shift:+.6f}",
    )
    print(
        "pre/post pairwise similarity Spearman:",
        f"{proj_spearman:.6f}",
    )

    # --------------------------------------------------------
    # These labels are intentionally conservative.
    #
    # They are not biological conclusions. They are merely
    # flags telling us which pipeline stage deserves inspection.
    # --------------------------------------------------------

    if raw_cos >= 0.70:
        diagnosis = "SEGMENTS_OR_MIXING_ALREADY_HIGHLY_COMPRESSED"

        explanation = (
            "The pooled GO representation is already highly "
            "similar across GO terms before proj_g. Inspect "
            "individual segment geometry and learned segment "
            "weights before changing the projection."
        )

    elif projection_shift >= 0.15:
        diagnosis = "PROJECTION_INTRODUCES_STRONG_COMPRESSION"

        explanation = (
            "GO representations are substantially more similar "
            "after proj_g than before it. The projection/alignment "
            "stage is the primary suspect."
        )

    elif projection_shift >= 0.07:
        diagnosis = "PROJECTION_INTRODUCES_MODERATE_COMPRESSION"

        explanation = (
            "proj_g noticeably compresses GO geometry, although "
            "the pre-projection space is not necessarily healthy "
            "or unhealthy by itself."
        )

    elif proj_spearman < 0.50:
        diagnosis = "PROJECTION_STRONGLY_REORDERS_GO_GEOMETRY"

        explanation = (
            "Mean cosine does not show a large compression, but "
            "pairwise semantic relationships are poorly preserved "
            "through proj_g."
        )

    else:
        diagnosis = "NO_SINGLE_MAJOR_PROJECTION_COLLAPSE_DETECTED"

        explanation = (
            "No single large collapse is localized to proj_g by "
            "these coarse geometry measures. Inspect segment-level "
            "geometry and gate behavior before modifying the model."
        )

    print()
    print("DIAGNOSIS:")
    print(diagnosis)
    print()
    print(explanation)

    # ========================================================
    # Save tensors required for later lightweight analyses
    # ========================================================

    print("\n")
    print("=" * 110)
    print("SAVING RESULTS")
    print("=" * 110)

    weight_df.to_csv(
        outdir / "go_segment_weights.csv",
        index=False,
    )

    segment_geometry_df.to_csv(
        outdir / "go_segment_geometry.csv",
        index=False,
    )

    stage_df.to_csv(
        outdir / "go_pipeline_geometry.csv",
        index=False,
    )

    distortion_df.to_csv(
        outdir / "go_pipeline_distortion.csv",
        index=False,
    )

    agreement_df.to_csv(
        outdir / "go_segment_agreement.csv",
        index=False,
    )

    np.save(
        outdir / "go_ids.npy",
        np.asarray(eval_ids, dtype=np.int64),
    )

    np.save(
        outdir / "go_segment_weights.float32.npy",
        segment_weights.numpy().astype(np.float32),
    )

    np.save(
        outdir / "go_segment_present.bool.npy",
        segment_present.numpy().astype(bool),
    )

    np.save(
        outdir / "go_seg_pooled.float32.npy",
        seg_pooled.numpy().astype(np.float32),
    )

    np.save(
        outdir / "go_ln.float32.npy",
        go_ln.numpy().astype(np.float32),
    )

    np.save(
        outdir / "go_projected.float32.npy",
        projected.numpy().astype(np.float32),
    )

    np.save(
        outdir / "go_normalized.float32.npy",
        normalized.numpy().astype(np.float32),
    )

    summary = {
        "n_go": len(eval_ids),
        "segment_names": segment_names,
        "representation_mode": str(
            model.go_segment_representation_mode
        ),
        "mix_alpha": (
            float(model.go_segment_mix_alpha)
            if model.go_segment_representation_mode == "mixed"
            else None
        ),
        "gate": entropy_stats,
        "diagnosis": diagnosis,
        "diagnosis_explanation": explanation,
        "seg_pooled_mean_cosine": raw_cos,
        "go_ln_mean_cosine": ln_cos,
        "projected_mean_cosine": proj_cos,
        "normalized_mean_cosine": norm_cos,
        "projection_cosine_shift": projection_shift,
        "projection_pairwise_spearman": proj_spearman,
        "seg_pooled_effective_rank": raw_rank,
        "projected_effective_rank": proj_rank,
        "normalized_effective_rank": norm_rank,
    }

    with open(
            outdir / "go_diagnostic_summary.json",
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("Saved:")
    for path in [
        outdir / "go_segment_weights.csv",
        outdir / "go_segment_geometry.csv",
        outdir / "go_pipeline_geometry.csv",
        outdir / "go_pipeline_distortion.csv",
        outdir / "go_segment_agreement.csv",
        outdir / "go_diagnostic_summary.json",
        outdir / "go_segment_projection_geometry.csv",
    ]:
        print(path)

    return {
        "segment_weights": weight_df,
        "segment_geometry": segment_geometry_df,
        "pipeline_geometry": stage_df,
        "pipeline_distortion": distortion_df,
        "segment_agreement": agreement_df,
        "summary": summary,
    }


# ============================================================
# Standalone CLI
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose GO representation geometry through the full encoding/projection pipeline."
    )

    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training YAML config.",
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint to diagnose.",
    )

    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory for diagnostic outputs.",
    )

    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. cuda:0 or cpu",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="GO encoding/projection batch size.",
    )

    p.add_argument(
        "--strict-exact",
        action="store_true",
        help="Require exact checkpoint/model key match.",
    )

    return p.parse_args()


def main():
    cli = parse_args()
    setup_logging_simple()

    # --------------------------------------------------------
    # Same configuration path as candidate dump script
    # --------------------------------------------------------

    args = configure(cli.config)

    if cli.device is not None:
        args.general_device = cli.device

    # Diagnostic explicitly loads checkpoint weights.
    # Do not allow resume/warmstart to interfere.
    args.resume = None
    args.warmstart_path = None
    args.eval_only = True
    args.wandb = False

    set_seed(args.seed)

    device = torch.device(
        args.general_device
        if args.general_device
        else (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    logging.info(
        "[GO-DIAG] device=%s",
        str(device),
    )

    # --------------------------------------------------------
    # Build same GO/model infrastructure as actual evaluation
    # --------------------------------------------------------

    logging.info("[GO-DIAG] loading GO cache")
    go_cache = build_go_cache(
        str(args.go_cache_path)
    )

    dag_parents = (
        dump_base.base.load_go_parents()
        if args.use_dag_in_ds
        else None
    )

    dag_children = (
        dump_base.base.load_go_children()
        if args.use_dag_in_ds
        else None
    )

    logging.info(
        "[GO-DIAG] building GO encoder/text store"
    )

    go_encoder, go_text_store = (
        build_go_encoder_and_text_store(
            args,
            device,
        )
    )

    logging.info(
        "[GO-DIAG] materializing GO text tokens"
    )

    go_text_store.materialize_tokens_once(
        batch_size=512,
        show_progress=True,
    )

    # --------------------------------------------------------
    # Canonical GO alignment
    # --------------------------------------------------------

    aligned = canonicalize_and_align_inputs(
        args=args,
        go_cache=go_cache,
        logger=logging.getLogger("go_diag"),
    )

    # --------------------------------------------------------
    # Trainer builder currently expects datasets/stores.
    # We construct them exactly as dump/eval does.
    # --------------------------------------------------------

    logging.info(
        "[GO-DIAG] building residue store"
    )

    res_store = build_stores(args)

    logging.info(
        "[GO-DIAG] building datasets"
    )

    datasets = build_datasets(
        args,
        res_store,
        go_text_store,
        dag_parents=dag_parents,
        pid2pos=aligned["pid2pos"],
        zs=aligned["zs"],
        fs=aligned["fs"],
    )

    logging.info(
        "[GO-DIAG] building trainer"
    )

    trainer = build_trainer_for_dump(
        args=args,
        device=device,
        go_cache=go_cache,
        go_encoder=go_encoder,
        go_text_store=go_text_store,
        datasets=datasets,
        aligned=aligned,
        dag_parents=dag_parents,
        dag_children=dag_children,
    )

    # --------------------------------------------------------
    # Load exact checkpoint under investigation
    # --------------------------------------------------------

    logging.info(
        "[GO-DIAG] loading checkpoint: %s",
        cli.checkpoint,
    )

    load_info = load_model_weights_only(
        trainer.model,
        cli.checkpoint,
        device=device,
        strict_exact=bool(cli.strict_exact),
    )

    logging.info(
        "[GO-DIAG] checkpoint meta: %s",
        load_info.get("meta", {}),
    )

    trainer.model.eval()

    # --------------------------------------------------------
    # Run D11
    # --------------------------------------------------------

    results = run_go_representation_diagnostic(
        trainer=trainer,
        outdir=Path(cli.outdir),
        chunk_size=int(cli.chunk_size),
    )

    print()
    print("=" * 110)
    print("D11 COMPLETE")
    print("=" * 110)

    summary = results["summary"]

    print(
        "Diagnosis:",
        summary["diagnosis"],
    )

    print(
        "seg_pooled cosine:",
        f'{summary["seg_pooled_mean_cosine"]:.6f}',
    )

    print(
        "go_ln cosine:",
        f'{summary["go_ln_mean_cosine"]:.6f}',
    )

    print(
        "projected cosine:",
        f'{summary["projected_mean_cosine"]:.6f}',
    )

    print(
        "normalized cosine:",
        f'{summary["normalized_mean_cosine"]:.6f}',
    )

    print(
        "projection shift:",
        f'{summary["projection_cosine_shift"]:+.6f}',
    )

    print(
        "projection pairwise Spearman:",
        f'{summary["projection_pairwise_spearman"]:.6f}',
    )

    print()
    print("Outputs:")
    print(Path(cli.outdir).resolve())


if __name__ == "__main__":
    main()