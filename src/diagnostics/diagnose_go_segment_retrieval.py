from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

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
    make_dump_loader,
)

from src.script import dump_retriever_candidates_global_local as dump_base

SEGMENT_NAMES = [
    "name",
    "namespace",
    "definition",
    "is_a",
    "part_of",
]

MODES = [
    "pooled",
    "name",
    "definition",
    "is_a",
]

KS = [50, 100, 200, 500]


def card_bin(n: int) -> str:
    if n <= 5:
        return "1_5"
    if n <= 10:
        return "6_10"
    if n <= 20:
        return "11_20"
    if n <= 40:
        return "21_40"
    if n <= 80:
        return "41_80"
    if n <= 160:
        return "81_160"
    return "161plus"


@torch.no_grad()
def encode_go_banks(
        trainer,
        chunk_size: int = 128,
) -> Dict[str, torch.Tensor]:
    model = trainer.model
    device = trainer.device
    store = trainer.ctx.go_text_store

    # CRITICAL:
    # Use the exact runtime GO order used by _build_eval_space.
    eval_ids = [
        int(x)
        for x in trainer._eval_ids_cpu.tolist()
    ]

    model.eval()

    buffers = {
        mode: []
        for mode in MODES
    }

    segment_index = {
        name: i
        for i, name in enumerate(
            model.go_segment_names
        )
    }

    for required in [
        "name",
        "definition",
        "is_a",
    ]:
        if required not in segment_index:
            raise RuntimeError(
                f"Segment {required!r} missing from "
                f"{model.go_segment_names}"
            )

    for start in tqdm(
            range(
                0,
                len(eval_ids),
                chunk_size,
            ),
            desc="encode raw GO banks",
    ):
        end = min(
            start + chunk_size,
            len(eval_ids),
        )

        gids = eval_ids[start:end]

        toks = store.batch(gids)

        seg_input_ids = toks[
            "seg_input_ids"
        ].to(
            device,
            non_blocking=True,
        )

        seg_attention_mask = toks[
            "seg_attention_mask"
        ].to(
            device,
            non_blocking=True,
        )

        seg_present = toks[
            "seg_present"
        ].to(
            device,
            non_blocking=True,
        )

        input_ids = toks[
            "input_ids"
        ].to(
            device,
            non_blocking=True,
        )

        attention_mask = toks[
            "attention_mask"
        ].to(
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

        segment_embs = out["segment_embs"]
        pooled = out["pooled"]

        raw_by_mode = {
            "pooled": pooled,
            "name": segment_embs[
                :,
                segment_index["name"],
                :,
            ],
            "definition": segment_embs[
                :,
                segment_index["definition"],
                :,
            ],
            "is_a": segment_embs[
                :,
                segment_index["is_a"],
                :,
            ],
        }

        # CRITICAL:
        # DO NOT apply go_ln / proj_g / normalization here.
        #
        # score_from_encoded_experts expects the same raw GO
        # representation stage returned by _build_eval_space.
        for mode, raw in raw_by_mode.items():
            buffers[mode].append(
                raw.detach()
                .float()
                .cpu()
            )

    banks = {
        mode: torch.cat(
            chunks,
            dim=0,
        )
        for mode, chunks in buffers.items()
    }

    n_expected = len(eval_ids)

    for mode, bank in banks.items():
        if bank.shape[0] != n_expected:
            raise RuntimeError(
                f"{mode}: expected {n_expected} GO terms, "
                f"got {bank.shape[0]}"
            )

        print(
            f"[GO-BANK] {mode:10s} "
            f"shape={tuple(bank.shape)} "
            f"norm={bank.norm(dim=-1).mean().item():.4f}"
        )

    return banks


@torch.no_grad()
def evaluate_loader(
        trainer,
        loader,
        banks: Dict[str, torch.Tensor],
        split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = trainer.device
    model = trainer.model

    model.eval()

    scale = trainer.logit_scale_tensor()

    stats = {
        mode: {
            "gold_total": 0,
            **{
                f"hits@{k}": 0
                for k in KS
            },
        }
        for mode in MODES
    }

    card_stats = {
        mode: {}
        for mode in MODES
    }

    card_bins = [
        "1_5",
        "6_10",
        "11_20",
        "21_40",
        "41_80",
        "81_160",
        "161plus",
    ]

    for mode in MODES:
        for bname in card_bins:
            card_stats[
                mode
            ][
                bname
            ] = {
                "n": 0,
                "gold_total": 0,
                **{
                    f"hits@{k}": 0
                    for k in KS
                },
            }

    sanity_done = False

    # Put GO banks on GPU once.
    banks_device = {
        mode: bank.to(
            device,
            non_blocking=True,
        )
        for mode, bank in banks.items()
    }

    for batch in tqdm(
            loader,
            desc=f"D12 retrieval {split_name}",
    ):

        H = batch[
            "prot_emb_pad"
        ].to(
            device,
            non_blocking=True,
        )

        if trainer.to_f32 is not None:
            H = trainer.to_f32(H)

        attn_valid, _ = (
            trainer._valid_and_pad_masks(
                batch
            )
        )

        # Exact normal evaluation space.
        G_eval, y_true = (
            trainer._build_eval_space(
                batch
            )
        )

        B = int(
            H.size(0)
        )

        # Exact protein-side representation.
        encoded = (
            model.encode_protein_experts(
                H,
                attn_valid,
            )
        )

        # ====================================================
        # CRITICAL SANITY CHECK
        #
        # Our pooled raw GO bank MUST reproduce G_eval.
        # If not, D12 is invalid and stops immediately.
        # ====================================================

        if not sanity_done:

            pooled = banks_device[
                "pooled"
            ]

            pooled_batch = (
                pooled
                .unsqueeze(0)
                .expand(
                    B,
                    -1,
                    -1,
                )
            )

            print(
                "[D12-SANITY] G_eval shape:",
                tuple(
                    G_eval.shape
                ),
            )

            print(
                "[D12-SANITY] pooled shape:",
                tuple(
                    pooled_batch.shape
                ),
            )

            print(
                "[D12-SANITY] G_eval norm mean:",
                float(
                    G_eval
                    .float()
                    .norm(dim=-1)
                    .mean()
                    .item()
                ),
            )

            print(
                "[D12-SANITY] pooled norm mean:",
                float(
                    pooled_batch
                    .float()
                    .norm(dim=-1)
                    .mean()
                    .item()
                ),
            )

            if (
                    G_eval.shape
                    != pooled_batch.shape
            ):
                raise RuntimeError(
                    "D12 pooled shape does not "
                    "match normal G_eval."
                )

            cos = F.cosine_similarity(
                G_eval
                .float()
                .reshape(
                    -1,
                    G_eval.shape[-1],
                ),
                pooled_batch
                .float()
                .reshape(
                    -1,
                    pooled_batch.shape[-1],
                ),
                dim=-1,
            )

            max_abs_diff = (
                    G_eval.float()
                    - pooled_batch.float()
            ).abs().max().item()

            print(
                "[D12-SANITY] G_eval vs pooled "
                "cosine mean/min/max = "
                f"{cos.mean().item():.8f} / "
                f"{cos.min().item():.8f} / "
                f"{cos.max().item():.8f}"
            )

            print(
                "[D12-SANITY] G_eval vs pooled "
                f"max_abs_diff = "
                f"{max_abs_diff:.8e}"
            )

            # Require essentially identical directions.
            if cos.min().item() < 0.999:
                raise RuntimeError(
                    "D12 pooled GO bank does not "
                    "reproduce normal G_eval. "
                    "Stopping before retrieval."
                )

            sanity_done = True

        n_gold = (
                y_true > 0
        ).sum(
            dim=1
        )

        # ====================================================
        # Four GO representation modes
        # ====================================================

        for mode in MODES:

            G_bank = banks_device[
                mode
            ]

            G = (
                G_bank
                .unsqueeze(0)
                .expand(
                    B,
                    -1,
                    -1,
                )
            )

            scores_raw, _ = (
                model.score_from_encoded_experts(
                    encoded=encoded,
                    G=G,
                    return_components=True,
                )
            )

            if scores_raw.shape != (
                    B,
                    G_bank.shape[0],
            ):
                raise RuntimeError(
                    f"Unexpected score shape "
                    f"for {mode}: "
                    f"{tuple(scores_raw.shape)}"
                )

            scores = (
                    scores_raw
                    * scale
            ).float()

            max_k = min(
                max(KS),
                scores.shape[1],
            )

            top_idx = torch.topk(
                scores,
                k=max_k,
                dim=1,
                largest=True,
                sorted=True,
            ).indices

            for i in range(B):

                ng = int(
                    n_gold[
                        i
                    ].item()
                )

                if ng <= 0:
                    continue

                bname = card_bin(
                    ng
                )

                stats[
                    mode
                ][
                    "gold_total"
                ] += ng

                card_stats[
                    mode
                ][
                    bname
                ][
                    "n"
                ] += 1

                card_stats[
                    mode
                ][
                    bname
                ][
                    "gold_total"
                ] += ng

                for k in KS:
                    kk = min(
                        k,
                        top_idx.shape[1],
                    )

                    cols = top_idx[
                        i,
                        :kk,
                    ]

                    hits = int(
                        (
                                y_true[
                                    i
                                ].index_select(
                                    0,
                                    cols,
                                ) > 0
                        )
                        .sum()
                        .item()
                    )

                    stats[
                        mode
                    ][
                        f"hits@{k}"
                    ] += hits

                    card_stats[
                        mode
                    ][
                        bname
                    ][
                        f"hits@{k}"
                    ] += hits

    # ========================================================
    # Aggregate
    # ========================================================

    overall_rows = []

    for mode in MODES:

        gold_total = stats[
            mode
        ][
            "gold_total"
        ]

        row = {
            "split": split_name,
            "go_repr": mode,
            "gold_total": gold_total,
        }

        for k in KS:
            row[
                f"coverage@{k}"
            ] = (
                    stats[
                        mode
                    ][
                        f"hits@{k}"
                    ]
                    / max(
                gold_total,
                1,
            )
            )

        overall_rows.append(
            row
        )

    card_rows = []

    for mode in MODES:

        for bname in card_bins:

            s = card_stats[
                mode
            ][
                bname
            ]

            if s["n"] == 0:
                continue

            row = {
                "split": split_name,
                "go_repr": mode,
                "card_bin": bname,
                "n": s["n"],
                "gold_total": s[
                    "gold_total"
                ],
            }

            for k in KS:
                row[
                    f"coverage@{k}"
                ] = (
                        s[
                            f"hits@{k}"
                        ]
                        / max(
                    s[
                        "gold_total"
                    ],
                    1,
                )
                )

            card_rows.append(
                row
            )

    return (
        pd.DataFrame(
            overall_rows
        ),
        pd.DataFrame(
            card_rows
        ),
    )

def analyze_isa_redundancy(
        jsonl_path: str | Path,
        eval_go_ids: List[int],
) -> tuple[pd.DataFrame, dict]:
    jsonl_path = Path(jsonl_path)

    eval_str = {
        f"GO:{int(x):07d}"
        for x in eval_go_ids
    }

    rows = []

    with open(
            jsonl_path,
            "r",
            encoding="utf-8",
    ) as f:
        for line in f:
            obj = json.loads(line)

            gid = str(
                obj.get(
                    "go_id",
                    "",
                )
            )

            if gid not in eval_str:
                continue

            parent_ids = tuple(
                sorted(
                    str(x)
                    for x in obj.get(
                        "is_a_parent_ids",
                        [],
                    )
                )
            )

            isa_text = str(
                obj.get(
                    "segments",
                    {},
                ).get(
                    "is_a",
                    "",
                )
            ).strip()

            rows.append({
                "go_id": gid,
                "is_a_text": isa_text,
                "parent_ids": "|".join(
                    parent_ids
                ),
                "n_parents": len(
                    parent_ids
                ),
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No evaluation GO IDs found in segmented JSONL"
        )

    text_counts = Counter(
        df["is_a_text"].tolist()
    )

    parent_counts = Counter(
        df["parent_ids"].tolist()
    )

    df[
        "same_is_a_text_group_size"
    ] = df[
        "is_a_text"
    ].map(text_counts)

    df[
        "same_parent_set_group_size"
    ] = df[
        "parent_ids"
    ].map(parent_counts)

    n = len(df)

    nonempty_text = (
            df["is_a_text"] != ""
    )

    nonempty_parent = (
            df["parent_ids"] != ""
    )

    summary = {
        "n_eval_go": n,

        "unique_is_a_texts": int(
            df.loc[
                nonempty_text,
                "is_a_text",
            ].nunique()
        ),

        "unique_parent_sets": int(
            df.loc[
                nonempty_parent,
                "parent_ids",
            ].nunique()
        ),

        "fraction_go_with_shared_is_a_text": float(
            (
                    df[
                        "same_is_a_text_group_size"
                    ] > 1
            ).mean()
        ),

        "fraction_go_with_shared_parent_set": float(
            (
                    df[
                        "same_parent_set_group_size"
                    ] > 1
            ).mean()
        ),

        "mean_same_is_a_text_group_size": float(
            df[
                "same_is_a_text_group_size"
            ].mean()
        ),

        "median_same_is_a_text_group_size": float(
            df[
                "same_is_a_text_group_size"
            ].median()
        ),

        "max_same_is_a_text_group_size": int(
            df[
                "same_is_a_text_group_size"
            ].max()
        ),

        "mean_same_parent_set_group_size": float(
            df[
                "same_parent_set_group_size"
            ].mean()
        ),

        "max_same_parent_set_group_size": int(
            df[
                "same_parent_set_group_size"
            ].max()
        ),

        "mean_n_is_a_parents": float(
            df[
                "n_parents"
            ].mean()
        ),

        "max_n_is_a_parents": int(
            df[
                "n_parents"
            ].max()
        ),
    }

    return df, summary


def build_runtime(
        cli,
        *,
        ids_path: str | None,
):
    args = configure(
        cli.config
    )

    if cli.device is not None:
        args.general_device = (
            cli.device
        )

    if ids_path is not None:
        args.val_ids_path = Path(
            ids_path
        )

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

    go_cache = build_go_cache(
        str(
            args.go_cache_path
        )
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

    (
        go_encoder,
        go_text_store,
    ) = (
        build_go_encoder_and_text_store(
            args,
            device,
        )
    )

    go_text_store.materialize_tokens_once(
        batch_size=512,
        show_progress=True,
    )

    aligned = (
        canonicalize_and_align_inputs(
            args=args,
            go_cache=go_cache,
            logger=logging.getLogger(
                "D12"
            ),
        )
    )

    res_store = build_stores(
        args
    )

    datasets = build_datasets(
        args,
        res_store,
        go_text_store,
        dag_parents=dag_parents,
        pid2pos=aligned[
            "pid2pos"
        ],
        zs=aligned["zs"],
        fs=aligned["fs"],
    )

    loader = make_dump_loader(
        dataset=datasets["val"],
        args=args,
        go_text_store=go_text_store,
        batch_size=int(
            cli.batch_size
            or args.eval_batch_size
            or args.batch_size
        ),
        num_workers=int(
            cli.num_workers
        ),
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

    load_model_weights_only(
        trainer.model,
        cli.checkpoint,
        device=device,
        strict_exact=bool(
            cli.strict_exact
        ),
    )

    trainer.model.eval()

    # Recreate normal evaluation GO alignment/cache.
    trainer._refresh_eval_go_cache(
        chunk=trainer.cfg.eval_go_bs
    )

    trainer._eval_cache_ready = False

    trainer._ensure_eval_cache_v2(
        chunk=trainer.cfg.eval_go_bs
    )

    return (
        args,
        trainer,
        loader,
        aligned,
    )


def parse_args():
    p = argparse.ArgumentParser(
        "D12 final GO segment retrieval diagnostic"
    )

    p.add_argument(
        "--config",
        required=True,
    )

    p.add_argument(
        "--checkpoint",
        required=True,
    )

    p.add_argument(
        "--valid_ids",
        required=True,
    )

    p.add_argument(
        "--test_ids",
        required=True,
    )

    p.add_argument(
        "--go_jsonl",
        required=True,
    )

    p.add_argument(
        "--outdir",
        required=True,
    )

    p.add_argument(
        "--device",
        default="cuda:0",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--chunk_size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--strict_exact",
        action="store_true",
    )

    return p.parse_args()


def main():
    cli = parse_args()
    setup_logging_simple()

    outdir = Path(cli.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # VALID
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D12: BUILDING VALID RUNTIME")
    print("=" * 110)

    (
        valid_args,
        valid_trainer,
        valid_loader,
        valid_aligned,
    ) = build_runtime(
        cli,
        ids_path=cli.valid_ids,
    )

    print("\n")
    print("=" * 110)
    print("D12A: BUILDING VALID GO SEGMENT BANKS")
    print("=" * 110)

    valid_banks = encode_go_banks(
        valid_trainer,
        chunk_size=int(
            cli.chunk_size
        ),
    )

    print("\n")
    print("=" * 110)
    print("D12A: VALID SEGMENT RETRIEVAL")
    print("=" * 110)

    (
        valid_overall,
        valid_card,
    ) = evaluate_loader(
        trainer=valid_trainer,
        loader=valid_loader,
        banks=valid_banks,
        split_name="valid",
    )

    print("\nVALID OVERALL")
    print(
        valid_overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nVALID BY CARDINALITY")
    print(
        valid_card.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Free VALID model before rebuilding TEST.
    del valid_banks
    del valid_loader
    del valid_trainer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # TEST
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D12: BUILDING TEST RUNTIME")
    print("=" * 110)

    (
        test_args,
        test_trainer,
        test_loader,
        test_aligned,
    ) = build_runtime(
        cli,
        ids_path=cli.test_ids,
    )

    print("\n")
    print("=" * 110)
    print("D12A: BUILDING TEST GO SEGMENT BANKS")
    print("=" * 110)

    test_banks = encode_go_banks(
        test_trainer,
        chunk_size=int(
            cli.chunk_size
        ),
    )

    print("\n")
    print("=" * 110)
    print("D12A: TEST SEGMENT RETRIEVAL")
    print("=" * 110)

    (
        test_overall,
        test_card,
    ) = evaluate_loader(
        trainer=test_trainer,
        loader=test_loader,
        banks=test_banks,
        split_name="test",
    )

    print("\nTEST OVERALL")
    print(
        test_overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nTEST BY CARDINALITY")
    print(
        test_card.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # Combine VALID + TEST
    # ========================================================

    overall_df = pd.concat(
        [
            valid_overall,
            test_overall,
        ],
        ignore_index=True,
    )

    card_df = pd.concat(
        [
            valid_card,
            test_card,
        ],
        ignore_index=True,
    )

    # Stable ordering.
    mode_order = {
        mode: i
        for i, mode in enumerate(MODES)
    }

    card_order = {
        "1_5": 0,
        "6_10": 1,
        "11_20": 2,
        "21_40": 3,
        "41_80": 4,
        "81_160": 5,
        "161plus": 6,
    }

    overall_df["_mode_order"] = (
        overall_df["go_repr"]
        .map(mode_order)
    )

    overall_df["_split_order"] = (
        overall_df["split"]
        .map({
            "valid": 0,
            "test": 1,
        })
    )

    overall_df = (
        overall_df
        .sort_values(
            [
                "_split_order",
                "_mode_order",
            ]
        )
        .drop(
            columns=[
                "_mode_order",
                "_split_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    card_df["_mode_order"] = (
        card_df["go_repr"]
        .map(mode_order)
    )

    card_df["_card_order"] = (
        card_df["card_bin"]
        .map(card_order)
    )

    card_df["_split_order"] = (
        card_df["split"]
        .map({
            "valid": 0,
            "test": 1,
        })
    )

    card_df = (
        card_df
        .sort_values(
            [
                "_split_order",
                "_mode_order",
                "_card_order",
            ]
        )
        .drop(
            columns=[
                "_mode_order",
                "_card_order",
                "_split_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # D12B: is_a redundancy
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D12B: IS_A REDUNDANCY")
    print("=" * 110)

    # GO evaluation space itself is the same observed GO bank.
    # Use TEST trainer's canonical eval IDs.
    eval_go_ids = [
        int(x)
        for x in test_trainer.eval_id_list
    ]

    (
        isa_df,
        isa_summary,
    ) = analyze_isa_redundancy(
        jsonl_path=cli.go_jsonl,
        eval_go_ids=eval_go_ids,
    )

    print(
        f"Evaluation GO terms              : "
        f"{isa_summary['n_eval_go']}"
    )

    print(
        f"Unique is_a texts                : "
        f"{isa_summary['unique_is_a_texts']}"
    )

    print(
        f"Unique parent sets               : "
        f"{isa_summary['unique_parent_sets']}"
    )

    print(
        f"GO with shared is_a text         : "
        f"{isa_summary['fraction_go_with_shared_is_a_text']:.4f}"
    )

    print(
        f"GO with shared parent set        : "
        f"{isa_summary['fraction_go_with_shared_parent_set']:.4f}"
    )

    print(
        f"Mean same-is_a group size        : "
        f"{isa_summary['mean_same_is_a_text_group_size']:.2f}"
    )

    print(
        f"Median same-is_a group size      : "
        f"{isa_summary['median_same_is_a_text_group_size']:.2f}"
    )

    print(
        f"Max same-is_a group size         : "
        f"{isa_summary['max_same_is_a_text_group_size']}"
    )

    print(
        f"Mean same-parent-set group size  : "
        f"{isa_summary['mean_same_parent_set_group_size']:.2f}"
    )

    print(
        f"Max same-parent-set group size   : "
        f"{isa_summary['max_same_parent_set_group_size']}"
    )

    print(
        f"Mean number of is_a parents      : "
        f"{isa_summary['mean_n_is_a_parents']:.2f}"
    )

    print(
        f"Max number of is_a parents       : "
        f"{isa_summary['max_n_is_a_parents']}"
    )

    # Largest redundancy groups, useful for sanity checking.
    text_groups = (
        isa_df[
            [
                "is_a_text",
                "same_is_a_text_group_size",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "same_is_a_text_group_size",
            ascending=False,
        )
        .head(20)
    )

    parent_groups = (
        isa_df[
            [
                "parent_ids",
                "same_parent_set_group_size",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "same_parent_set_group_size",
            ascending=False,
        )
        .head(20)
    )

    print("\nLargest shared is_a text groups:")
    print(
        text_groups.to_string(
            index=False,
        )
    )

    print("\nLargest shared parent-set groups:")
    print(
        parent_groups.to_string(
            index=False,
        )
    )

    # ========================================================
    # D12C: final comparison relative to learned pooled
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D12C: DELTA VS LEARNED POOLED REPRESENTATION")
    print("=" * 110)

    delta_rows = []

    for split in [
        "valid",
        "test",
    ]:
        sdf = overall_df[
            overall_df["split"] == split
            ]

        pooled_rows = sdf[
            sdf["go_repr"] == "pooled"
            ]

        if len(pooled_rows) != 1:
            raise RuntimeError(
                f"Expected one pooled row for {split}"
            )

        pooled = pooled_rows.iloc[0]

        for _, row in sdf.iterrows():

            if row["go_repr"] == "pooled":
                continue

            out = {
                "split": split,
                "go_repr": row[
                    "go_repr"
                ],
            }

            for k in KS:
                metric = (
                    f"coverage@{k}"
                )

                out[
                    f"delta_{metric}"
                ] = (
                        float(row[metric])
                        - float(
                    pooled[metric]
                )
                )

            delta_rows.append(out)

    delta_df = pd.DataFrame(
        delta_rows
    )

    print(
        delta_df.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    # High-cardinality TEST deltas.
    high_card_rows = []

    test_card_only = card_df[
        card_df["split"] == "test"
        ]

    for bname in [
        "41_80",
        "81_160",
        "161plus",
    ]:

        bdf = test_card_only[
            test_card_only[
                "card_bin"
            ] == bname
            ]

        pooled_rows = bdf[
            bdf["go_repr"] == "pooled"
            ]

        if len(pooled_rows) != 1:
            continue

        pooled = pooled_rows.iloc[0]

        for _, row in bdf.iterrows():

            if row[
                "go_repr"
            ] == "pooled":
                continue

            out = {
                "card_bin": bname,
                "go_repr": row[
                    "go_repr"
                ],
                "n": int(
                    row["n"]
                ),
            }

            for k in KS:
                metric = (
                    f"coverage@{k}"
                )

                out[
                    f"delta_{metric}"
                ] = (
                        float(
                            row[metric]
                        )
                        - float(
                    pooled[metric]
                )
                )

            high_card_rows.append(
                out
            )

    high_card_delta_df = (
        pd.DataFrame(
            high_card_rows
        )
    )

    print(
        "\nTEST HIGH-CARDINALITY DELTA VS POOLED"
    )

    print(
        high_card_delta_df.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    # ========================================================
    # Save
    # ========================================================

    overall_path = (
            outdir
            / "segment_retrieval_overall.csv"
    )

    card_path = (
            outdir
            / "segment_retrieval_by_cardinality.csv"
    )

    delta_path = (
            outdir
            / "segment_retrieval_delta_vs_pooled.csv"
    )

    high_card_delta_path = (
            outdir
            / "segment_retrieval_high_cardinality_delta.csv"
    )

    isa_path = (
            outdir
            / "is_a_redundancy_per_go.csv"
    )

    isa_summary_path = (
            outdir
            / "is_a_redundancy_summary.json"
    )

    overall_df.to_csv(
        overall_path,
        index=False,
    )

    card_df.to_csv(
        card_path,
        index=False,
    )

    delta_df.to_csv(
        delta_path,
        index=False,
    )

    high_card_delta_df.to_csv(
        high_card_delta_path,
        index=False,
    )

    isa_df.to_csv(
        isa_path,
        index=False,
    )

    with open(
            isa_summary_path,
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            isa_summary,
            f,
            indent=2,
        )

    # ========================================================
    # Final compact summary
    # ========================================================

    print("\n")
    print("=" * 110)
    print("D12 COMPLETE")
    print("=" * 110)

    print("\nOVERALL RETRIEVAL")
    print(
        overall_df[
            [
                "split",
                "go_repr",
                "coverage@50",
                "coverage@100",
                "coverage@200",
                "coverage@500",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nRemember:"
    )

    print(
        "This is an inference-only ablation of a checkpoint "
        "trained with learned pooled GO representations."
    )

    print(
        "A single-segment bank performing worse does NOT prove "
        "that a model trained specifically with that segment "
        "would perform worse."
    )

    print("\nSaved:")
    for p in [
        overall_path,
        card_path,
        delta_path,
        high_card_delta_path,
        isa_path,
        isa_summary_path,
    ]:
        print(p)


if __name__ == "__main__":
    main()