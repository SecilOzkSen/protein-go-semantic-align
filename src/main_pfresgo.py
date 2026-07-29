"""PFresGO entry point for the b1-v1-refinement branch.

This adapter keeps the training implementation in ``src.main`` and replaces
only dataset-specific resource loaders.  It uses PFresGO's supplied splits,
the full active GO branch as retrieval space, and derives seen/rare/zero-shot
sets from the training split without creating extra preprocessing files.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

import src.main as base
from src.utils.helpers import load_go_texts_canonical


def _go_int(value: Any) -> int:
    text = str(value).strip()
    return int(text.split(":", 1)[1] if text.upper().startswith("GO:") else text)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_ids(path: Path) -> List[int]:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    elif suffix == ".json":
        value = _read_json(path)
        if isinstance(value, dict):
            value = value.get("ids", value.get("go_ids", value.get("terms", value)))
    else:
        value = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if isinstance(value, dict):
        value = list(value.keys())
    return sorted({_go_int(x) for x in value})


def _active_go_terms(go_vocab_path: Path) -> Dict[str, dict]:
    raw = _read_json(go_vocab_path)
    return {
        str(go_id): info
        for go_id, info in raw.items()
        if isinstance(info, dict) and not bool(info.get("is_obsolete", False))
    }


def _build_dag(go_vocab_path: Path) -> Tuple[dict, dict]:
    parents: Dict[int, List[Tuple[int, str]]] = {}
    children: Dict[int, List[Tuple[int, str]]] = {}
    active = _active_go_terms(go_vocab_path)
    active_ids = {_go_int(go_id) for go_id in active}
    for go_id, info in active.items():
        child = _go_int(go_id)
        edges: List[Tuple[int, str]] = []
        for parent_raw in info.get("is_a", []) or []:
            parent = _go_int(parent_raw)
            if parent in active_ids:
                edges.append((parent, "is_a"))
        for parent_raw in info.get("part_of", []) or []:
            parent = _go_int(parent_raw)
            if parent in active_ids:
                edges.append((parent, "part_of"))
        parents[child] = edges
        for parent, relation in edges:
            children.setdefault(parent, []).append((child, relation))
    for go_id in active_ids:
        parents.setdefault(go_id, [])
        children.setdefault(go_id, [])
    return parents, children


def _namespace_map(go_vocab_path: Path) -> Dict[int, str]:
    return {
        _go_int(go_id): str(info.get("namespace", ""))
        for go_id, info in _active_go_terms(go_vocab_path).items()
    }


def _training_buckets(train_ids_path: Path, pid2pos_path: Path, branch_ids: Iterable[int], rare_lt: int):
    train_ids = {
        line.strip()
        for line in train_ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    pid2pos = _read_json(pid2pos_path)
    branch = set(int(x) for x in branch_ids)
    counts: Counter[int] = Counter()
    for pid in train_ids:
        counts.update(_go_int(g) for g in pid2pos.get(pid, []) if _go_int(g) in branch)
    seen = sorted(counts)
    rare = sorted(g for g, count in counts.items() if count < rare_lt)
    zero = sorted(branch - set(seen))
    return seen, rare, zero


def _validate(args, pf: dict, branch_ids: List[int]) -> None:
    required = {
        "train_ids_path": args.train_ids_path,
        "val_ids_path": args.val_ids_path,
        "pid2pos_path": args.pid2pos,
        "go_basic_json": args.go_basic_json,
        "go_cache_path": args.go_cache_path,
        "embed_dir_res": args.embed_dir_res,
        "go_text_path": pf.get("go_text_path"),
    }
    missing = [f"{name}={path}" for name, path in required.items() if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing PFresGO resources:\n  " + "\n  ".join(missing))
    if not branch_ids:
        raise RuntimeError("The selected branch GO id file is empty.")
    cache_ids = set(base.build_go_cache(str(args.go_cache_path)).id2row)
    absent = sorted(set(branch_ids) - cache_ids)
    if absent:
        raise RuntimeError(f"GO cache misses {len(absent)} branch terms; examples: {absent[:10]}")
    if bool(args.use_lora) or float(args.lr_lora or 0.0) != 0.0:
        raise ValueError("PFresGO config must keep LoRA disabled: use_lora=false and lr_lora=0.0")
    allow_eval_checkpoint = bool(pf.get("allow_checkpoint_for_eval", False)) and bool(args.eval_only)
    if (args.resume or args.warmstart_path) and not allow_eval_checkpoint:
        raise ValueError("PFresGO benchmark training must start clean: resume=null and warmstart_path=null")


def configure(config_path: str):
    cfg_path = Path(config_path).resolve()

    raw = yaml.safe_load(
        cfg_path.read_text(encoding="utf-8")
    ) or {}

    pf = raw.get("pfresgo", {})
    if not pf:
        raise ValueError(
            "Config is missing the top-level 'pfresgo' block"
        )

    # ---------------------------------------------------------
    # PFresGO segment configuration
    # ---------------------------------------------------------
    allowed_segments = [
        "name",
        "namespace",
        "definition",
        "is_a",
        "part_of",
    ]

    pf_enabled_segments = pf.get(
        "enabled_segments",
        ["name", "definition"],
    )

    if not isinstance(pf_enabled_segments, list):
        raise ValueError(
            "pfresgo.enabled_segments must be a list"
        )

    pf_enabled_segments = [
        str(segment).strip()
        for segment in pf_enabled_segments
        if str(segment).strip()
    ]

    if not pf_enabled_segments:
        raise ValueError(
            "At least one PFresGO segment must be enabled"
        )

    unknown_segments = (
        set(pf_enabled_segments) - set(allowed_segments)
    )
    if unknown_segments:
        raise ValueError(
            f"Unknown PFresGO segments: "
            f"{sorted(unknown_segments)}. "
            f"Allowed segments: {allowed_segments}"
        )

    # Remove duplicates while preserving canonical segment order.
    pf_enabled_segment_set = set(pf_enabled_segments)
    pf_enabled_segments = [
        segment
        for segment in allowed_segments
        if segment in pf_enabled_segment_set
    ]

    # ---------------------------------------------------------
    # Load main structured config
    # ---------------------------------------------------------
    args = base.load_structured_cfg(str(cfg_path))

    # Store explicitly so other code paths can access it.
    args.enabled_segments = list(pf_enabled_segments)
    args.pfresgo_enabled_segments = list(pf_enabled_segments)

    # ---------------------------------------------------------
    # Branch configuration
    # ---------------------------------------------------------
    args.pfresgo_branch = str(
        pf.get("branch", "")
    ).strip().upper()

    if args.pfresgo_branch not in {"BP", "MF", "CC"}:
        raise ValueError(
            "pfresgo.branch must be BP, MF, or CC"
        )

    # ---------------------------------------------------------
    # Benchmark protocol
    # ---------------------------------------------------------
    benchmark_protocol = str(
        pf.get("benchmark_protocol", "standard")
    ).strip().lower()

    if benchmark_protocol not in {"standard", "zeroshot"}:
        raise ValueError(
            "pfresgo.benchmark_protocol must be "
            "'standard' or 'zeroshot'"
        )

    args.pfresgo_benchmark_protocol = benchmark_protocol

    if "branch_go_ids_path" not in pf:
        raise ValueError(
            "pfresgo.branch_go_ids_path is required"
        )

    if "go_text_path" not in pf:
        raise ValueError(
            "pfresgo.go_text_path is required"
        )

    branch_ids_path = Path(
        pf["branch_go_ids_path"]
    ).expanduser().resolve()

    go_text_path = Path(
        pf["go_text_path"]
    ).expanduser().resolve()

    branch_ids = _read_ids(branch_ids_path)

    seen, rare, zero = _training_buckets(
        Path(args.train_ids_path),
        Path(args.pid2pos),
        branch_ids,
        int(pf.get("rare_lt", 20)),
    )

    # ---------------------------------------------------------
    # Evaluation split
    # ---------------------------------------------------------
    split = str(
        pf.get("evaluation_split", "valid")
    ).strip().lower()

    if split == "test":
        test_ids_path = pf.get("test_ids_path")
        if not test_ids_path:
            raise ValueError(
                "pfresgo.test_ids_path is required when "
                "pfresgo.evaluation_split=test"
            )

        args.val_ids_path = str(
            Path(test_ids_path).expanduser().resolve()
        )

    elif split != "valid":
        raise ValueError(
            "pfresgo.evaluation_split must be 'valid' or 'test'"
        )

    # ---------------------------------------------------------
    # Dynamic GO sets
    # ---------------------------------------------------------
    sentinels = {
        "__PFRESGO_BRANCH__": branch_ids,
        "__PFRESGO_SEEN__": seen,
        "__PFRESGO_RARE__": rare,
        "__PFRESGO_ZERO__": zero,
    }

    args.eval_space = "observed"
    args.go_path_observed = "__PFRESGO_BRANCH__"
    args.go_path_seen = "__PFRESGO_SEEN__"

    if benchmark_protocol == "zeroshot":
        # Special DeepGOZero-style experiment.
        # These sets may be used by downstream dataset code
        # to remove or mask selected training annotations.
        args.few_shot_path = "__PFRESGO_RARE__"
        args.zero_shot_path = "__PFRESGO_ZERO__"

        args.apply_few_zero_training_filter = True

    else:
        # Standard BP / MF / CC benchmark.
        # Do not remove proteins or annotations based on
        # rare/zero-shot buckets.
        args.few_shot_path = None
        args.zero_shot_path = None

        args.apply_few_zero_training_filter = False

    # ---------------------------------------------------------
    # Patch ID loaders
    # ---------------------------------------------------------
    original_pickle_loader = base.load_raw_pickle

    def load_ids_compat(path):
        if path is None:
            return []

        key = str(path)

        if key in sentinels:
            return list(sentinels[key])

        resolved_path = Path(path)

        if resolved_path.suffix.lower() in {
            ".json",
            ".txt",
            ".pkl",
            ".pickle",
        }:
            return _read_ids(resolved_path)

        return original_pickle_loader(path)

    base.load_raw_pickle = load_ids_compat

    def load_go_set_compat(path):
        if not path:
            return set()
        return set(load_ids_compat(path))

    base.load_go_set = load_go_set_compat

    # ---------------------------------------------------------
    # Patch PFresGO ontology loaders
    # ---------------------------------------------------------
    dag_cache = None
    namespace_cache = None

    def load_pfresgo_parents():
        nonlocal dag_cache
        if dag_cache is None:
            dag_cache = _build_dag(
                Path(args.go_basic_json)
            )
        return dag_cache[0]

    def load_pfresgo_children():
        nonlocal dag_cache
        if dag_cache is None:
            dag_cache = _build_dag(
                Path(args.go_basic_json)
            )
        return dag_cache[1]

    def load_pfresgo_namespaces():
        nonlocal namespace_cache
        if namespace_cache is None:
            namespace_cache = _namespace_map(
                Path(args.go_basic_json)
            )
        return namespace_cache

    base.load_go_parents = load_pfresgo_parents
    base.load_go_children = load_pfresgo_children
    base.load_go_namespaces = load_pfresgo_namespaces

    # ---------------------------------------------------------
    # Patch GO text loader
    # ---------------------------------------------------------
    original_text_loader = base.load_go_texts_by_phase

    def load_pfresgo_texts(
        folder,
        phase=0,
        return_segments=False,
        enabled_segments=None,
        **kwargs,
    ):
        """
        PFresGO-compatible GO text loader.

        Priority:
        1. Explicit enabled_segments supplied by the caller.
        2. pfresgo.enabled_segments from YAML.
        """

        effective_segments = (
            list(enabled_segments)
            if enabled_segments is not None
            else list(pf_enabled_segments)
        )

        unknown = (
            set(effective_segments) - set(allowed_segments)
        )
        if unknown:
            raise ValueError(
                f"Unknown GO segments passed to loader: "
                f"{sorted(unknown)}"
            )

        # Preserve canonical ordering.
        effective_set = set(effective_segments)
        effective_segments = [
            segment
            for segment in allowed_segments
            if segment in effective_set
        ]

        if not effective_segments:
            raise ValueError(
                "GO text loader received no enabled segments"
            )

        # PFresGO uses its canonical JSONL for negative phases.
        if int(phase) < 0:
            return load_go_texts_canonical(
                str(go_text_path),
                phase=phase,
                return_segments=return_segments,
                enabled_segments=effective_segments,
            )

        # Preserve the original loader for non-canonical phases.
        # Some older loader versions may not yet accept
        # enabled_segments, so support both signatures.
        try:
            return original_text_loader(
                folder,
                phase=phase,
                return_segments=return_segments,
                enabled_segments=effective_segments,
                **kwargs,
            )
        except TypeError as exc:
            if "enabled_segments" not in str(exc):
                raise

            return original_text_loader(
                folder,
                phase=phase,
                return_segments=return_segments,
                **kwargs,
            )

    base.load_go_texts_by_phase = load_pfresgo_texts

    # ---------------------------------------------------------
    # Validation and diagnostics
    # ---------------------------------------------------------
    _validate(args, pf, branch_ids)

    print(
        f"[PFresGO] "
        f"branch={args.pfresgo_branch} "
        f"split={split} "
        f"protocol={benchmark_protocol} "
        f"segments={pf_enabled_segments} "
        f"candidates={len(branch_ids)} "
        f"seen={len(seen)} "
        f"rare={len(rare)} "
        f"zero={len(zero)} "
        f"apply_few_zero_training_filter="
        f"{args.apply_few_zero_training_filter}"
    )

    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="PFresGO branch-specific training/evaluation")
    parser.add_argument("--config", required=True)
    cli = parser.parse_args()
    args = configure(cli.config)
    base.setup_logging(Path(args.output_dir), level=args.log_level)
    base.set_seed(args.seed)
    base.run_training(args)


if __name__ == "__main__":
    main()
