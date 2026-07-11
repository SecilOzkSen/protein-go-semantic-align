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
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    pf = raw.get("pfresgo", {})
    if not pf:
        raise ValueError("Config is missing the top-level 'pfresgo' block")

    args = base.load_structured_cfg(str(cfg_path))
    args.pfresgo_branch = str(pf.get("branch", "")).upper()
    if args.pfresgo_branch not in {"BP", "MF", "CC"}:
        raise ValueError("pfresgo.branch must be BP, MF, or CC")

    branch_ids_path = Path(pf["branch_go_ids_path"])
    go_text_path = Path(pf["go_text_path"])
    branch_ids = _read_ids(branch_ids_path)
    seen, rare, zero = _training_buckets(
        Path(args.train_ids_path), Path(args.pid2pos), branch_ids, int(pf.get("rare_lt", 20))
    )

    split = str(pf.get("evaluation_split", "valid")).lower()
    if split == "test":
        args.val_ids_path = Path(pf["test_ids_path"])
    elif split != "valid":
        raise ValueError("pfresgo.evaluation_split must be 'valid' or 'test'")

    sentinels = {
        "__PFRESGO_BRANCH__": branch_ids,
        "__PFRESGO_SEEN__": seen,
        "__PFRESGO_RARE__": rare,
        "__PFRESGO_ZERO__": zero,
    }
    args.eval_space = "observed"
    args.go_path_observed = "__PFRESGO_BRANCH__"
    args.go_path_seen = "__PFRESGO_SEEN__"
    args.few_shot_path = "__PFRESGO_RARE__"
    args.zero_shot_path = "__PFRESGO_ZERO__"

    original_pickle_loader = base.load_raw_pickle

    def load_ids_compat(path):
        key = str(path)
        if key in sentinels:
            return list(sentinels[key])
        p = Path(path)
        return _read_ids(p) if p.suffix.lower() in {".json", ".txt"} else original_pickle_loader(path)

    base.load_raw_pickle = load_ids_compat
    base.load_go_set = lambda path: set(load_ids_compat(path)) if path else set()
    base.load_go_parents = lambda: _build_dag(Path(args.go_basic_json))[0]
    base.load_go_children = lambda: _build_dag(Path(args.go_basic_json))[1]
    base.load_go_namespaces = lambda: _namespace_map(Path(args.go_basic_json))

    original_text_loader = base.load_go_texts_by_phase

    def load_pfresgo_texts(_folder, phase=0, return_segments=False):
        if int(phase) < 0:
            return load_go_texts_canonical(str(go_text_path), phase=phase, return_segments=return_segments)
        return original_text_loader(_folder, phase=phase, return_segments=return_segments)

    base.load_go_texts_by_phase = load_pfresgo_texts
    _validate(args, pf, branch_ids)
    print(
        f"[PFresGO] branch={args.pfresgo_branch} split={split} "
        f"candidates={len(branch_ids)} seen={len(seen)} rare={len(rare)} zero={len(zero)}"
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
