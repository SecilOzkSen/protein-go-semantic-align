# Created by Secil Sen — PKL (set/list/dict keys) -> JSON (sorted int list)
import os
import json
import pickle
from typing import Iterable, Set
from pathlib import Path

from src.configs.paths import (
    PROCESSED_PATH,                               # new: base for defaults
    ZERO_SHOT_TERMS_ID_ONLY_JSON,                # outputs (Path)
    FEW_SHOT_IC_TERMS_ID_ONLY_JSON,
    COMMON_IC_GO_TERMS_ID_ONLY_JSON,
)

# ---- Optional ENV overrides for input PKLs ----
# If not set, we fall back to PROCESSED_PATH / "go_terms" / default filenames
ZS_PKL_PATH = Path(os.getenv("ZERO_SHOT_TERMS_PKL_PATH", "")) \
    if os.getenv("ZERO_SHOT_TERMS_PKL_PATH") else (PROCESSED_PATH / "go_terms" / "zero_shot_terms.pkl")
FS_PKL_PATH = Path(os.getenv("FEW_SHOT_IC_TERMS_PKL_PATH", "")) \
    if os.getenv("FEW_SHOT_IC_TERMS_PKL_PATH") else (PROCESSED_PATH / "go_terms" / "ic_few_shot_terms.pkl")
CM_PKL_PATH = Path(os.getenv("COMMON_IC_GO_TERMS_PKL_PATH", "")) \
    if os.getenv("COMMON_IC_GO_TERMS_PKL_PATH") else (PROCESSED_PATH / "go_terms" / "ic_common_terms.pkl")


def _coerce_go_to_int(x) -> int:
    s = str(x).strip()
    if s.upper().startswith("GO:"):
        s = s[3:]
    return int(s)


def _load_set_pkl(path: Path) -> Set[int]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # allow: list[int], set[int], dict[go->...]
    if isinstance(data, dict):
        data = list(data.keys())
    return set(_coerce_go_to_int(x) for x in data)


def dump_json(path: Path, data: Iterable[int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(int(x) for x in data), f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {path}")


if __name__ == "__main__":
    print("[INFO] input PKLs:")
    print(f"  ZS: {ZS_PKL_PATH}")
    print(f"  FS: {FS_PKL_PATH}")
    print(f"  CM: {CM_PKL_PATH}")

    ZS = _load_set_pkl(ZS_PKL_PATH)
    FS = _load_set_pkl(FS_PKL_PATH)
    CM = _load_set_pkl(CM_PKL_PATH)

    # Cleaning (priority: ZS > FS > CM)
    orig_sizes = (len(ZS), len(FS), len(CM))
    FS -= ZS
    CM -= ZS
    CM -= FS

    print(f"[INFO] sizes (before): ZS={orig_sizes[0]}, FS={orig_sizes[1]}, CM={orig_sizes[2]}")
    print(f"[INFO] sizes (after ) : ZS={len(ZS)}, FS={len(FS)}, CM={len(CM)}")

    inter = (ZS & FS) | (ZS & CM) | (FS & CM)
    if inter:
        print(f"[WARN] non-empty intersections persisted (unexpected): {len(inter)} terms")

    dump_json(ZERO_SHOT_TERMS_ID_ONLY_JSON, ZS)
    dump_json(FEW_SHOT_IC_TERMS_ID_ONLY_JSON, FS)
    dump_json(COMMON_IC_GO_TERMS_ID_ONLY_JSON, CM)

    print("[DONE] Converted sets are ready for the pipeline.")
