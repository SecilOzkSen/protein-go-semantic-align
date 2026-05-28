import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm.auto import tqdm

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# -----------------------------------------------------------------------------
# IO / parsing helpers
# -----------------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def go_to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    if not s:
        return None
    if s.startswith("GO:"):
        s = s.split(":", 1)[1]
    if s.startswith("GO_"):
        s = s.split("_", 1)[1]
    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def normalize_namespace(ns: Any) -> Optional[str]:
    if ns is None:
        return None
    s = str(ns).strip().lower()
    if s in {"mf", "mfo", "molecular_function", "molecular function"}:
        return "MF"
    if s in {"bp", "bpo", "biological_process", "biological process"}:
        return "BP"
    if s in {"cc", "cco", "cellular_component", "cellular component"}:
        return "CC"
    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_terms_from_go_json(data: Any):
    """
    Yields (gid, term_dict). Robust to common go_vocab/go_basic JSON formats.
    """
    if isinstance(data, list):
        for t in data:
            if isinstance(t, dict):
                gid = go_to_int(t.get("id") or t.get("go_id") or t.get("GO") or t.get("go"))
                if gid is not None:
                    yield gid, t
        return

    if isinstance(data, dict):
        if isinstance(data.get("terms"), list):
            for t in data["terms"]:
                if isinstance(t, dict):
                    gid = go_to_int(t.get("id") or t.get("go_id") or t.get("GO") or t.get("go"))
                    if gid is not None:
                        yield gid, t
            return

        for k, v in data.items():
            gid = go_to_int(k)
            if isinstance(v, dict):
                gid2 = go_to_int(v.get("id") or v.get("go_id") or v.get("GO") or v.get("go"))
                if gid2 is not None:
                    gid = gid2
                if gid is not None:
                    yield gid, v
            else:
                # Rare format: {"GO:0008150": "biological_process"}
                if gid is not None:
                    yield gid, {"namespace": v}


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def extract_parent_ids(term: Dict[str, Any]) -> List[int]:
    """
    Robustly extracts is_a / part_of style parents from heterogeneous GO JSON.
    """
    vals: List[Any] = []

    direct_keys = [
        "parents",
        "parent_ids",
        "is_a",
        "is_a_parents",
        "part_of",
        "part_of_parents",
        "ancestors_direct",
    ]
    for k in direct_keys:
        vals.extend(_as_list(term.get(k)))

    rel = term.get("relations") or term.get("relationship") or term.get("relationships")
    if isinstance(rel, dict):
        for k in ["is_a", "part_of", "parents"]:
            vals.extend(_as_list(rel.get(k)))
    elif isinstance(rel, list):
        for r in rel:
            if isinstance(r, dict):
                rtype = str(r.get("type") or r.get("relation") or "").lower()
                if rtype in {"is_a", "part_of", "parent"}:
                    vals.extend(_as_list(r.get("id") or r.get("target") or r.get("go_id")))
            else:
                vals.append(r)

    out: List[int] = []
    for v in vals:
        if isinstance(v, dict):
            gid = go_to_int(v.get("id") or v.get("go_id") or v.get("target"))
        else:
            gid = go_to_int(v)
        if gid is not None:
            out.append(int(gid))

    # Stable unique order.
    seen = set()
    uniq = []
    for gid in out:
        if gid not in seen:
            seen.add(gid)
            uniq.append(gid)
    return uniq


def load_go_graph(go_basic_json: Path, eval_set: Set[int]) -> Tuple[Dict[int, str], Dict[int, List[int]], Dict[int, List[int]]]:
    data = load_json(go_basic_json)

    namespace: Dict[int, str] = {}
    parents_raw: Dict[int, List[int]] = defaultdict(list)

    for gid, term in iter_terms_from_go_json(data):
        if gid not in eval_set:
            continue
        ns = normalize_namespace(
            term.get("namespace")
            or term.get("aspect")
            or term.get("branch")
            or term.get("ontology")
        )
        if ns is not None:
            namespace[int(gid)] = ns
        parents_raw[int(gid)] = extract_parent_ids(term)

    # Keep only eval-space parents.
    parents: Dict[int, List[int]] = {}
    children: Dict[int, List[int]] = defaultdict(list)

    for gid in eval_set:
        ps = [int(p) for p in parents_raw.get(int(gid), []) if int(p) in eval_set]
        parents[int(gid)] = ps
        for p in ps:
            children[int(p)].append(int(gid))

    # Stable order.
    for k in list(children.keys()):
        children[k] = sorted(set(children[k]))
    for k in list(parents.keys()):
        parents[k] = sorted(set(parents[k]))

    return namespace, parents, dict(children)


def load_true_rows(dump_dir: Path) -> List[List[int]]:
    npy = dump_dir / "true_go_ids.npy"
    js = dump_dir / "true_go_ids.json"
    if npy.exists():
        arr = np.load(npy, mmap_mode="r")
        rows: List[List[int]] = []
        for row in arr:
            rows.append([int(x) for x in row if int(x) >= 0])
        return rows
    if js.exists():
        data = load_json(js)
        rows = []
        for row in data:
            xs = []
            for x in row:
                gid = go_to_int(x)
                if gid is not None and gid >= 0:
                    xs.append(int(gid))
            rows.append(xs)
        return rows
    raise FileNotFoundError(f"No true_go_ids.npy/json found in {dump_dir}")


def copy_if_exists(src_dir: Path, dst_dir: Path, name: str) -> None:
    src = src_dir / name
    if src.exists():
        shutil.copy2(src, dst_dir / name)


# -----------------------------------------------------------------------------
# Text-neighbor cache
# -----------------------------------------------------------------------------


def build_text_neighbor_cache(
    *,
    go_z_path: Path,
    needed_cols: np.ndarray,
    topn: int,
    device: str = "cuda:0",
    batch_size: int = 512,
    exclude_self: bool = True,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Computes nearest GO semantic neighbors for the requested eval-space columns.
    Uses go_z from the P3a dump, normalized dot product.
    """
    if topn <= 0:
        return {}
    if torch is None:
        raise RuntimeError("torch is required for text-neighbor expansion")

    needed = np.asarray(sorted(set(int(x) for x in needed_cols.tolist())), dtype=np.int64)
    if needed.size == 0:
        return {}

    logging.info("[text-neighbor] loading go_z: %s", str(go_z_path))
    go_z_np = np.load(go_z_path, mmap_mode="r")
    G = int(go_z_np.shape[0])

    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    logging.info("[text-neighbor] device=%s G=%d needed=%d topn=%d", str(dev), G, needed.size, topn)

    go = torch.as_tensor(np.asarray(go_z_np, dtype=np.float32), device=dev)
    go = torch.nn.functional.normalize(go, dim=-1)

    out: Dict[int, List[Tuple[int, float]]] = {}
    k_eff = min(G, int(topn) + (1 if exclude_self else 0) + 5)

    for s in tqdm(range(0, needed.size, batch_size), desc="text-neighbor cache"):
        e = min(needed.size, s + batch_size)
        cols_np = needed[s:e]
        q = go[torch.as_tensor(cols_np, dtype=torch.long, device=dev)]
        sim = q @ go.T  # [B,G]
        vals, idxs = torch.topk(sim, k=k_eff, dim=1)
        vals_np = vals.detach().cpu().numpy()
        idxs_np = idxs.detach().cpu().numpy()

        for i, col in enumerate(cols_np):
            pairs: List[Tuple[int, float]] = []
            for j, v in zip(idxs_np[i], vals_np[i]):
                jj = int(j)
                if exclude_self and jj == int(col):
                    continue
                pairs.append((jj, float(v)))
                if len(pairs) >= topn:
                    break
            out[int(col)] = pairs

    del go
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# -----------------------------------------------------------------------------
# Expansion logic
# -----------------------------------------------------------------------------


TYPE_DIRECT = 0
TYPE_PARENT = 1
TYPE_CHILD = 2
TYPE_SIBLING = 3
TYPE_TEXT = 4
TYPE_FILLER = -1


def add_candidate(
    cand: Dict[int, Dict[str, Any]],
    *,
    col: int,
    score: float,
    typ: int,
    seed_col: int,
    seed_score: float,
    seed_rank: int,
    distance: int,
    text_sim: float = 0.0,
):
    col = int(col)
    if col not in cand:
        cand[col] = {
            "score": float(score),
            "direct": 0,
            "parent": 0,
            "child": 0,
            "sibling": 0,
            "text": 0,
            "best_type": int(typ),
            "seed_col": int(seed_col),
            "seed_score": float(seed_score),
            "seed_rank": int(seed_rank),
            "distance": int(distance),
            "text_sim": float(text_sim),
        }
    else:
        # Keep the strongest score and associated seed metadata.
        if float(score) > cand[col]["score"]:
            cand[col]["score"] = float(score)
            cand[col]["best_type"] = int(typ)
            cand[col]["seed_col"] = int(seed_col)
            cand[col]["seed_score"] = float(seed_score)
            cand[col]["seed_rank"] = int(seed_rank)
            cand[col]["distance"] = int(distance)
            cand[col]["text_sim"] = float(text_sim)

    if typ == TYPE_DIRECT:
        cand[col]["direct"] = 1
    elif typ == TYPE_PARENT:
        cand[col]["parent"] = 1
    elif typ == TYPE_CHILD:
        cand[col]["child"] = 1
    elif typ == TYPE_SIBLING:
        cand[col]["sibling"] = 1
    elif typ == TYPE_TEXT:
        cand[col]["text"] = 1


def get_parents_k_hops(gid: int, parents: Dict[int, List[int]], hops: int) -> List[Tuple[int, int]]:
    if hops <= 0:
        return []
    out: List[Tuple[int, int]] = []
    frontier = [(int(gid), 0)]
    seen = {int(gid)}
    for _ in range(hops):
        new_frontier: List[Tuple[int, int]] = []
        for node, d in frontier:
            for p in parents.get(int(node), []):
                if p in seen:
                    continue
                seen.add(p)
                out.append((int(p), d + 1))
                new_frontier.append((int(p), d + 1))
        frontier = new_frontier
        if not frontier:
            break
    return out


def same_namespace(a: int, b: int, namespace: Dict[int, str], enabled: bool) -> bool:
    if not enabled:
        return True
    na = namespace.get(int(a))
    nb = namespace.get(int(b))
    if na is None or nb is None:
        return True
    return na == nb


def build_one_row(
    *,
    row_idx: int,
    base_cols: np.ndarray,
    base_scores: np.ndarray,
    eval_go_ids: np.ndarray,
    id_to_col: Dict[int, int],
    true_set: Set[int],
    namespace: Dict[int, str],
    parents: Dict[int, List[int]],
    children: Dict[int, List[int]],
    text_neighbors: Dict[int, List[Tuple[int, float]]],
    direct_k: int,
    seed_k: int,
    max_out: int,
    parent_hops: int,
    child_hops: int,
    sibling: bool,
    text_neighbor_k: int,
    same_ns_only: bool,
    parent_penalty: float,
    child_penalty: float,
    sibling_penalty: float,
    text_penalty: float,
    text_sim_weight: float,
    child_limit_per_seed: int,
    sibling_limit_per_seed: int,
    filler_score: float,
) -> Dict[str, np.ndarray]:
    cand: Dict[int, Dict[str, Any]] = {}

    direct_n = min(int(direct_k), base_cols.shape[0])
    seed_n = min(int(seed_k), direct_n)

    # Direct P3a candidates.
    for r in range(direct_n):
        col = int(base_cols[r])
        score = float(base_scores[r])
        add_candidate(
            cand,
            col=col,
            score=score,
            typ=TYPE_DIRECT,
            seed_col=col,
            seed_score=score,
            seed_rank=r,
            distance=0,
        )

    # Semantic expansions from top seed candidates.
    for r in range(seed_n):
        seed_col = int(base_cols[r])
        seed_gid = int(eval_go_ids[seed_col])
        seed_score = float(base_scores[r])

        # Parents, up to parent_hops.
        if parent_hops > 0:
            for pgid, dist in get_parents_k_hops(seed_gid, parents, parent_hops):
                if not same_namespace(seed_gid, pgid, namespace, same_ns_only):
                    continue
                pcol = id_to_col.get(int(pgid))
                if pcol is None:
                    continue
                score = seed_score - parent_penalty * float(max(1, dist))
                add_candidate(
                    cand,
                    col=pcol,
                    score=score,
                    typ=TYPE_PARENT,
                    seed_col=seed_col,
                    seed_score=seed_score,
                    seed_rank=r,
                    distance=int(dist),
                )

        # Children, breadth limited.
        if child_hops > 0:
            frontier = [(seed_gid, 0)]
            seen_nodes = {seed_gid}
            added = 0
            for _ in range(child_hops):
                new_frontier: List[Tuple[int, int]] = []
                for node, dist0 in frontier:
                    child_list = children.get(int(node), [])
                    for cgid in child_list:
                        if cgid in seen_nodes:
                            continue
                        seen_nodes.add(cgid)
                        if not same_namespace(seed_gid, cgid, namespace, same_ns_only):
                            continue
                        ccol = id_to_col.get(int(cgid))
                        if ccol is None:
                            continue
                        dist = dist0 + 1
                        score = seed_score - child_penalty * float(max(1, dist))
                        add_candidate(
                            cand,
                            col=ccol,
                            score=score,
                            typ=TYPE_CHILD,
                            seed_col=seed_col,
                            seed_score=seed_score,
                            seed_rank=r,
                            distance=int(dist),
                        )
                        added += 1
                        if child_limit_per_seed > 0 and added >= child_limit_per_seed:
                            break
                        new_frontier.append((int(cgid), dist))
                    if child_limit_per_seed > 0 and added >= child_limit_per_seed:
                        break
                if child_limit_per_seed > 0 and added >= child_limit_per_seed:
                    break
                frontier = new_frontier
                if not frontier:
                    break

        # Same-parent siblings.
        if sibling and sibling_limit_per_seed != 0:
            added = 0
            sibs: Set[int] = set()
            for pgid in parents.get(seed_gid, []):
                for sgid in children.get(int(pgid), []):
                    if int(sgid) == int(seed_gid):
                        continue
                    sibs.add(int(sgid))
            for sgid in sorted(sibs):
                if not same_namespace(seed_gid, sgid, namespace, same_ns_only):
                    continue
                scol = id_to_col.get(int(sgid))
                if scol is None:
                    continue
                score = seed_score - sibling_penalty
                add_candidate(
                    cand,
                    col=scol,
                    score=score,
                    typ=TYPE_SIBLING,
                    seed_col=seed_col,
                    seed_score=seed_score,
                    seed_rank=r,
                    distance=1,
                )
                added += 1
                if sibling_limit_per_seed > 0 and added >= sibling_limit_per_seed:
                    break

        # Text/embedding neighbors in GO semantic space.
        if text_neighbor_k > 0:
            pairs = text_neighbors.get(seed_col, [])[:text_neighbor_k]
            for ncol, sim in pairs:
                ngid = int(eval_go_ids[int(ncol)])
                if not same_namespace(seed_gid, ngid, namespace, same_ns_only):
                    continue
                score = seed_score - text_penalty + text_sim_weight * float(sim)
                add_candidate(
                    cand,
                    col=int(ncol),
                    score=score,
                    typ=TYPE_TEXT,
                    seed_col=seed_col,
                    seed_score=seed_score,
                    seed_rank=r,
                    distance=1,
                    text_sim=float(sim),
                )

    # Rank by score descending, direct candidates get tiny tie priority.
    items = list(cand.items())
    items.sort(key=lambda kv: (float(kv[1]["score"]), int(kv[1]["direct"])), reverse=True)
    items = items[:max_out]

    n = len(items)
    out_cols = np.full(max_out, -1, dtype=np.int32)
    out_scores = np.full(max_out, filler_score, dtype=np.float32)
    out_labels = np.zeros(max_out, dtype=np.int8)
    out_valid = np.zeros(max_out, dtype=np.int8)

    direct_flag = np.zeros(max_out, dtype=np.int8)
    parent_flag = np.zeros(max_out, dtype=np.int8)
    child_flag = np.zeros(max_out, dtype=np.int8)
    sibling_flag = np.zeros(max_out, dtype=np.int8)
    text_flag = np.zeros(max_out, dtype=np.int8)
    best_type = np.full(max_out, TYPE_FILLER, dtype=np.int8)
    seed_col_arr = np.full(max_out, -1, dtype=np.int32)
    seed_score_arr = np.zeros(max_out, dtype=np.float32)
    seed_rank_arr = np.full(max_out, -1, dtype=np.int32)
    rel_dist_arr = np.full(max_out, -1, dtype=np.int16)
    text_sim_arr = np.zeros(max_out, dtype=np.float32)

    for i, (col, meta) in enumerate(items):
        gid = int(eval_go_ids[int(col)])
        out_cols[i] = int(col)
        out_scores[i] = float(meta["score"])
        out_labels[i] = 1 if gid in true_set else 0
        out_valid[i] = 1
        direct_flag[i] = int(meta["direct"])
        parent_flag[i] = int(meta["parent"])
        child_flag[i] = int(meta["child"])
        sibling_flag[i] = int(meta["sibling"])
        text_flag[i] = int(meta["text"])
        best_type[i] = int(meta["best_type"])
        seed_col_arr[i] = int(meta["seed_col"])
        seed_score_arr[i] = float(meta["seed_score"])
        seed_rank_arr[i] = int(meta["seed_rank"])
        rel_dist_arr[i] = int(meta["distance"])
        text_sim_arr[i] = float(meta["text_sim"])

    # Fill remaining slots with deterministic direct candidates if possible, otherwise col 0.
    # They are invalid, so they do not affect loss/eval if valid_mask is used.
    if n < max_out:
        fill_col = int(base_cols[0]) if base_cols.shape[0] > 0 else 0
        out_cols[n:] = fill_col

    return {
        "top_go_cols": out_cols,
        "top_scores": out_scores,
        "top_labels": out_labels,
        "top_valid": out_valid,
        "direct_p3a": direct_flag,
        "parent_expansion": parent_flag,
        "child_expansion": child_flag,
        "sibling_expansion": sibling_flag,
        "text_neighbor_expansion": text_flag,
        "expansion_type": best_type,
        "seed_col": seed_col_arr,
        "seed_score": seed_score_arr,
        "seed_rank": seed_rank_arr,
        "relation_distance": rel_dist_arr,
        "text_neighbor_sim": text_sim_arr,
    }


# -----------------------------------------------------------------------------
# Main dumping
# -----------------------------------------------------------------------------


def pad_true_ids(true_rows: List[List[int]], pad_value: int = -1) -> np.ndarray:
    max_len = max((len(x) for x in true_rows), default=0)
    arr = np.full((len(true_rows), max_len), pad_value, dtype=np.int64)
    for i, xs in enumerate(true_rows):
        if xs:
            arr[i, : len(xs)] = np.asarray(xs, dtype=np.int64)
    return arr


def main():
    p = argparse.ArgumentParser("Build P3a semantic expansion candidate dump.")

    p.add_argument("--p3a_dump", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--go_basic_json", type=str, default="/workspace/data/go_vocab.json")

    p.add_argument("--direct_k", type=int, default=500, help="Keep this many original P3a candidates.")
    p.add_argument("--seed_k", type=int, default=100, help="Expand only from top seed_k P3a candidates.")
    p.add_argument("--max_out", type=int, default=1000)

    p.add_argument("--parent_hops", type=int, default=2)
    p.add_argument("--child_hops", type=int, default=1)
    p.add_argument("--child_limit_per_seed", type=int, default=25)
    p.add_argument("--use_siblings", action="store_true")
    p.add_argument("--sibling_limit_per_seed", type=int, default=25)

    p.add_argument("--text_neighbor_k", type=int, default=0)
    p.add_argument("--text_neighbor_seed_k", type=int, default=None, help="Optional smaller seed_k for text neighbors.")
    p.add_argument("--text_neighbor_device", type=str, default="cuda:0")
    p.add_argument("--text_neighbor_batch_size", type=int, default=512)

    p.add_argument("--same_namespace_only", action="store_true")

    p.add_argument("--parent_penalty", type=float, default=0.10)
    p.add_argument("--child_penalty", type=float, default=0.15)
    p.add_argument("--sibling_penalty", type=float, default=0.20)
    p.add_argument("--text_penalty", type=float, default=0.15)
    p.add_argument("--text_sim_weight", type=float, default=0.05)
    p.add_argument("--filler_score", type=float, default=-1e6)

    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    setup_logging()

    p3a_dump = Path(args.p3a_dump)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {out_dir}. Pass --overwrite to replace.")
        logging.warning("[semexp] removing existing output dir: %s", str(out_dir))
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_go_ids = np.load(p3a_dump / "eval_go_ids.npy", mmap_mode="r").astype(np.int64)
    eval_set = set(int(x) for x in eval_go_ids.tolist())
    id_to_col = {int(g): i for i, g in enumerate(eval_go_ids.tolist())}

    top_cols = np.load(p3a_dump / "top_go_cols.int32.npy", mmap_mode="r")
    top_scores = np.load(p3a_dump / "top_scores.float32.npy", mmap_mode="r")
    true_rows = load_true_rows(p3a_dump)

    n_samples = int(top_cols.shape[0])
    topk_avail = int(top_cols.shape[1])
    direct_k = min(int(args.direct_k), topk_avail)
    seed_k = min(int(args.seed_k), direct_k)
    max_out = int(args.max_out)

    if len(true_rows) != n_samples:
        raise RuntimeError(f"true rows {len(true_rows)} != top_cols rows {n_samples}")

    logging.info("[semexp] p3a_dump=%s", str(p3a_dump))
    logging.info("[semexp] out_dir=%s", str(out_dir))
    logging.info("[semexp] n_samples=%d eval_go=%d direct_k=%d seed_k=%d max_out=%d", n_samples, len(eval_go_ids), direct_k, seed_k, max_out)

    namespace, parents, children = load_go_graph(Path(args.go_basic_json), eval_set)
    logging.info("[semexp] namespace=%d parents=%d children=%d", len(namespace), len(parents), len(children))

    # Text-neighbor expansion seed collection.
    text_neighbors: Dict[int, List[Tuple[int, float]]] = {}
    if int(args.text_neighbor_k) > 0:
        tn_seed_k = int(args.text_neighbor_seed_k or seed_k)
        tn_seed_k = min(tn_seed_k, seed_k)
        needed_cols = np.unique(np.asarray(top_cols[:, :tn_seed_k], dtype=np.int64).reshape(-1))
        text_neighbors = build_text_neighbor_cache(
            go_z_path=p3a_dump / "go_z.float16.npy",
            needed_cols=needed_cols,
            topn=int(args.text_neighbor_k),
            device=str(args.text_neighbor_device),
            batch_size=int(args.text_neighbor_batch_size),
        )
        logging.info("[semexp] text-neighbor cache entries=%d", len(text_neighbors))

    # Copy shared bank files.
    for name in [
        "eval_go_ids.npy",
        "go_z.float16.npy",
        "protein_z.float16.npy",
        "protein_ids.json",
        "true_go_ids.json",
        "true_go_ids.npy",
    ]:
        copy_if_exists(p3a_dump, out_dir, name)

    # Ensure true_go_ids.npy exists.
    if not (out_dir / "true_go_ids.npy").exists():
        np.save(out_dir / "true_go_ids.npy", pad_true_ids(true_rows))

    # Output memmaps.
    mm = {
        "top_go_cols": np.lib.format.open_memmap(out_dir / "top_go_cols.int32.npy", mode="w+", dtype=np.int32, shape=(n_samples, max_out)),
        "top_scores": np.lib.format.open_memmap(out_dir / "top_scores.float32.npy", mode="w+", dtype=np.float32, shape=(n_samples, max_out)),
        "top_labels": np.lib.format.open_memmap(out_dir / "top_labels.int8.npy", mode="w+", dtype=np.int8, shape=(n_samples, max_out)),
        "top_valid": np.lib.format.open_memmap(out_dir / "top_valid.int8.npy", mode="w+", dtype=np.int8, shape=(n_samples, max_out)),
        "direct_p3a": np.lib.format.open_memmap(out_dir / "direct_p3a.int8.npy", mode="w+", dtype=np.int8, shape=(n_samples, max_out)),
        "parent_expansion": np.lib.format.open_memmap(out_dir / "parent_expansion.int8.npy", mode="w+", dtype=np.int8, shape=(n_samples, max_out)),
        "child_expansion": np.lib.format.open_memmap(out_dir / "child_expansion.int8.npy", mode="w+", dtype=np.int8, shape=(n_samples, max_out)),
        "sibling_expansion": np.lib.format.open_memmap(out_dir / "sibling_expansion.int8.npy", mode="w+", dtype=np.int8, shape=(n_samples, max_out)),
        "text_neighbor_expansion": np.lib.format.open_memmap(out_dir / "text_neighbor_expansion.int8.npy", mode="w+", dtype=np.int8, shape=(n_samples, max_out)),
        "expansion_type": np.lib.format.open_memmap(out_dir / "expansion_type.int8.npy", mode="w+", dtype=np.int8, shape=(n_samples, max_out)),
        "seed_col": np.lib.format.open_memmap(out_dir / "seed_col.int32.npy", mode="w+", dtype=np.int32, shape=(n_samples, max_out)),
        "seed_score": np.lib.format.open_memmap(out_dir / "seed_score.float32.npy", mode="w+", dtype=np.float32, shape=(n_samples, max_out)),
        "seed_rank": np.lib.format.open_memmap(out_dir / "seed_rank.int32.npy", mode="w+", dtype=np.int32, shape=(n_samples, max_out)),
        "relation_distance": np.lib.format.open_memmap(out_dir / "relation_distance.int16.npy", mode="w+", dtype=np.int16, shape=(n_samples, max_out)),
        "text_neighbor_sim": np.lib.format.open_memmap(out_dir / "text_neighbor_sim.float32.npy", mode="w+", dtype=np.float32, shape=(n_samples, max_out)),
    }

    total_valid = 0
    total_direct = 0
    total_parent = 0
    total_child = 0
    total_sibling = 0
    total_text = 0

    for i in tqdm(range(n_samples), desc="build P3a semantic expansion"):
        true_set = set(int(g) for g in true_rows[i])
        row = build_one_row(
            row_idx=i,
            base_cols=np.asarray(top_cols[i], dtype=np.int64),
            base_scores=np.asarray(top_scores[i], dtype=np.float32),
            eval_go_ids=eval_go_ids,
            id_to_col=id_to_col,
            true_set=true_set,
            namespace=namespace,
            parents=parents,
            children=children,
            text_neighbors=text_neighbors,
            direct_k=direct_k,
            seed_k=seed_k,
            max_out=max_out,
            parent_hops=int(args.parent_hops),
            child_hops=int(args.child_hops),
            sibling=bool(args.use_siblings),
            text_neighbor_k=int(args.text_neighbor_k),
            same_ns_only=bool(args.same_namespace_only),
            parent_penalty=float(args.parent_penalty),
            child_penalty=float(args.child_penalty),
            sibling_penalty=float(args.sibling_penalty),
            text_penalty=float(args.text_penalty),
            text_sim_weight=float(args.text_sim_weight),
            child_limit_per_seed=int(args.child_limit_per_seed),
            sibling_limit_per_seed=int(args.sibling_limit_per_seed),
            filler_score=float(args.filler_score),
        )
        for key, arr in row.items():
            mm[key][i, :] = arr

        total_valid += int(row["top_valid"].sum())
        total_direct += int((row["direct_p3a"] * row["top_valid"]).sum())
        total_parent += int((row["parent_expansion"] * row["top_valid"]).sum())
        total_child += int((row["child_expansion"] * row["top_valid"]).sum())
        total_sibling += int((row["sibling_expansion"] * row["top_valid"]).sum())
        total_text += int((row["text_neighbor_expansion"] * row["top_valid"]).sum())

    # Flush memmaps.
    for x in mm.values():
        x.flush()

    metadata = {
        "source": "p3a_semantic_expansion",
        "p3a_dump": str(p3a_dump),
        "go_basic_json": str(args.go_basic_json),
        "n_samples": int(n_samples),
        "n_go": int(len(eval_go_ids)),
        "direct_k": int(direct_k),
        "seed_k": int(seed_k),
        "max_out": int(max_out),
        "parent_hops": int(args.parent_hops),
        "child_hops": int(args.child_hops),
        "child_limit_per_seed": int(args.child_limit_per_seed),
        "use_siblings": bool(args.use_siblings),
        "sibling_limit_per_seed": int(args.sibling_limit_per_seed),
        "text_neighbor_k": int(args.text_neighbor_k),
        "same_namespace_only": bool(args.same_namespace_only),
        "penalties": {
            "parent_penalty": float(args.parent_penalty),
            "child_penalty": float(args.child_penalty),
            "sibling_penalty": float(args.sibling_penalty),
            "text_penalty": float(args.text_penalty),
            "text_sim_weight": float(args.text_sim_weight),
        },
        "valid_candidate_counts": {
            "total_valid": int(total_valid),
            "direct_flags": int(total_direct),
            "parent_flags": int(total_parent),
            "child_flags": int(total_child),
            "sibling_flags": int(total_sibling),
            "text_neighbor_flags": int(total_text),
        },
        "files": {
            "top_go_cols": "top_go_cols.int32.npy",
            "top_scores": "top_scores.float32.npy",
            "top_labels": "top_labels.int8.npy",
            "top_valid": "top_valid.int8.npy",
            "direct_p3a": "direct_p3a.int8.npy",
            "parent_expansion": "parent_expansion.int8.npy",
            "child_expansion": "child_expansion.int8.npy",
            "sibling_expansion": "sibling_expansion.int8.npy",
            "text_neighbor_expansion": "text_neighbor_expansion.int8.npy",
            "expansion_type": "expansion_type.int8.npy",
            "seed_col": "seed_col.int32.npy",
            "seed_score": "seed_score.float32.npy",
            "seed_rank": "seed_rank.int32.npy",
            "relation_distance": "relation_distance.int16.npy",
            "text_neighbor_sim": "text_neighbor_sim.float32.npy",
        },
        "schema": {
            "direct_p3a": "Candidate appears in original P3a direct top-K.",
            "parent_expansion": "Candidate was added as a GO parent/ancestor of a P3a seed.",
            "child_expansion": "Candidate was added as a GO child/descendant of a P3a seed.",
            "sibling_expansion": "Candidate was added as a same-parent sibling of a P3a seed.",
            "text_neighbor_expansion": "Candidate was added as a nearest GO semantic neighbor in go_z space.",
            "seed_col": "Eval-space column of the P3a seed that generated the selected score.",
            "seed_rank": "Rank index of the generating seed in the original P3a dump.",
        },
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    (out_dir / "DONE").write_text("done\n", encoding="utf-8")
    logging.info("[semexp] done: %s", str(out_dir))
    logging.info("[semexp] total_valid=%d direct=%d parent=%d child=%d sibling=%d text=%d", total_valid, total_direct, total_parent, total_child, total_sibling, total_text)


if __name__ == "__main__":
    main()
