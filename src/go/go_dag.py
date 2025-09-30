
"""
Minimal GO DAG helpers (functional, side-effect free).

Exports:
- ancestors_with_hops(term, parents, max_hops, allowed_rels, include_self)
- descendants_with_hops(term, children, max_hops, allowed_rels, include_self)
- siblings(term, parents, children)
- hop_limited_exclusion(positives, parents, children, max_up, max_down, allow_siblings)
- mask_negatives(neg_candidates, pos_set, parents, children, ...)
- expand_with_ancestors(pos_terms, parents, zs_blocklist, ...)

Notes:
- parents/children are adjacency maps: dict[int, list[tuple[int, str]]]
- allowed_rels is an optional set[str] of relation names to follow.
"""
from __future__ import annotations
from collections import deque
from functools import lru_cache
import json
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple, List, Any
from src.configs.paths import GO_PARENTS, GO_CHILDREN

GO = int
Rel = str
ParentsMap = Mapping[GO, Iterable[Tuple[GO, str]]]
ChildrenMap = Mapping[GO, Iterable[Tuple[GO, Rel]]]

# Global Record
_PARENTS: Optional[ParentsMap] = None
_PARENTS_ID: Optional[int] = None

# ---------------- Parent Installation --------
def _install_parents_once(parents: ParentsMap) -> None:
    """Yeni bir parents objesi geldiyse cache'i temizle ve kaydet."""
    global _PARENTS, _PARENTS_ID
    if _PARENTS_ID is not id(parents):
        _PARENTS = parents
        _PARENTS_ID = id(parents)
        _ancestors_with_hops_cached.cache_clear()

def _norm_rels(allowed_rels: Optional[Set[str]]) -> Tuple[str, ...]:
    return tuple(sorted(allowed_rels)) if allowed_rels else ("is_a", "part_of")


# ---------------- Map Loaders ----------------
def load_go_parents() -> Dict[Any]:
    return _load_dag_jsons(GO_PARENTS)

def load_go_children() -> Dict[Any]:
    return _load_dag_jsons(GO_CHILDREN)


def _load_dag_jsons(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw_dag = json.load(f)
    converted: Mapping[int, Sequence[Tuple[int, str]]] = {
        int(go_id.split(":")[1]): [
            (int(child[0].split(":")[1]), child[1]) for child in children
        ]
        for go_id, children in raw_dag.items()
    }
    return converted

# ---------------- BFS hop distances ----------------

@lru_cache(maxsize=262_144)
def _ancestors_with_hops_cached(
    term: GO,
    max_hops: Optional[int],
    allowed_rels_tuple: Tuple[str, ...],
    include_self: bool,
) -> Dict[GO, int]:
    if _PARENTS is None:
        raise RuntimeError("Parents map is not installed")
    return _bfs_ancestors(term, _PARENTS, max_hops, allowed_rels_tuple, include_self)

def ancestors_with_hops(
    term: GO,
    parents: ParentsMap,
    max_hops: Optional[int] = None,
    allowed_rels: Optional[Set[str]] = None,
    include_self: bool = False,
) -> Dict[GO, int]:
    _install_parents_once(parents)
    return _ancestors_with_hops_cached(term, max_hops, _norm_rels(allowed_rels), include_self)


def _bfs_ancestors(
    term: GO,
    parents: ParentsMap,
    max_hops: Optional[int],
    allowed_rels_tuple: Tuple[str, ...],
    include_self: bool,
) -> Dict[GO, int]:
    out: Dict[GO, int] = {}
    seen = {term}
    queue: List[Tuple[GO, int]] = [(term, 0)]
    while queue:
        node, d = queue.pop(0)
        if d > 0 or include_self:
            if d == 0 and include_self:
                out[node] = 0
            elif d > 0:
                out[node] = d
        if max_hops is not None and d >= max_hops:
            continue
        for p, rel in parents.get(node, ()):
            if rel in allowed_rels_tuple and p not in seen:
                seen.add(p)
                queue.append((p, d + 1))
    return out


@lru_cache(maxsize=262_144)
def descendants_with_hops(
    term: GO,
    children: ChildrenMap,
    max_hops: Optional[int] = None,
    allowed_rels: Optional[Set[str]] = None,
    include_self: bool = False,
) -> Dict[GO, int]:
    dist: Dict[GO, int] = {}
    q: deque[Tuple[GO, int]] = deque([(term, 0)])
    seen: Set[GO] = {term}
    while q:
        node, d = q.popleft()
        if d > 0:
            if node not in dist or d < dist[node]:
                dist[node] = d
        if max_hops is not None and d == max_hops:
            continue
        for c, rel in children.get(node, ()):
            if allowed_rels and rel not in allowed_rels:
                continue
            if c not in seen:
                seen.add(c)
                q.append((c, d + 1))
    if include_self:
        dist[term] = 0
    return dist


def siblings(term: GO, parents: ParentsMap, children: ChildrenMap) -> Set[GO]:
    sibs: Set[GO] = set()
    for p, rel in parents.get(term, ()):
        for c, _rel2 in children.get(p, ()):
            if c != term:
                sibs.add(c)
    return sibs


# ---------------- Masks & policies ----------------

def hop_limited_exclusion(
    positives: Iterable[GO],
    parents: ParentsMap,
    children: ChildrenMap,
    max_up: int = 0,
    max_down: int = 0,
    allow_siblings: bool = False,
    allowed_rels: Optional[Set[str]] = None,
    forbid_self: bool = True,
) -> Set[GO]:
    """Terms to EXCLUDE around positives based on hop radii and sibling rule."""
    pset: Set[GO] = set(int(x) for x in positives)
    excl: Set[GO] = set()
    for p in pset:
        if forbid_self:
            excl.add(p)
        if max_up > 0:
            excl |= set(ancestors_with_hops(p, parents, max_hops=max_up, allowed_rels=allowed_rels).keys())
        if max_down > 0:
            excl |= set(descendants_with_hops(p, children, max_hops=max_down, allowed_rels=allowed_rels).keys())
        if not allow_siblings:
            excl |= siblings(p, parents, children)
    return excl


def mask_negatives(
    neg_candidates: Sequence[GO],
    pos_set: Iterable[GO],
    parents: ParentsMap,
    children: ChildrenMap,
    *,
    forbid_ancestors: bool = True,
    forbid_descendants: bool = True,
    forbid_self: bool = True,
    max_hops_up: Optional[int] = None,
    max_hops_down: Optional[int] = None,
    allowed_rels: Optional[Set[str]] = None,
) -> List[GO]:
    pset: Set[GO] = set(int(x) for x in pos_set)
    banned: Set[GO] = set()
    if forbid_self:
        banned |= pset
    if forbid_ancestors:
        a = set()
        for t in pset:
            a |= set(ancestors_with_hops(t, parents, max_hops=max_hops_up, allowed_rels=allowed_rels).keys())
        banned |= a
    if forbid_descendants:
        d = set()
        for t in pset:
            d |= set(descendants_with_hops(t, children, max_hops=max_hops_down, allowed_rels=allowed_rels).keys())
        banned |= d
    return [g for g in neg_candidates if int(g) not in banned]


def expand_with_ancestors(
    pos_terms: Sequence[GO],
    parents: Mapping[int, Sequence[Tuple[int, str]]],
    zs_blocklist: Set[GO],
    *,
    min_pos: int = 3,
    max_add: int = 4,
    max_hops: Optional[int] = 3,
    stoplist: Optional[Set[GO]] = None,
    gamma: float = 0.7,
    allowed_rels: Optional[Set[str]] = None,
) -> Tuple[List[GO], List[float], Dict[GO, bool]]:
    """
    Ensure a minimum number of positives by adding close ancestors as soft positives.
    Returns (expanded_terms, weights, is_generalized)
    """
    base: List[GO] = [int(x) for x in pos_terms if int(x) not in zs_blocklist]
    base_set: Set[GO] = set(base)
    if len(base_set) >= min_pos:
        return list(base_set), [1.0] * len(base_set), {t: False for t in base_set}

    cand = {}
    for t in base_set:
        ancestors = ancestors_with_hops(term=t, parents=parents, max_hops=max_hops, allowed_rels=allowed_rels)
        cand.update(ancestors)
    if zs_blocklist:
        cand = {g: h for g, h in cand.items() if g not in zs_blocklist}
    if stoplist:
        cand = {g: h for g, h in cand.items() if g not in stoplist}
    add_sorted = sorted(cand.items(), key=lambda kv: (kv[1], kv[0]))[:max_add]
    added = [g for g, _ in add_sorted]

    expanded: List[GO] = list(base_set) + added
    is_generalized = {t: (t in added) for t in expanded}
    weights: List[float] = []
    for t in expanded:
        if t in base_set:
            weights.append(1.0)
        else:
            hop = max(1, cand.get(t, 1))
            w = max(0.1, gamma ** (hop - 1))
            weights.append(float(w))
    return expanded, weights, is_generalized
