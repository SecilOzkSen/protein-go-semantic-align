
"""
make_go_parents_children.py
---------------------------
Build GO DAG adjacency JSONs (parents/children) from either:
  (A) go_basic_obo_terms_v2.pkl  (recommended if you already produced it), or
  (B) go-basic.obo               (direct OBO parse; supports is_a + relationship: part_of/regulates)

Output schema (compatible with load_parents_map_any in your go_dag.py):
    parents.json  : { child_int : [[parent_int, "rel"], ...], ... }
    children.json : { parent_int: [[child_int,  "rel"], ...], ... }

Usage examples
--------------
# From your existing PKL
python make_go_parents_children.py \
  --input go_basic_obo_terms_v2.pkl \
  --out-dir data \
  --allowed-rels is_a,part_of,regulates,positively_regulates,negatively_regulates

# From go-basic.obo
python make_go_parents_children.py \
  --input go-basic.obo \
  --out-dir data \
  --allowed-rels is_a,part_of,regulates,positively_regulates,negatively_regulates

# Keep only is_a edges and save under custom names with string keys
python make_go_parents_children.py \
  --input go-basic.obo \
  --out-dir data \
  --parents-json go_is_a_parents.json --children-json go_is_a_children.json \
  --allowed-rels is_a --string-keys
"""
import argparse, json, pickle, re, os
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from src.configs.paths import GO_TERMS_PKL,GO_CHILDREN, GO_PARENTS
from src.configs.parameters import ALLOWED_RELS_FOR_DAG

PARENT_REL = "is_a"
GO_RE = re.compile(r'GO:(\d{7})')

def _gid_str(x) -> str:
    s = str(x).strip()
    if s.startswith("GO:"):
        return s
    return f"GO:{int(s):07d}"

def _gid_int(s: str) -> int:
    m = GO_RE.search(s.strip())
    if not m:
        raise ValueError(f"Not a GO id: {s}")
    return int(m.group(1))


def build_from_pickle(path: str, allowed: Set[str] | None) -> tuple[dict, dict]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    parents: Dict[int, List[Tuple[int,str]]] = defaultdict(list)
    for k, v in obj.items():
        try:
            child = _gid_int(str(k))
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        rel_keys = (set(v.keys()) & set(ALLOWED_RELS_FOR_DAG))
        # If no known keys, try "parents"/aliases

        for rel in (rel_keys or ALLOWED_RELS_FOR_DAG):
            if rel not in v:
                continue
            if allowed and rel not in allowed:
                continue
            rels = v.get(rel, [])
            for pid in rels:
                try:
                    p_int = _gid_int(str(pid))
                except Exception:
                    continue
                parents[child].append((p_int, rel))

    children: Dict[int, List[Tuple[int,str]]] = defaultdict(list)
    for c, lst in parents.items():
        for p, rel in lst:
            children[p].append((c, rel))
    return dict(parents), dict(children)


def main(input_file: str = GO_TERMS_PKL,
         allowed_rels: str = ALLOWED_RELS_FOR_DAG,
         parents_path: str = GO_PARENTS,
         childrens_path: str = GO_CHILDREN,
         string_keys: bool = True # keep keys as string
         ):

    P, C = build_from_pickle(input_file, allowed_rels)

    # Coerce keys
    if string_keys:
        P = { f"GO:{int(k):07d}": [[ f"GO:{int(p):07d}", rel] for (p, rel) in lst ] for k, lst in P.items() }
        C = { f"GO:{int(k):07d}": [[ f"GO:{int(c):07d}", rel] for (c, rel) in lst ] for k, lst in C.items() }
    else:
        P = { int(k): [[ int(p), rel] for (p, rel) in lst ] for k, lst in P.items() }
        C = { int(k): [[ int(c), rel] for (c, rel) in lst ] for k, lst in C.items() }

    with open(parents_path, "w", encoding="utf-8") as f:
        json.dump(P, f, ensure_ascii=False)
    with open(childrens_path, "w", encoding="utf-8") as f:
        json.dump(C, f, ensure_ascii=False)

    print({
        "parents_path": parents_path,
        "children_path": childrens_path,
        "n_parents_nodes": len(P),
        "n_children_nodes": len(C),
        "allowed_rels": allowed_rels
    })

if __name__ == "__main__":
    main()
