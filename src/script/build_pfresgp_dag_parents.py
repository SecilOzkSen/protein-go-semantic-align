# src/script/build_pfresgo_dag_parents.py

import json
from pathlib import Path

import obonet

GO_OBO = Path("/workspace/stargo/datasets/pfresgo/go.obo")
BP_IDS = Path("/workspace/data_pfresgo/processed/go_ids_bp.json")
OUT = Path("/workspace/data_pfresgo/processed/pfresgo_bp_child_to_parents.json")


def go_to_int(go_id: str) -> int:
    return int(go_id.split(":")[1])


with BP_IDS.open("r", encoding="utf-8") as f:
    raw = json.load(f)

# Handle either list[int], list[str], or dictionary-like vocab.
if isinstance(raw, dict):
    raw_ids = list(raw.keys())
else:
    raw_ids = raw

bp_ids = set()

for x in raw_ids:
    if isinstance(x, int):
        bp_ids.add(x)
    else:
        s = str(x)
        if s.startswith("GO:"):
            bp_ids.add(go_to_int(s))
        else:
            bp_ids.add(int(s))

graph = obonet.read_obo(str(GO_OBO))

child_to_parents = {}

for child in graph.nodes:
    try:
        child_int = go_to_int(child)
    except Exception:
        continue

    if child_int not in bp_ids:
        continue

    parents = []

    # In obonet's GO graph, edges are child -> parent.
    for parent in graph.successors(child):
        try:
            parent_int = go_to_int(parent)
        except Exception:
            continue

        if parent_int in bp_ids:
            parents.append(parent_int)

    if parents:
        child_to_parents[child_int] = sorted(set(parents))

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8") as f:
    json.dump(child_to_parents, f, indent=2, sort_keys=True)

print("BP GO terms:", len(bp_ids))
print("Children with >=1 benchmark parent:", len(child_to_parents))
print("Edges:", sum(len(x) for x in child_to_parents.values()))
print("Saved:", OUT)