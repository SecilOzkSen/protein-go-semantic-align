import json
from pathlib import Path

import numpy as np
import obonet

GO_OBO = Path(
    "/workspace/stargo/datasets/pfresgo/go.obo"
)

EVAL_IDS = Path(
    "/workspace/candidate_dumps/"
    "name_def_isa_global_local_weakpos_slots8_step82334_train_top500/"
    "eval_go_ids.npy"
)

OUT = Path(
    "/workspace/data_pfresgo/processed/"
    "pfresgo_bp_child_to_parents.json"
)


def go_to_int(go_id: str) -> int:
    return int(str(go_id).split(":")[-1])


# Exact GO vocabulary seen by the BP reranker.
eval_ids = np.load(EVAL_IDS).astype(np.int64)
eval_set = set(int(x) for x in eval_ids.tolist())

print("Evaluation GO terms:", len(eval_set))

graph = obonet.read_obo(str(GO_OBO))

child_to_parents = {}

for child in graph.nodes:
    try:
        child_int = go_to_int(child)
    except Exception:
        continue

    if child_int not in eval_set:
        continue

    parents = []

    # Direct GO parents only.
    for parent in graph.successors(child):
        try:
            parent_int = go_to_int(parent)
        except Exception:
            continue

        # Keep only parents that are themselves predicted
        # in the BP benchmark space.
        if parent_int in eval_set:
            parents.append(parent_int)

    if parents:
        child_to_parents[child_int] = sorted(set(parents))

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT.open("w", encoding="utf-8") as f:
    json.dump(
        child_to_parents,
        f,
        indent=2,
        sort_keys=True,
    )

print("Children with benchmark parents:", len(child_to_parents))
print(
    "Direct edges:",
    sum(len(v) for v in child_to_parents.values()),
)
print("Saved:", OUT)

print("\nExamples:")
for i, (child, parents) in enumerate(child_to_parents.items()):
    print(child, "->", parents)
    if i >= 4:
        break