"""
GO Text Generation Pipeline for BiomedBERT
==========================================

Overview
--------
This script generates curriculum-style JSONL datasets (phase_1..phase_4) for GO term text
to be embedded by BiomedBERT. Each phase controls:
- the maximum upward depth for `is_a` and `part_of` paths, and
- the parent selection strategy (FIRST | ALL_TOPK | ALL).

A) Curriculum: only switch the dataset
- phase_1 → phase_2 → phase_3 → (optional) phase_4
- Tokenization parameters and formatting logic remain fixed; only depth and parent strategy change.
- Training stays fast and deterministic because the tokenizer is not reconfigured between phases.

B) Token budget–aware (max_length-aware) formatting
- Measure with the tokenizer and truncate in this order:
  1) DEF (GO term definition)
  2) [ISA] path nodes (nearest → farthest, applying phase-specific strategy)
  3) [PART] path nodes (nearest → farthest, applying phase-specific strategy)
  4) [SYN] (if synonyms exist)
- This ensures the same code works for different max_length values (e.g., 384, 512).
- JSONLs are generated separately per phase, but truncation rules are shared.

C) Negative sampling / augmentation
- Performed at runtime (e.g., in collate_fn or a custom Sampler).
- No need to regenerate tokens — only the positive/negative balance in the batch changes.

Why this approach?
- Curriculum-compatible: phase progression changes only the information content; no tokenization overhead.
- Max-token safe: the budget-aware formatter guarantees that no phase exceeds the context limit.
- Reproducible & fast: JSONL + pre-tokenized Arrow cache makes training start-up very quick.

----------------------------------------------------------------------
Strategies
----------------------------------------------------------------------
* FIRST: Only the first parent of the current GO term is included when building the text
  to be embedded. For each relationship type (is_a or part_of), the algorithm follows the
  chain from the current term up to the specified depth, always picking the first parent
  in the list at each step. This keeps the text short and deterministic, but may ignore
  additional valid parent paths.

* ALL_TOPK: Includes the top-K parents at the first level above the current term, and then
  follows only the first parent for each of those K chains as it ascends to the maximum
  depth. This balances breadth (multiple starting parents) with depth control, preventing
  token explosion. The value of K is configurable per relationship type.

* ALL: Includes all parents at every level up to the specified depth, following all possible
  parent chains without pruning. This covers the entire reachable subgraph in the DAG for
  the given depth, ensuring no parent path is missed. However, this strategy can significantly
  increase token length and may require careful use to avoid exceeding model limits.

----------------------------------------------------------------------
Phases
----------------------------------------------------------------------
* Phase 1: Uses only the is_a relationship with a maximum depth of 1.
  Strategy is FIRST for both is_a and part_of (though part_of is disabled here with depth = 0).
  This produces the shortest possible paths — one direct parent — and is ideal for initial,
  low-token training.

* Phase 2: Uses is_a with depth up to 3 (strategy = ALL_TOPK, K = 2) and part_of with depth = 1
  (strategy = FIRST). This expands the context by including multiple top-level is_a parents
  while keeping part_of short.

* Phase 3: Uses is_a with depth up to 3 and part_of with depth up to 2, both with strategy =
  ALL_TOPK (K = 3 for is_a, K = 2 for part_of). This further broadens the parent coverage
  while still constraining branch width.

* Phase 4 (Experimental): Uses is_a with depth up to 3 and part_of with depth up to 2, both with
  strategy = ALL. This includes all possible parent chains at every level, maximizing coverage
  but also significantly increasing token length. Recommended for ablation or analysis, not for
  routine training.
"""


import json
import pickle
from collections import defaultdict, deque
import re


OUT_FILES = {
    "phase_1": "go_texts_phase_1.jsonl",  # Phase-1: short (DEF + [ISA] depth=1)
    "phase_2": "go_texts_phase_2.jsonl",  # Phase-2: medium (DEF + [ISA] d=2-3 + [PART] d=1)
    "phase_3": "go_texts_phase_3.jsonl",  # Phase-3: full (DEF + [ISA] d=3 + [PART] d=2 + optional synonyms later)
    "phase_4": "go_texts_phase_4.jsonl",  # Phase-3: full (DEF + [ISA] d=3 + [PART] d=2 + optional synonyms later)
}

# per-phase depths
PHASE_CONFIG = {
    "phase_1": {
        "ISA_DEPTH": 1,
        "PART_DEPTH": 0,
        "ISA_STRATEGY": "first",
        "PART_STRATEGY": "first",
        "ISA_TOPK": 1,
        "PART_TOPK": 1
    },
    "phase_2": {
        "ISA_DEPTH": 3,
        "PART_DEPTH": 1,
        "ISA_STRATEGY": "all_topk",
        "PART_STRATEGY": "first",
        "ISA_TOPK": 2,
        "PART_TOPK": 1
    },
    "phase_3": {
        "ISA_DEPTH": 3,
        "PART_DEPTH": 2,
        "ISA_STRATEGY": "all_topk",
        "PART_STRATEGY": "all_topk",
        "ISA_TOPK": 3,
        "PART_TOPK": 2
    },
    "phase_4": {  # Experimental: all parents
        "ISA_DEPTH": 3,
        "PART_DEPTH": 2,
        "ISA_STRATEGY": "all",
        "PART_STRATEGY": "all",
        "ISA_TOPK": None,  # ignored for 'all'
        "PART_TOPK": None
    }
}

TOK_GOPATH = "[GOPATH]"
TOK_PATH   = "[PATH]"
TOK_ISA    = "[ISA]"
TOK_PART   = "[PART]"


def load_terms(path):
    """
    Loads the serialized GO term dictionary from a pickle file.
    :param path: file path (str)
    :return: dict keyed by GO ID with fields like name, definition, namespace, is_a, part_of.
    """
    with open(path, "rb") as f:
        return pickle.load(f)

def dedup_preserve_order(lst):
    """
    Removes duplicates while preserving the original order (O(1) amortized per element using an ordered dict trick).
    :param lst: list
    :return: list without duplicates, same order.
    """
    return list(dict.fromkeys(lst))

def split_relations(terms):
    """
    Builds reverse adjacency lists for is_a and part_of relations, respecting OBO order.
    Also performs final de‑duplication and guards against self‑loops.
    :param terms: terms dict
    :return: (isa_rev, part_rev) where each is dict[child_id] -> list[parent_ids].
    """
    isa_rev = defaultdict(list)
    partof_rev = defaultdict(list)
    if __name__ == '__main__':
        for gid, data in terms.items():
            isa_list = dedup_preserve_order((data.get("is_a") or []))
            partof_list = dedup_preserve_order((data.get("part_of") or []))
            if isa_list:
                isa_rev[gid].extend(isa_list)
            if partof_list:
                partof_rev[gid].extend(partof_list)

    for child, parents in list(isa_rev.items()):
        parents = [p.strip() for p in parents if p]  # cleaning
        parents = [p for p in parents if p != child]  # avoid self-loops
        isa_rev[child] = dedup_preserve_order(parents)  # preserve order

    for child, parents in list(partof_rev.items()):
        parents = [p.strip() for p in parents if p]  # cleaning
        parents = [p for p in parents if p != child]  # avoid self-loops
        partof_rev[child] = dedup_preserve_order(parents)  # preserve order

    empty_count = sum(1 for parents in partof_rev.values() if not parents or len(parents) == 0)
    print(f"Empty lists in part_rev: {empty_count} / {len(partof_rev)}")
    print(partof_rev["GO:0000182"])
    return isa_rev, partof_rev

def ascend_chain_first(rev_adj, start_id, max_depth):
    """
    Walks upward in the DAG along a single chain, choosing the first parent at each level (preserving OBO order), up to max_depth.
    :param rev_adj: reverse adjacency, starting GO ID, depth limit
    :param start_id: starting GO id
    :param max_depth: maximum depth to go
    :return: list of ancestor GO IDs ordered from closest to farthest.
    """
    if max_depth <= 0:
        return []
    path, cur = [], start_id
    seen = {start_id}
    for _ in range(max_depth):
        parents = rev_adj.get(cur, [])
        if not parents:
            break
        next = parents[0] # FIRST PARENT FROM OBO FILE.
        if next in seen:
            break
        path.append(next)
        seen.add(next)
        cur = next
    return path
def build_chains_all(rev_adj, start_id, max_depth):
    if max_depth <= 0:
        return []
    # BFS over paths
    paths = deque([[start_id]])
    results = []

    while paths:
        path = paths.popleft()  # path starts with start_id, then parents...
        cur = path[-1]
        depth_used = len(path) - 1  # excluding start_id
        if depth_used == max_depth:
            # collect this chain (without start_id)
            results.append(path[1:])
            continue
        parents = rev_adj.get(cur, [])
        if not parents:
            # no more parents; collect whatever we have (exclude start_id)
            if depth_used > 0:
                results.append(path[1:])
            continue
        for p in parents:
            if p in path:   # avoid cycles
                continue
            new_path = path + [p]
            paths.append(new_path)

    # Dedup chains preserving order
    uniq = []
    seen = set()
    for ch in results:
        key = tuple(ch)
        if key not in seen and ch:
            seen.add(key)
            uniq.append(ch)
    return uniq

def build_chains_topk_at_level_1(rev_adj, start_id, max_depth, topk):
    """
    Creates multiple chains but only branches at level‑1: takes the first topk parents at the first step,
    then extends each branch with the first‑parent rule for remaining depth.
    :param rev_adj: reverse adjacency
    :param start_id: start ID
    :param max_depth: depth
    :param topk: topk
    :param terms:
    :return: Output: list of chains (each chain is a list of GO IDs)
    """
    if max_depth <= 0:
        return []
    parents_lv1 = rev_adj.get(start_id, [])[:max(0, topk)]
    chains = []
    seen_root = {start_id}
    for p in parents_lv1:
        if not p or p in seen_root:
            continue
        # For this leaf, continue with 'first' for the rest of the depth.
        tail = ascend_chain_first(rev_adj, p, max_depth - 1)
        chain = [p] + tail
        chains.append(dedup_preserve_order(chain))
    return chains


def id2name(terms_dict, gid):
    """
    Helper to convert a GO ID to its name, falling back to the ID if name is missing.
    :param terms_dict: terms dictionary
    :param gid: go id
    :return: name or GOid
    """
    return (terms_dict.get(gid, {}).get("name") or gid).strip()

def extract_definition(def_str):
    match = re.search(r'"(.*?)"', def_str)
    return match.group(1) if match else def_str

def build_text(gid,
               data,
               isa_rev,
               part_rev,
               config,
               terms_dict):
    """
    Composes the final text for a GO term.
    Uses either single‑chain or top‑k‑at-level‑1 strategy per relation (ISA/PART), depending on configuration
    :return: JSON‑serializable dict with go_id, namespace, name, text.
        Name. Definition [GOPATH] [ISA] ancestor1 > ancestor2 [PATH] [PART] ancestor1 > …
    """
    name = (data.get("name") or gid).strip()
    definition = extract_definition((data.get("definition") or "").strip())
    header = f"{name}."
    if definition:
        header += f" {definition}"

    # Start with IS_A chains
    isa_chains = []
    if config["ISA_DEPTH"] > 0:
        if config["ISA_STRATEGY"] == "first":
            c = ascend_chain_first(isa_rev, gid, config["ISA_DEPTH"])
            if c:
                isa_chains.append(c)
        elif config["ISA_STRATEGY"] == "all_topk":
            isa_chains = build_chains_topk_at_level_1(isa_rev,
                                                      gid,
                                                      config["ISA_DEPTH"],
                                                      config.get("ISA_TOPK", 1))
        elif config["ISA_STRATEGY"] == "all":
            isa_chains = build_chains_all(isa_rev, gid, config["ISA_DEPTH"])
    # PART_OF chains
    part_chains = []
    if config["PART_DEPTH"] > 0:
        if config["PART_STRATEGY"] == "first":
            c = ascend_chain_first(part_rev, gid, config["PART_DEPTH"])
            if c:
                part_chains.append(c)
        elif config["PART_STRATEGY"] == "all_topk":
            part_chains = build_chains_topk_at_level_1(part_rev,
                                                       gid,
                                                       config["PART_DEPTH"],
                                                       config.get("PART_TOPK", 1))
        elif config["PART_STRATEGY"] == "all":
            part_chains = build_chains_all(part_rev, gid, config["PART_DEPTH"])

    # NOW building text blocks
    chunks = []
    for chain in isa_chains:
        if chain:
            isa_str = " > ".join((terms_dict.get(x, {}).get("name") or x).strip() for x in chain)
            chunks.append(f"{TOK_ISA} {isa_str}")
    for chain in part_chains:
        if chain:
            part_str = " > ".join((terms_dict.get(x, {}).get("name") or x).strip() for x in chain)
            chunks.append(f"{TOK_PART} {part_str}")

    path_block = f" {TOK_GOPATH} " + f" {TOK_PATH} ".join(chunks) if chunks else ""
    text = (header + path_block).strip()

    if gid == "GO:0000073":
        print()

    return {
        "go_id": gid,
        "namespace": data.get("namespace", ""),
        "name": name,
        "text": text
    }

def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main(input_path):
    terms = load_terms(input_path)
    isa_rev, part_rev = split_relations(terms)

    for phase, out_path in OUT_FILES.items():
        config = PHASE_CONFIG[phase]
        exs = []
        for gid, data in terms.items():
            ex = build_text(
                gid,
                data,
                isa_rev=isa_rev,
                part_rev=part_rev,
                config=config,
                terms_dict=terms
            )
            exs.append(ex)
        # stable order
        exs.sort(key=lambda x: x["go_id"])
        write_jsonl(out_path, exs)
        print(
            f"Wrote {out_path} with {len(exs)} entries | "
            f"ISA(depth={config['ISA_DEPTH']}, strategy={config['ISA_STRATEGY']}) ; "
            f"PART(depth={config['PART_DEPTH']}, strategy={config['PART_STRATEGY']})"
        )

if __name__ == "__main__":
    main("/Users/secilsen/PhD/protein_function_dataset/data/processed/go_basic_obo_terms_v2.pkl")










