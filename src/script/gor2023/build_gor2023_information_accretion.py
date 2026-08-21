#!/usr/bin/env python3
"""Build GOR2023 information-accretion (IA) weights.

The implementation follows the CAFA information-accretion definition:

    IA(t) = -log P(t = 1 | all direct parents of t = 1)

Counts are protein counts after propagating each released training annotation
through ``is_a`` and ``part_of`` ancestors. A pseudocount of one is applied:

    IA(t) = -log((1 + N(t)) / (1 + N(all direct parents of t)))

The logarithm base does not affect weighted precision, weighted recall, or
weighted Fmax because it multiplies every IA value by the same constant. This
script uses the natural logarithm, matching the original CAFA2 implementation.

Outputs are directly readable by BioComputingUP/CAFA-evaluator's ``-ia``
argument: one ``GO_ID<TAB>IA`` pair per line, without a header.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Set

BRANCHES = {
    "mf": "molecular_function",
    "bp": "biological_process",
    "cc": "cellular_component",
}


@dataclass
class Term:
    go_id: str
    namespace: str = ""
    name: str = ""
    parents: List[str] = field(default_factory=list)
    alt_ids: List[str] = field(default_factory=list)
    obsolete: bool = False


def normalize_go_id(value: object) -> str:
    text = str(value).strip()
    if text.upper().startswith("GO:"):
        return "GO:" + text.split(":", 1)[1].zfill(7)
    if text.isdigit():
        return f"GO:{int(text):07d}"
    raise ValueError(f"Invalid GO identifier: {value!r}")


def parse_obo(path: Path) -> tuple[Dict[str, Term], Dict[str, str]]:
    terms: Dict[str, Term] = {}
    current: MutableMapping[str, object] | None = None

    def finish() -> None:
        nonlocal current
        if not current or "id" not in current:
            current = None
            return
        go_id = normalize_go_id(current["id"])
        term = Term(
            go_id=go_id,
            namespace=str(current.get("namespace", "")),
            name=str(current.get("name", "")),
            parents=list(current.get("parents", [])),
            alt_ids=list(current.get("alt_ids", [])),
            obsolete=bool(current.get("obsolete", False)),
        )
        terms[go_id] = term
        current = None

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "[Term]":
                finish()
                current = {"parents": [], "alt_ids": []}
                continue
            if line.startswith("["):
                finish()
                continue
            if current is None or not line or line.startswith("!"):
                continue

            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:].strip()
            elif line.startswith("alt_id: "):
                current["alt_ids"].append(normalize_go_id(line[8:].strip()))
            elif line.startswith("is_a: "):
                parent = line[6:].split("!", 1)[0].strip()
                current["parents"].append(normalize_go_id(parent))
            elif line.startswith("relationship: part_of "):
                parent = line[len("relationship: part_of "):].split("!", 1)[0].strip()
                current["parents"].append(normalize_go_id(parent))
            elif line == "is_obsolete: true":
                current["obsolete"] = True
        finish()

    active = {go_id: term for go_id, term in terms.items() if not term.obsolete}
    alt_to_primary: Dict[str, str] = {}
    for go_id, term in active.items():
        for alt_id in term.alt_ids:
            if alt_id in alt_to_primary and alt_to_primary[alt_id] != go_id:
                raise RuntimeError(f"Ambiguous alt_id {alt_id}")
            alt_to_primary[alt_id] = go_id

    active_ids = set(active)
    for term in active.values():
        term.parents = sorted({p for p in term.parents if p in active_ids})
    return active, alt_to_primary


def load_annotations(
        path: Path,
        active_terms: Mapping[str, Term],
        alt_to_primary: Mapping[str, str],
        namespace: str,
) -> Dict[str, Set[str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected PID-to-GO JSON object in {path}")

    result: Dict[str, Set[str]] = {}
    unresolved: Counter[str] = Counter()
    wrong_namespace: Counter[str] = Counter()
    for pid, values in raw.items():
        labels: Set[str] = set()
        for value in values or []:
            go_id = normalize_go_id(value)
            go_id = alt_to_primary.get(go_id, go_id)
            term = active_terms.get(go_id)
            if term is None:
                unresolved[go_id] += 1
                continue
            if term.namespace != namespace:
                wrong_namespace[go_id] += 1
                continue
            labels.add(go_id)
        result[str(pid)] = labels

    if unresolved:
        examples = unresolved.most_common(10)
        raise RuntimeError(f"Unresolved active annotations in {path}: {examples}")
    if wrong_namespace:
        examples = wrong_namespace.most_common(10)
        raise RuntimeError(f"Namespace-mismatched annotations in {path}: {examples}")
    return result


def ancestor_closure(
        go_id: str,
        terms: Mapping[str, Term],
        namespace: str,
        memo: MutableMapping[str, frozenset[str]],
        visiting: Set[str],
) -> frozenset[str]:
    if go_id in memo:
        return memo[go_id]
    if go_id in visiting:
        raise RuntimeError(f"Cycle detected in GO graph at {go_id}")
    visiting.add(go_id)
    closure = {go_id}
    for parent in terms[go_id].parents:
        if terms[parent].namespace == namespace:
            closure.update(ancestor_closure(parent, terms, namespace, memo, visiting))
    visiting.remove(go_id)
    memo[go_id] = frozenset(closure)
    return memo[go_id]


def compute_branch_ia(
        annotations: Mapping[str, Set[str]],
        terms: Mapping[str, Term],
        namespace: str,
) -> tuple[Dict[str, float], Dict[str, dict]]:
    branch_ids = sorted(go_id for go_id, term in terms.items() if term.namespace == namespace)
    branch_set = set(branch_ids)
    memo: Dict[str, frozenset[str]] = {}
    supports: Dict[str, Set[str]] = {go_id: set() for go_id in branch_ids}

    for pid, labels in annotations.items():
        propagated: Set[str] = set()
        for go_id in labels:
            propagated.update(ancestor_closure(go_id, terms, namespace, memo, set()))
        for go_id in propagated:
            if go_id in branch_set:
                supports[go_id].add(pid)

    ia: Dict[str, float] = {}
    details: Dict[str, dict] = {}
    tolerance = 1e-12
    for go_id in branch_ids:
        parents = [p for p in terms[go_id].parents if p in branch_set]
        n_term = len(supports[go_id])
        if not parents:
            n_parent = len(annotations)
            value = 0.0
        else:
            parent_support = set(supports[parents[0]])
            for parent in parents[1:]:
                parent_support.intersection_update(supports[parent])
            n_parent = len(parent_support)
            if n_term > n_parent:
                raise RuntimeError(
                    f"Closure invariant failed for {go_id}: term={n_term}, parents={n_parent}"
                )
            probability = (1.0 + n_term) / (1.0 + n_parent)
            value = -math.log(probability)
            if value < -tolerance:
                raise RuntimeError(f"Negative IA for {go_id}: {value}")
            value = max(0.0, value)

        ia[go_id] = value
        details[go_id] = {
            "go_id": go_id,
            "name": terms[go_id].name,
            "namespace": namespace,
            "parents": parents,
            "term_support": n_term,
            "all_parent_support": n_parent,
            "ia": value,
        }
    return ia, details


def first_existing(paths: Iterable[Path]) -> Path:
    candidates = list(paths)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate paths exists:\n  " + "\n  ".join(map(str, candidates)))


def write_ia(path: Path, values: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for go_id in sorted(values):
            handle.write(f"{go_id}\t{values[go_id]:.15g}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/GOR2023"))
    parser.add_argument(
        "--obo",
        type=Path,
        default=Path("/workspace/GOR2023/go/ontology/go-2023-01-01.obo"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/GOR2023/evaluation/ia"),
    )
    args = parser.parse_args()

    terms, alt_to_primary = parse_obo(args.obo)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined: Dict[str, float] = {}
    audit = {
        "definition": "-ln((1 + N(term)) / (1 + N(all direct parents)))",
        "pseudocount": 1,
        "relations": ["is_a", "part_of"],
        "ontology": str(args.obo),
        "ontology_active_terms": len(terms),
        "branches": {},
    }

    print("=" * 72)
    print("GOR2023 INFORMATION ACCRETION BUILD")
    print("=" * 72)
    for branch, namespace in BRANCHES.items():
        annotation_path = first_existing(
            [
                args.root / branch / "pid_to_go_train_canonical.json",
                args.root / branch / "pid_to_go_train_released_canonical.json",
                args.root / "go" / "canonical" / branch / "pid_to_go_train_released_canonical.json",
                args.root / "go" / "canonical" / f"{branch}_pid_to_go_train_released_canonical.json",
            ]
        )
        annotations = load_annotations(annotation_path, terms, alt_to_primary, namespace)
        values, details = compute_branch_ia(annotations, terms, namespace)
        overlap = set(combined).intersection(values)
        if overlap:
            raise RuntimeError(f"Cross-branch GO overlap: {sorted(overlap)[:10]}")
        combined.update(values)

        write_ia(args.output_dir / f"{branch}_information_accretion.tsv", values)
        with (args.output_dir / f"{branch}_information_accretion_details.json").open(
                "w", encoding="utf-8"
        ) as handle:
            json.dump(details, handle, indent=2, sort_keys=True)

        arr = list(values.values())
        nonzero = sum(value > 0.0 for value in arr)
        zero = len(arr) - nonzero
        audit["branches"][branch.upper()] = {
            "namespace": namespace,
            "annotations": str(annotation_path),
            "proteins": len(annotations),
            "terms": len(values),
            "nonzero_ia": nonzero,
            "zero_ia": zero,
            "min_ia": min(arr) if arr else None,
            "max_ia": max(arr) if arr else None,
            "mean_ia": sum(arr) / len(arr) if arr else None,
        }
        print(
            f"{branch.upper()}: proteins={len(annotations):,} terms={len(values):,} "
            f"nonzero={nonzero:,} zero={zero:,} max={max(arr):.6f}"
        )

    write_ia(args.output_dir / "information_accretion.tsv", combined)
    audit["combined_terms"] = len(combined)
    with (args.output_dir / "information_accretion_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)

    print(f"COMBINED: {len(combined):,}")
    print(f"Outputs: {args.output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
