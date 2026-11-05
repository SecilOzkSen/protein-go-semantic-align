"""
GO Text JSONL Generator, markerless segments + typed ontology parents.

Output JSONL example:

{
  "go_id": "GO:0000001",
  "domain": "Biological Process",
  "namespace": "biological process",
  "name": "mitochondrion inheritance",
  "definition": "distribution of mitochondria...",
  "is_a_parents": ["organelle inheritance", "mitochondrion distribution"],
  "part_of_parents": [],
  "segments": {
    "name": "mitochondrion inheritance.",
    "namespace": "biological process.",
    "definition": "distribution of mitochondria...",
    "is_a": "organelle inheritance; mitochondrion distribution.",
    "part_of": "none."
  },
  "segment_order": ["name", "definition", "is_a"],
  "segment_format": "markerless_v1",
  "text": "mitochondrion inheritance.\ndistribution of mitochondria...\norganelle inheritance; mitochondrion distribution."
}

Design:
- No custom special tokens.
- No textual segment markers such as Name:, Definition:, or Is-a parents:.
- Segment identity is carried structurally by the separate segment axis.
- is_a and part_of are NOT merged.
- Path is intentionally excluded for the first stable retriever run.
- GO id is kept as metadata because downstream cache/eval usually needs it.
- The locked retriever input is name + definition + is_a.
"""

import json
import pickle
import re
from typing import Dict, Any, List, Optional, Iterable


# -----------------------------
# IO
# -----------------------------

def load_terms(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_positives(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, examples: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# -----------------------------
# Normalization helpers
# -----------------------------

def normalize_namespace(ns: str) -> str:
    """
    Text form used inside GO text.
    """
    if not ns:
        return "other"

    ns_l = ns.strip().lower().replace("_", " ")

    if "molecular" in ns_l:
        return "molecular function"
    if "cellular" in ns_l:
        return "cellular component"
    if "biological" in ns_l:
        return "biological process"

    return ns_l or "other"


def normalize_domain(ns: str) -> str:
    """
    Metadata form.
    """
    ns_norm = normalize_namespace(ns)

    if ns_norm == "molecular function":
        return "Molecular Function"
    if ns_norm == "cellular component":
        return "Cellular Component"
    if ns_norm == "biological process":
        return "Biological Process"

    return "Other"


def clean_text(s: Any) -> str:
    if s is None:
        return ""

    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_definition(def_str: Any) -> str:
    """
    Handles OBO-style definitions:
      '"Any molecular function ..." [GOC:...]'
    and already-clean definitions.
    """
    def_str = clean_text(def_str)

    if not def_str:
        return ""

    match = re.search(r'"(.*?)"', def_str)
    if match:
        return clean_text(match.group(1))

    return def_str


def extract_synonym(s: Any) -> Optional[str]:
    """
    Handles OBO-style synonyms:
      '"foo bar" EXACT []'
    """
    s = clean_text(s)
    if not s:
        return None

    match = re.search(r'"(.*?)"', s)
    if match:
        s = clean_text(match.group(1))

    return s or None


def _as_go_id(x: Any) -> Optional[str]:
    """
    Normalize parent identifiers to GO:XXXXXXX when possible.

    Handles:
      - "GO:0001234"
      - "GO:0001234 ! name"
      - "part_of GO:0001234"
      - 1234
      - "1234"
    """
    if x is None:
        return None

    if isinstance(x, int):
        return f"GO:{int(x):07d}"

    s = str(x).strip()
    if not s:
        return None

    m = re.search(r"GO:\s*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"GO:{int(m.group(1)):07d}"

    if s.isdigit():
        return f"GO:{int(s):07d}"

    return None


def _iter_values(vals: Any) -> Iterable[Any]:
    if vals is None:
        return []
    if isinstance(vals, (list, tuple, set)):
        return vals
    return [vals]


def _get_term_name(
        terms: Dict[str, Dict[str, Any]],
        go_id: str,
) -> Optional[str]:
    data = terms.get(go_id)

    if not data:
        return None

    if data.get("is_obsolete", False):
        return None

    name = clean_text(data.get("name") or "")
    return name or None


# -----------------------------
# Parent extraction
# -----------------------------

def _get_relation_parent_ids(data: Dict[str, Any], relation: str) -> List[str]:
    """
    Extract parent ids for a specific relation.

    Primary expected keys:
      data["is_a"]
      data["part_of"]

    Also tries to handle common alternatives:
      data["relationship"] = ["part_of GO:000xxxx", ...]
      data["relationships"] = [...]
      data["relations"] = {"part_of": [...], "is_a": [...]}
    """
    relation = relation.strip().lower()
    out: List[str] = []

    # Direct field, preferred.
    vals = data.get(relation, None)
    for v in _iter_values(vals):
        gid = _as_go_id(v)
        if gid is not None:
            out.append(gid)

    # Dict-style relation containers.
    for container_key in ("relations", "relationships_by_type"):
        rels = data.get(container_key, None)
        if isinstance(rels, dict):
            vals = rels.get(relation, None)
            for v in _iter_values(vals):
                gid = _as_go_id(v)
                if gid is not None:
                    out.append(gid)

    # OBO-style relationship lines.
    if relation == "part_of":
        for container_key in ("relationship", "relationships"):
            vals = data.get(container_key, None)
            for v in _iter_values(vals):
                s = str(v).strip()
                if s.lower().startswith("part_of") or " part_of " in f" {s.lower()} ":
                    gid = _as_go_id(s)
                    if gid is not None:
                        out.append(gid)

    # Unique deterministic by GO id.
    return sorted(set(out))


def _parent_names_from_ids(
        terms: Dict[str, Dict[str, Any]],
        parent_ids: List[str],
        max_parents: Optional[int],
) -> List[str]:
    names: List[str] = []

    for pid in parent_ids:
        pn = _get_term_name(terms, pid)
        if pn:
            names.append(pn)

    # Unique deterministic by string.
    names = sorted(set(names))

    if max_parents is not None and max_parents > 0:
        names = names[:max_parents]

    return names


# -----------------------------
# Text formatting
# -----------------------------

def format_list_inline(items: List[str]) -> str:
    if not items:
        return "none"
    return "; ".join(items)


def build_text_entry(
        go_id: str,
        data: Dict[str, Any],
        terms: Dict[str, Dict[str, Any]],
        max_is_a_parents: int = 3,
        max_part_of_parents: int = 3,
        include_synonyms: bool = False,
        max_synonyms: int = 3,
        segment_order: List[str] = None  # Leave empty if you need full text.
) -> Dict[str, Any]:
    go_id_norm = _as_go_id(go_id) or str(go_id)

    name = clean_text(data.get("name") or "") or go_id_norm

    raw_def = data.get("definition") or data.get("def") or ""
    definition = extract_definition(raw_def)

    ns_raw = clean_text(data.get("namespace") or "")
    namespace = normalize_namespace(ns_raw)
    domain = normalize_domain(ns_raw)

    # Typed parents.
    is_a_ids = _get_relation_parent_ids(data, "is_a")
    part_of_ids = _get_relation_parent_ids(data, "part_of")

    is_a_parent_names = _parent_names_from_ids(
        terms=terms,
        parent_ids=is_a_ids,
        max_parents=max_is_a_parents,
    )

    part_of_parent_names = _parent_names_from_ids(
        terms=terms,
        parent_ids=part_of_ids,
        max_parents=max_part_of_parents,
    )

    # Optional synonyms, off by default for the first stable run.
    synonyms: List[str] = []
    if include_synonyms:
        raw_synonyms = data.get("synonym") or data.get("synonyms") or []
        for s in _iter_values(raw_synonyms):
            syn = extract_synonym(s)
            if syn:
                synonyms.append(syn)

        synonyms = sorted(set(synonyms))

        if max_synonyms is not None and max_synonyms > 0:
            synonyms = synonyms[:max_synonyms]

    # Markerless segments. Segment identity is represented structurally by
    # segment_order and by the separate segment axis in the GO text store.
    segments: Dict[str, str] = {
        "name": ensure_sentence(name),
        "namespace": ensure_sentence(namespace),
        "definition": ensure_sentence(definition),
        "is_a": ensure_sentence(format_list_inline(is_a_parent_names)),
        "part_of": ensure_sentence(format_list_inline(part_of_parent_names)),
    }

    if include_synonyms:
        segments["synonyms"] = ensure_sentence(format_list_inline(synonyms))

    if segment_order is None:
        # Full text
        segment_order = ["name", "namespace", "definition", "is_a", "part_of"]
    else:
        # Never mutate a list supplied by the caller.
        segment_order = list(segment_order)

    if include_synonyms:
        segment_order.append("synonyms")

    text = "\n".join(segments[k] for k in segment_order)

    return {
        "go_id": go_id_norm,
        "domain": domain,
        "namespace": namespace,
        "name": name,
        "definition": definition,
        "is_a_parent_ids": is_a_ids,
        "part_of_parent_ids": part_of_ids,
        "is_a_parents": is_a_parent_names,
        "part_of_parents": part_of_parent_names,
        "segments": segments,
        "segment_order": segment_order,
        "segment_format": "markerless_v1",
        "text": text,
    }


def ensure_sentence(s: str) -> str:
    s = clean_text(s)
    if not s:
        return "none"
    if s[-1] in ".!?":
        return s
    return s + "."


# -----------------------------
# Main
# -----------------------------

def main(
        input_path: str,
        out_path: str,
        pid_positives_path: Optional[str] = None,
        only_positive_terms: bool = False,
        max_is_a_parents: int = 3,
        max_part_of_parents: int = 3,
        include_synonyms: bool = False,
        max_synonyms: int = 3,
        include_obsolete: bool = False,
        segment_order: Optional[List[str]] = None,
) -> None:
    terms = load_terms(input_path)

    positive_go_ids = set()

    if pid_positives_path is not None:
        pid_positives = load_positives(
            pid_positives_path
        )

        for go_ids in pid_positives.values():
            for go_id in go_ids:
                go_id_norm = _as_go_id(go_id)

                if go_id_norm is not None:
                    positive_go_ids.add(go_id_norm)

    examples: List[Dict[str, Any]] = []

    dropped_not_positive = 0
    dropped_no_go_id = 0
    dropped_obsolete = 0
    dropped_unknown_namespace = 0

    valid_namespaces = {
        "biological_function",
        "biological_process",
        "molecular_function",
        "cellular_component",
    }

    for go_id, data in terms.items():
        go_id_norm = _as_go_id(go_id)

        if go_id_norm is None:
            dropped_no_go_id += 1
            continue

        if (
                not include_obsolete
                and data.get("is_obsolete", False)
        ):
            dropped_obsolete += 1
            continue

        namespace_raw = clean_text(
            data.get("namespace") or ""
        )

        if namespace_raw not in valid_namespaces:
            dropped_unknown_namespace += 1
            continue

        if (
                only_positive_terms
                and go_id_norm not in positive_go_ids
        ):
            dropped_not_positive += 1
            continue

        example = build_text_entry(
            go_id=go_id_norm,
            data=data,
            terms=terms,
            max_is_a_parents=max_is_a_parents,
            max_part_of_parents=max_part_of_parents,
            include_synonyms=include_synonyms,
            max_synonyms=max_synonyms,
            segment_order=segment_order,
        )

        examples.append(example)

    examples.sort(
        key=lambda item: (
            item["domain"],
            item["name"],
            item["go_id"],
        )
    )

    domain_counts = {}

    for example in examples:
        domain = example["domain"]
        domain_counts[domain] = (
                domain_counts.get(domain, 0) + 1
        )

    print(f"Total GO terms in vocab: {len(terms)}")
    print(
        "Total positive GO terms: "
        f"{len(positive_go_ids)}"
    )
    print(f"Total GO terms written: {len(examples)}")
    print(f"Dropped no GO ID: {dropped_no_go_id}")
    print(f"Dropped obsolete: {dropped_obsolete}")
    print(
        "Dropped unknown namespace: "
        f"{dropped_unknown_namespace}"
    )
    print(
        "Dropped not positive: "
        f"{dropped_not_positive}"
    )

    print("\nWritten terms by domain:")

    for domain, count in sorted(
            domain_counts.items()
    ):
        print(f"  {domain}: {count}")

    write_jsonl(out_path, examples)

    print(
        f"\nWrote {out_path} "
        f"with {len(examples)} entries."
    )



# MZSGO extension: OBO input and exact candidate-universe auditing.
import argparse
from collections import Counter
from pathlib import Path
import hashlib


def parse_obo(path):
    terms = {}
    current = None
    kind = None
    def flush():
        if kind == "Term" and current and current.get("id"):
            gid = current.pop("id")
            if gid in terms:
                raise ValueError(f"Duplicate ontology ID: {gid}")
            terms[gid] = current.copy()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                flush()
                kind = line[1:-1]
                current = {}
                continue
            if kind != "Term" or not line or line.startswith("!") or ": " not in line:
                continue
            k, v = line.split(": ", 1)
            if k in ("id", "name", "namespace", "def"):
                current[k] = v
            elif k in ("is_a", "alt_id", "replaced_by", "consider"):
                current.setdefault(k, []).append(v)
            elif k == "relationship" and v.startswith("part_of "):
                current.setdefault("part_of", []).append(v)
            elif k == "is_obsolete":
                current[k] = v.lower() == "true"
    flush()
    return terms


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_mzsgo():
    ap = argparse.ArgumentParser(description="Prepare canonical MZSGO GO text for BP, MF or CC")
    ap.add_argument("--branch", required=True, type=str.upper, choices=("BP", "MF", "CC"))
    ap.add_argument("--root", type=Path, default=Path("/workspace/MZSGO_temporal"))
    ap.add_argument("--obo", type=Path, required=True)
    ap.add_argument("--candidate-ids", type=Path, default=None)
    ap.add_argument("--temporal-ids", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    branch = args.branch.lower()
    processed = args.root / "processed" / branch
    candidate_path = args.candidate_ids or processed / f"candidate_go_ids_{branch}.json"
    temporal_path = args.temporal_ids or processed / f"temporal_go_ids_{branch}.json"
    out_path = args.out or processed / "go_texts_canonical_segmented.jsonl"
    report_path = args.report or out_path.with_suffix(".audit.json")
    expected_ns = {
        "BP": "biological process", "MF": "molecular function", "CC": "cellular component"
    }[args.branch]

    terms = parse_obo(args.obo)
    candidate_raw = json.loads(candidate_path.read_text())
    temporal_raw = json.loads(temporal_path.read_text())
    if not isinstance(candidate_raw, list) or not isinstance(temporal_raw, list):
        raise ValueError("Expected JSON lists for candidate and temporal IDs")
    candidates_raw = [_as_go_id(x) for x in candidate_raw]
    temporal_raw_ids = [_as_go_id(x) for x in temporal_raw]
    if (None in candidates_raw or None in temporal_raw_ids
            or len(set(candidates_raw)) != len(candidates_raw)
            or len(set(temporal_raw_ids)) != len(temporal_raw_ids)):
        raise ValueError("Malformed or duplicate GO IDs")
    candidates, temporal = set(candidates_raw), set(temporal_raw_ids)
    if not temporal <= candidates:
        raise ValueError("Temporal GO IDs not subset of candidates")

    alt_to_primary = {}
    for gid, row in terms.items():
        for alt in row.get("alt_id", []):
            alt_id = _as_go_id(alt)
            if alt_id in alt_to_primary and alt_to_primary[alt_id] != gid:
                raise ValueError(f"Ambiguous alternate ID {alt_id}")
            alt_to_primary[alt_id] = gid

    unresolved, obsolete, wrong_ns, alternate = [], [], [], {}
    for gid in sorted(candidates):
        row = terms.get(gid)
        if row is None:
            unresolved.append(gid)
            if gid in alt_to_primary:
                alternate[gid] = alt_to_primary[gid]
        elif row.get("is_obsolete"):
            obsolete.append(gid)
        elif normalize_namespace(row.get("namespace", "")) != expected_ns:
            wrong_ns.append(gid)

    report = {
        "branch": args.branch, "expected_namespace": expected_ns,
        "ontology": str(args.obo), "ontology_sha256": sha256(args.obo),
        "candidate_ids": str(candidate_path), "candidate_sha256": sha256(candidate_path),
        "temporal_ids": str(temporal_path), "temporal_sha256": sha256(temporal_path),
        "candidate_count": len(candidates), "temporal_count": len(temporal),
        "seen_count": len(candidates - temporal),
        "unresolved": unresolved, "obsolete": obsolete,
        "wrong_namespace": wrong_ns, "alternate_id_suggestions": alternate,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    if unresolved or obsolete or wrong_ns:
        raise RuntimeError(f"Ontology parity failed. See {report_path}. No JSONL written.")

    examples = []
    for gid in sorted(candidates):
        entry = build_text_entry(gid, terms[gid], terms,
                                 max_is_a_parents=3, max_part_of_parents=3,
                                 segment_order=["name", "definition", "is_a"])
        # Preserve one-to-one ID/name order for parent-bank construction.
        parent_ids = [pid for pid in entry["is_a_parent_ids"]
                      if pid in terms and not terms[pid].get("is_obsolete")
                      and _get_term_name(terms, pid)]
        entry["is_a_parent_ids"] = parent_ids
        entry["is_a_parents"] = [_get_term_name(terms, pid) for pid in parent_ids]
        entry["segments"]["is_a"] = ensure_sentence(
            format_list_inline(entry["is_a_parents"][:3]))
        entry["text"] = "\n".join(entry["segments"][k] for k in entry["segment_order"])
        examples.append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out_path), examples)
    report.update({
        "written": len(examples),
        "missing_definitions": sum(not e["definition"] for e in examples),
        "no_is_a_parents": sum(not e["is_a_parent_ids"] for e in examples),
        "output_sha256": sha256(out_path),
    })
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if not isinstance(v, (list, dict))}, indent=2))
    print("Saved:", out_path, report_path)


if __name__ == "__main__":
    run_mzsgo()
