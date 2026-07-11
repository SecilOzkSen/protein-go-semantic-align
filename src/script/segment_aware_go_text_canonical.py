"""
GO Text JSONL Generator, natural markers + typed ontology parents

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
    "name": "Name: mitochondrion inheritance.",
    "namespace": "Namespace: biological process.",
    "definition": "Definition: distribution of mitochondria...",
    "is_a": "Is-a parents: organelle inheritance; mitochondrion distribution.",
    "part_of": "Part-of parents: none."
  },
  "text": "Name: mitochondrion inheritance.\nNamespace: biological process.\nDefinition: distribution of mitochondria...\nIs-a parents: organelle inheritance; mitochondrion distribution.\nPart-of parents: none."
}

Design:
- No custom special tokens.
- Natural markers only: Name, Namespace, Definition, Is-a parents, Part-of parents.
- is_a and part_of are NOT merged.
- Path is intentionally excluded for the first stable retriever run.
- GO id is kept as metadata because downstream cache/eval usually needs it.
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

    # Natural marker segments.
    # Keep these markers stable across all GO terms.
    segments: Dict[str, str] = {
        "name": f"Name: {ensure_sentence(name)}",
        "namespace": f"Namespace: {ensure_sentence(namespace)}",
        "definition": f"Definition: {ensure_sentence(definition)}",
        "is_a": f"Is-a parents: {ensure_sentence(format_list_inline(is_a_parent_names))}",
        "part_of": f"Part-of parents: {ensure_sentence(format_list_inline(part_of_parent_names))}",
    }

    if include_synonyms:
        segments["synonyms"] = f"Synonyms: {format_list_inline(synonyms)}."

    segment_order = ["name", "namespace", "definition", "is_a", "part_of"]
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


if __name__ == "__main__":
    main(
        input_path=(
            "/workspace/data_pfresgo/"
            "processed/go_vocab.pkl"
        ),
        out_path=(
            "/workspace/data_pfresgo/"
            "processed/"
            "go_texts_canonical_segmented.jsonl"
        ),

        # PFresGO full branch ontology kullanacak.
        pid_positives_path=None,
        only_positive_terms=False,

        max_is_a_parents=3,
        max_part_of_parents=3,

        include_synonyms=False,
        max_synonyms=3,

        # Obsolete terms candidate vocabulary'ye alınmayacak.
        include_obsolete=False,
    )