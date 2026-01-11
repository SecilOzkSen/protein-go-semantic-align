"""
GO Text JSONL Generator (tokensız, parent name + definition)

Output JSONL example:

{
  "text": "Name: DNA binding\n\nDefinition: Any molecular function ...\n\nParents:\n- nucleic acid binding\n- binding",
  "domain": "Molecular Function",
  "name": "DNA binding"
}

Notes:
- GO id yazmıyoruz
- Parents: is_a + part_of parent GO name'leri (relation etiketi yok)
- Parent sayısı max 3 (deterministik)
"""

import json
import pickle
import re
from typing import Dict, Any, List, Optional


def load_terms(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_positives(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def normalize_domain(ns: str) -> str:
    if not ns:
        return "Other"
    ns_l = ns.lower()
    if "molecular" in ns_l:
        return "Molecular Function"
    if "cellular" in ns_l:
        return "Cellular Component"
    if "biological" in ns_l:
        return "Biological Process"
    return "Other"


def extract_definition(def_str: str) -> str:
    def_str = def_str.strip()
    if not def_str:
        return ""
    match = re.search(r'"(.*?)"', def_str)
    return match.group(1) if match else def_str


def _as_go_id(x: Any) -> Optional[str]:
    """
    Normalize parent identifiers to 'GO:XXXXXXX' if possible.
    Handles:
      - 'GO:0001234'
      - 1234
      - '1234'
    """
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        if s.upper().startswith("GO:"):
            # normalize width if possible
            tail = s.split(":", 1)[1]
            if tail.isdigit():
                return f"GO:{int(tail):07d}"
            return s.upper()
        if s.isdigit():
            return f"GO:{int(s):07d}"
        return None
    if isinstance(x, int):
        return f"GO:{int(x):07d}"
    return None


def _get_parent_ids(data: Dict[str, Any]) -> List[str]:
    """
    Returns parent ids from is_a and part_of fields.
    """
    out: List[str] = []
    for key in ("is_a", "part_of"):
        vals = data.get(key, None)
        if not vals:
            continue
        # allow single value or list
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        for v in vals:
            gid = _as_go_id(v)
            if gid is not None:
                out.append(gid)
    # unique, deterministic
    out = sorted(set(out))
    return out


def _get_term_name(terms: Dict[str, Dict[str, Any]], go_id: str) -> Optional[str]:
    d = terms.get(go_id)
    if not d:
        return None
    name = (d.get("name") or "").strip()
    return name or None


def build_text_entry(
    go_id: str,
    data: Dict[str, Any],
    terms: Dict[str, Dict[str, Any]],
    max_parents: int = 3,
) -> Dict[str, Any]:
    """
    Produces one JSONL entry with tokensız text:
      Name
      Definition
      Parents: - parent name (is_a + part_of merged)
    """
    name = (data.get("name") or "").strip() or go_id
    raw_def = (data.get("definition") or "").strip()
    definition = extract_definition(raw_def)

    ns = (data.get("namespace") or "").strip()
    domain = normalize_domain(ns)

    # parents
    parent_ids = _get_parent_ids(data)
    parent_names: List[str] = []
    for pid in parent_ids:
        pn = _get_term_name(terms, pid)
        if pn:
            parent_names.append(pn)

    # unique + deterministic sort by string
    parent_names = sorted(set(parent_names))

    # cap
    if max_parents is not None and max_parents > 0:
        parent_names = parent_names[:max_parents]

    if len(parent_names) == 0:
        parents_block = "Parents:\n- none"
    else:
        parents_block = "Parents:\n" + "\n".join([f"- {p}" for p in parent_names])

    # final text
    # Definition empty olabilir, yine de format sabit kalsın
    if definition:
        text = f"Name: {name}\n\nDefinition: {definition}\n\n{parents_block}"
    else:
        text = f"Name: {name}\n\n{parents_block}"

    return {
        "domain": domain,
        "name": name,
        "text": text,
    }


def write_jsonl(path: str, examples: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main(
    input_path: str,
    out_path: str,
    pid_positives_path: str = "/Users/secilsen/PhD/protein-go-semantic-align/src/scripts/pid_to_positives_canonical.json",
    max_parents: int = 3,
):
    terms = load_terms(input_path)
    pid_positives = load_positives(pid_positives_path)

    # positives go ids set
    set_gid = set()
    for gids in pid_positives.values():
        for g in gids:
            set_gid.add(f"GO:{int(g):07d}")

    examples: List[Dict[str, Any]] = []
    for gid, data in terms.items():
        if gid not in set_gid:
            continue
        ex = build_text_entry(gid, data, terms, max_parents=max_parents)
        examples.append(ex)

    # stabilite için name bazlı ya da gid bazlı sıralama
    # gid elimizde ama yazmıyoruz, sadece deterministik olması için sort ediyoruz
    examples.sort(key=lambda x: (x["domain"], x["name"]))

    print(f"Total GO terms to write: {len(examples)}")
    print(f"Total pidpositives GO terms: {len(set_gid)}")

    write_jsonl(out_path, examples)
    print(f"Wrote {out_path} with {len(examples)} entries.")


if __name__ == "__main__":
    main(
        "/Users/secilsen/PhD/protein-go-semantic-align/src/data/raw/go_basic_obo_terms_v2.pkl",
        "../data/processed/go_terms/canonical/go_texts_canonical_2.jsonl",
        max_parents=3,
    )