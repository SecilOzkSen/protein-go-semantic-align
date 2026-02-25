"""
Canonical GO Text JSONL Generator for BiomedBERT

{
  "go_id": "GO:0003677",
  "domain": "Molecular Function",
  "name": "DNA binding",
  "definition": "Any molecular function by which a gene product interacts selectively and non-covalently with DNA (deoxyribonucleic acid)."
}

"""

import json
import pickle
import re
from typing import Dict, Any


def load_terms(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads the serialized GO term dictionary from a pickle file.
    Expected fields per GO:
      - name
      - definition
      - namespace
      - is_a, part_of
    """
    with open(path, "rb") as f:
        return pickle.load(f)

def load_positives(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def normalize_domain(ns: str) -> str:
    """
    Map GO namespace to human readable domain string.
    """
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
    """
    GO OBO formatında definition genelde:
      "Some text" [PMID:..., ...]
    şeklinde. Tırnak içini çekiyoruz, yoksa olduğu gibi bırakıyoruz.
    """
    def_str = def_str.strip()
    if not def_str:
        return ""
    match = re.search(r'"(.*?)"', def_str)
    return match.group(1) if match else def_str


def build_entry(go_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tek bir GO terimi için canonical JSON entry üretir.
    """
    name = (data.get("name") or go_id).strip()
    raw_def = (data.get("definition") or "").strip()
    definition = extract_definition(raw_def)
    ns = (data.get("namespace") or "").strip()
    domain = normalize_domain(ns)

    return {
        "go_id": go_id,
        "domain": domain,
        "name": name,
        "definition": definition
    }


def write_jsonl(path: str, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main(input_path: str, out_path: str, pid_positives_path: str = "/Users/secilsen/PhD/protein-go-semantic-align/src/scripts/pid_to_positives_canonical.json"):
    terms = load_terms(input_path)
    pid_positives = load_positives(pid_positives_path)

    set_gid = set()
    for gids in pid_positives.values():
        for g in gids:
            set_gid.add(f"GO:{g:07d}")

    examples = []
    for gid, data in terms.items():
        if gid not in set_gid:
            continue
        ex = build_entry(gid, data)
        examples.append(ex)

    # stabilite için GO ID'ye göre sırala
    examples.sort(key=lambda x: x["go_id"])

    print(f"Total canonical GO terms to write: {len(examples)}")
    print(f"Total pidpositives GO terms: {len(set_gid)}")

    write_jsonl(out_path, examples)
    print(f"Wrote {out_path} with {len(examples)} entries in canonical format.")


if __name__ == "__main__":
    main(
        "/Users/secilsen/PhD/protein-go-semantic-align/src/data/raw/go_basic_obo_terms_v2.pkl",
        "../data/processed/go_terms/canonical/go_texts_canonical.jsonl",
    )
