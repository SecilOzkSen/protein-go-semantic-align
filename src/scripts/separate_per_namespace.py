from src.configs.paths import GO_TERMS_PKL
import pickle
from pathlib import Path
import json

def namespace_converter(ns: str) -> str:
    """Convert full namespace name to its abbreviation."""
    ns_map = {
        "biological_process": "BP",
        "cellular_component": "CC",
        "molecular_function": "MF"
    }
    return ns_map.get(ns, "")

def separate_terms_per_namespace(go_terms_pkl: Path = GO_TERMS_PKL) -> dict[str, Any]:
    """
    Load GO terms from a pickle file and separate them by namespace.

    Args:
        go_terms_pkl (Path): Path to the pickle file containing GO terms.

    Returns:
        dict[str, set[int]]: A dictionary with namespaces as keys and sets of GO term IDs as values.
    """

    with open(go_terms_pkl, "rb") as f:
        go_terms = pickle.load(f)
    go_term_pair = dict()

    all_terms = set(go_terms.keys())

    for g in all_terms:
        ns = go_terms[g].get("namespace", "")
        go_term_pair[g] = namespace_converter(ns)

    return go_term_pair

if __name__ == "__main__":
    pairs = separate_terms_per_namespace()
    with open("go_terms_per_namespace.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)