from src.utils import load_raw_json, load_raw_pickle, normalize_go_str
from src.configs.paths import COMMON_IC_GO_TERMS_ID_ONLY_JSON, GO_TERMS_PKL, GO_ANCESTOR_STOPLIST

INCLUDE_COMMON = False                         # keep False for a minimal list
COMMON_PATH = COMMON_IC_GO_TERMS_ID_ONLY_JSON    # used only if INCLUDE_COMMON=True
GO_TERMS_PATH = GO_TERMS_PKL       # optional validation (if exists)
EXTRA_IDS = []

# The three GO roots
ROOT_IDS = [
    8150,  # biological_process
    3674,  # molecular_function
    5575,  # cellular_component
]

def _validate_ids(ids, go_terms_dict):
    if go_terms_dict is None:
        return list(ids), []
    keyset = set([int(normalize_go_str(key)) for key in go_terms_dict.keys()])
    valid = [g for g in ids if g in keyset]
    missing = [g for g in ids if g not in keyset]
    return valid, missing

def main():
    stopset = set(ROOT_IDS)
    stopset.update(EXTRA_IDS)

    # Optionally include common (very frequent) GO terms if requested
    if INCLUDE_COMMON and COMMON_PATH.exists():
        common = load_raw_json(COMMON_PATH)
        if isinstance(common, (set, list, tuple)):
            stopset.update(common)
        elif isinstance(common, dict):
            # accept dict form like {go_id: count}
            stopset.update(common.keys())

    # Optional: validate against go_terms.pkl or .json if present (for typos)
    go_terms = load_raw_pickle(GO_TERMS_PATH)

    valid_ids, missing_ids = _validate_ids(stopset, go_terms)

    with GO_ANCESTOR_STOPLIST.open("w", encoding="utf-8") as f:
        for go_id in sorted(set(valid_ids)):
            f.write(f"{go_id}\n")

    print(f"Wrote {len(valid_ids)} IDs to {GO_ANCESTOR_STOPLIST}")
    if missing_ids:
        print(f"Note: {len(missing_ids)} IDs not found in go_terms and were skipped:")
        for m in missing_ids:
            print(f"  - {m}")

if __name__ == "__main__":
    main()
