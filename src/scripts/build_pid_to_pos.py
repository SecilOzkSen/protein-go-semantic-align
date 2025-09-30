"""
Convert GOA/CAFA annotation pickles into pi.json (raw positives only).
"""
import json, pickle
from typing import Dict, List, Set
from src.configs.paths import PID_TO_POSITIVES, GOA_PARSED_FILE, GO_TERMS_PKL


def _normalize_go_str(s: str) -> str:
    s = s.strip()
    if s.startswith("GO:"):
        s = s[3:]
    return s


def _create_idmap(go_terms_pkl: str = GO_TERMS_PKL):
    output: Dict[str, int] = {}
    with open(go_terms_pkl, 'rb') as f:
        go_terms = pickle.load(f)
    for go_key, _ in go_terms.items():
        go_S = go_key.split('GO:')[1]
        output[go_S] = int(go_S)
    return output


def main(goa_pickle: str = GOA_PARSED_FILE,
         output: str = PID_TO_POSITIVES,
         drop_empty: bool = True):
    with open(goa_pickle, "rb") as f:
        pid2gos: Dict[str, List[str]] = pickle.load(f)

    go2int = _create_idmap()

    out: Dict[str, List[int]] = {}
    total_p = total_pairs = 0
    kept_p = kept_pairs = 0
    missing_go = 0
    dropped_empty = 0

    for pid, gos in pid2gos.items():
        total_p += 1
        mapped: List[int] = []
        for g in gos:
            total_pairs += 1
            k = _normalize_go_str(str(g))
            if k not in go2int:
                missing_go += 1
                continue
            mapped.append(int(go2int[k]))
            kept_pairs += 1
        if drop_empty and len(mapped) == 0:
            dropped_empty += 1
            continue
        out[pid] = sorted(set(mapped))
        kept_p += 1

    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    stats = {
        "proteins_total": total_p,
        "pairs_total": total_pairs,
        "proteins_kept": kept_p,
        "pairs_kept_mapped": kept_pairs,
        "go_ids_missing_in_idmap": missing_go,
        "proteins_dropped_empty": dropped_empty,
        "output_path": output,
    }
    print(stats)


if __name__ == "__main__":
    main()
