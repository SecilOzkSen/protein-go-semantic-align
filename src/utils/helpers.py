from typing import Dict, Any, List
import os
import json
from pathlib import Path
import pickle
def load_raw_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_raw_txt(path:Path):
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def load_go_set(path: Path):
    if not path or not os.path.isfile(path): return []
    original = load_raw_pickle(path)
    return set(int(str(x).replace("GO:", "")) for x in original)

def load_raw_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def normalize_go_str(s: str) -> str:
    s = s.strip()
    if s.startswith("GO:"):
        s = s[3:]
    return s

def load_go_texts(path: str) -> Dict[int, str]:
    """Load GO texts either from JSONL"""
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    for el in records:
        go_id_str = el['go_id']
        go_int = int(go_id_str.replace("GO:", "")) if "GO:" in go_id_str else int(go_id_str)
        out[go_int] = el["text"]
    return out

def load_go_texts_canonical(go_text_path: str, phase=-2) -> Dict[int, str]:
    """
    Canonical GO text loader for format:
    {
      "go_id": "GO:0000015",
      "domain": "...",
      "name": "...",
      "definition": "..."
    }

    Output: {go_int: text}
    where text = name + ". " + definition
    """
    import os, json

    if not os.path.exists(go_text_path):
        raise FileNotFoundError(f"Canonical GO text file not found: {go_text_path}")

    out = {}
    n_lines = 0
    n_ok = 0

    with open(go_text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            n_lines += 1
            try:
                el = json.loads(line)
            except json.JSONDecodeError:
                continue

            # --- GO ID ---
            if "go_id" not in el:
                continue

            gid_raw = el["go_id"]  # string: "GO:0000015"
            if not isinstance(gid_raw, str):
                continue

            if gid_raw.startswith("GO:"):
                gid_raw = gid_raw[3:]  # remove "GO:"
            try:
                go_int = int(gid_raw)
            except ValueError:
                continue

            # --- Text (we combine name + definition) ---
            name = el.get("name", "").strip()
            definition = el.get("definition", "").strip()
            if phase == -2:
                text = el.get("text", "").strip()
            elif name and definition:
                text = f"{name}. {definition}"
            elif name:
                text = name
            elif definition:
                text = definition
            else:
                # no usable text
                continue

            out[go_int] = text
            n_ok += 1

    print(f"[load_go_texts_canonical] Loaded {n_ok} GO terms out of {n_lines} lines")
    return out



def load_go_texts_by_phase(go_text_folder: str, phase: int = 0) -> Dict[int, str]:
    if phase < 0: #ablation 1
        fname = "go_texts_canonical.jsonl"
        path = os.path.join(go_text_folder, fname)
        return load_go_texts_canonical(path, phase=phase)
    else:
        fname = f"go_texts_phase_{phase+1}.jsonl"
        path = os.path.join(go_text_folder, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Phase file not found: {path}")

        out: Dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                el = json.loads(line)
                go_id_str = el["go_id"]
                go_int = int(go_id_str.replace("GO:", "")) if "GO:" in go_id_str else int(go_id_str)
                out[go_int] = el["text"].strip() if "text" in el else el["name"].strip()
        return out

def _coerce_int_list(xs) -> List[int]:
    return [int(x) for x in xs]


def _coerce_id2row(d) -> Dict[int, int]:
    # json loads keys as str
    return {int(k): int(v) for k, v in d.items()}


def _coerce_row2id_list_from_dict(d) -> List[int]:
    """
    Accept {row: gid} or {gid: row}, return row2id list where row2id[row]=gid.
    """
    items = [(int(k), int(v)) for k, v in d.items()]
    if not items:
        return []

    keys = [k for k, _ in items]
    vals = [v for _, v in items]

    def looks_like_rows(arr):
        return min(arr) == 0 and max(arr) == len(arr) - 1 and len(set(arr)) == len(arr)

    if looks_like_rows(keys):
        # {row: gid}
        items.sort(key=lambda kv: kv[0])
        return [gid for _, gid in items]

    if looks_like_rows(vals):
        # {gid: row}
        items.sort(key=lambda kv: kv[1])
        return [gid for gid, _ in items]

    raise ValueError("row2id dict must be {row:gid} or {gid:row} with rows 0..N-1.")

def go_str_to_int_any(x) -> int:
    if isinstance(x, int):
        return int(x)
    s = str(x).strip()
    try:
        return int(normalize_go_str(s))
    except Exception:
        # fallback: GO:0001234
        if s.upper().startswith("GO:"):
            return int(s.split(":")[1])
        return int(s)

def build_altid_map_from_go_terms(go_terms: dict) -> dict[int, int]:
    """
    go_terms: {"GO:0008150": {"alt_id":[...], ...}, ...}
    returns: {alt_int -> primary_int}
    """
    m: dict[int, int] = {}
    for primary_go_str, info in (go_terms or {}).items():
        try:
            primary = go_str_to_int_any(primary_go_str)
        except Exception:
            continue
        alts = info.get("alt_id", []) if isinstance(info, dict) else []
        for a in alts or []:
            try:
                alt = go_str_to_int_any(a)
                m[int(alt)] = int(primary)
            except Exception:
                pass
    return m

def canonicalize_id_list(ids: list, alt_map: dict[int, int]) -> list[int]:
    out = []
    for g in ids:
        gi = go_str_to_int_any(g)
        out.append(int(alt_map.get(gi, gi)))
    # keep deterministic
    return sorted(set(out))

def canonicalize_pid2pos(pid2pos: dict, alt_map: dict[int, int]) -> dict:
    """
    pid2pos: {pid: [go_ids]}
    returns NEW dict with canonicalized ids and duplicates removed.
    """
    new = {}
    for pid, gos in pid2pos.items():
        if not gos:
            new[pid] = []
            continue
        can = canonicalize_id_list(list(gos), alt_map)
        new[pid] = can
    return new





