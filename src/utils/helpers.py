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
    original = load_raw_json(path)
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


def load_go_texts_by_phase(go_text_folder: str, phase: int = 0) -> Dict[int, str]:
    if phase < 0: #ablation 1
        fname = "go_texts_canonical.jsonl"
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





