from typing import Dict, Any, List
import os
import json
from pathlib import Path
import pickle

from configs.paths import GO_VOCAB
ALL_SEGMENT_NAMES = [
    "name",
    "namespace",
    "definition",
    "is_a",
    "part_of",
]

SEGMENT_DEFAULT_TEXT = {
    "name": "none.",
    "namespace": "none.",
    "definition": "none.",
    "is_a": "none.",
    "part_of": "none.",
}

LEGACY_SEGMENT_PREFIXES = {
    "name": "Name:",
    "namespace": "Namespace:",
    "definition": "Definition:",
    "is_a": "Is-a parents:",
    "part_of": "Part-of parents:",
}


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

def _clean_text(x) -> str:
    import re
    if x is None:
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def _ensure_period(x: str) -> str:
    x = _clean_text(x)
    if not x:
        return "none."
    if x[-1] in ".!?":
        return x
    return x + "."


def _segment_is_present(text: str) -> bool:
    t = _clean_text(text).lower()
    if not t:
        return False
    if t in {"none", "none."}:
        return False
    if t.endswith(": none.") or t.endswith(": none"):
        return False
    return True


def _assert_markerless_segment(
        segment_name: str,
        segment_text: str,
        go_id: str,
) -> None:
    """Fail fast if a legacy marker-prefixed segment enters a new run."""
    prefix = LEGACY_SEGMENT_PREFIXES[segment_name]
    if _clean_text(segment_text).lower().startswith(prefix.lower()):
        raise ValueError(
            "Legacy marker-prefixed GO segment detected for "
            f"{go_id}, segment={segment_name!r}: {segment_text!r}. "
            "Regenerate the GO JSONL with segment_format='markerless_v1'."
        )


from typing import Dict, Iterable, Optional, Set, Tuple


def load_go_texts_canonical(
        go_text_path: str,
        phase: int = -2,
        return_segments: bool = False,
        enabled_segments: Optional[Iterable[str]] = None,
        require_markerless: bool = True,
):
    """
    Canonical GO text loader with automatic segment ablation.

    New retriever runs use markerless_v1 segments. With
    require_markerless=True, legacy values such as "Name: ..." fail fast
    instead of silently changing the token distribution.

    enabled_segments örnekleri:
        ["name", "definition"]
        ["name", "definition", "is_a"]
        ["name", "definition", "is_a", "part_of"]

    return_segments=False:
        id2text

    return_segments=True:
        id2text
        id2segments
        id2seg_present
    """
    import json
    import os

    if not os.path.exists(go_text_path):
        raise FileNotFoundError(
            f"Canonical GO text file not found: {go_text_path}"
        )

    if enabled_segments is None:
        enabled: Set[str] = set(ALL_SEGMENT_NAMES)
    else:
        enabled = {str(x).strip() for x in enabled_segments}

    unknown = enabled.difference(ALL_SEGMENT_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown GO segments: {sorted(unknown)}. "
            f"Allowed segments: {ALL_SEGMENT_NAMES}"
        )

    if not enabled:
        raise ValueError("At least one GO text segment must be enabled.")

    # Çıktı segment ekseni yalnızca aktif segmentlerden oluşur.
    segment_names = [
        segment_name
        for segment_name in ALL_SEGMENT_NAMES
        if segment_name in enabled
    ]

    id2text: Dict[int, str] = {}
    id2segments: Dict[int, Dict[str, str]] = {}
    id2seg_present: Dict[int, Dict[str, bool]] = {}

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

            gid_raw = el.get("go_id")
            if not isinstance(gid_raw, str):
                continue

            gid_raw = gid_raw.strip()
            if gid_raw.upper().startswith("GO:"):
                gid_raw = gid_raw.split(":", 1)[1]

            try:
                go_int = int(gid_raw)
            except ValueError:
                continue

            # --------------------------------------------------
            # Raw fields
            # --------------------------------------------------
            name = _clean_text(el.get("name", ""))
            definition = _clean_text(el.get("definition", ""))
            namespace = (
                    _clean_text(el.get("namespace", ""))
                    or _clean_text(el.get("domain", ""))
            )

            is_a_parents = el.get("is_a_parents", []) or []
            part_of_parents = el.get("part_of_parents", []) or []

            if isinstance(is_a_parents, str):
                is_a_parents = [is_a_parents]

            if isinstance(part_of_parents, str):
                part_of_parents = [part_of_parents]

            is_a_txt = "; ".join(
                cleaned
                for parent in is_a_parents
                if (cleaned := _clean_text(parent))
            )

            part_of_txt = "; ".join(
                cleaned
                for parent in part_of_parents
                if (cleaned := _clean_text(parent))
            )

            # --------------------------------------------------
            # Build canonical value for every possible segment
            # --------------------------------------------------
            generated_segments = {
                "name": (
                    _ensure_period(name)
                    if name
                    else "none."
                ),
                "namespace": (
                    _ensure_period(namespace)
                    if namespace
                    else "none."
                ),
                "definition": (
                    _ensure_period(definition)
                    if definition
                    else "none."
                ),
                "is_a": (
                    _ensure_period(is_a_txt)
                    if is_a_txt
                    else "none."
                ),
                "part_of": (
                    _ensure_period(part_of_txt)
                    if part_of_txt
                    else "none."
                ),
            }

            raw_segments = el.get("segments")

            segments: Dict[str, str] = {}

            for segment_name in segment_names:
                # Önce JSONL içindeki segmenti kullan.
                if isinstance(raw_segments, dict):
                    segment_text = _clean_text(
                        raw_segments.get(segment_name, "")
                    )
                else:
                    segment_text = ""

                # JSONL'de yoksa canonical field'lardan üret.
                if not segment_text:
                    segment_text = generated_segments[segment_name]

                if require_markerless:
                    _assert_markerless_segment(
                        segment_name=segment_name,
                        segment_text=segment_text,
                        go_id=f"GO:{go_int:07d}",
                    )

                segments[segment_name] = segment_text

            # --------------------------------------------------
            # Presence: yalnızca enabled segmentler için hesaplanır
            # --------------------------------------------------
            raw_presence = {
                "name": bool(name),
                "namespace": bool(namespace),
                "definition": bool(definition),
                "is_a": bool(is_a_txt),
                "part_of": bool(part_of_txt),
            }

            present = {
                segment_name: (
                        raw_presence[segment_name]
                        and _segment_is_present(segments[segment_name])
                )
                for segment_name in segment_names
            }

            # En az bir aktif segment kullanılabilir olmalı.
            if not any(present.values()):
                continue

            # --------------------------------------------------
            # Full text: eski el["text"] alanını kullanmıyoruz.
            # Ablation'a göre aktif segmentlerden yeniden kuruyoruz.
            # --------------------------------------------------
            text_parts = [
                segments[segment_name]
                for segment_name in segment_names
                if present[segment_name]
            ]

            text = "\n".join(text_parts).strip()
            if not text:
                continue

            id2text[go_int] = text
            id2segments[go_int] = segments
            id2seg_present[go_int] = present

            n_ok += 1

    print(
        "[load_go_texts_canonical] "
        f"enabled_segments={segment_names} | "
        f"require_markerless={require_markerless} | "
        f"loaded={n_ok}/{n_lines}"
    )

    if return_segments:
        return id2text, id2segments, id2seg_present

    return id2text


def load_go_texts_by_phase(
        go_text_folder: str,
        phase: int = 0,
        return_segments: bool = False,
        enabled_segments=None,
        require_markerless: bool = True,
) -> Dict[int, str]:
    if phase < 0:  # ablation 1
        fname = "go_texts_canonical.jsonl"
        path = os.path.join(go_text_folder, fname)
        return load_go_texts_canonical(
            path,
            phase=phase,
            return_segments=return_segments,
            enabled_segments=enabled_segments,
            require_markerless=require_markerless,
        )
    else:
        fname = f"go_texts_phase_{phase + 1}.jsonl"
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

def load_go_namespaces() -> dict[int, str]:
    with open(GO_VOCAB, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = {}
    for k, v in raw.items():
        gid = go_str_to_int_any(k)
        if isinstance(v, dict):
            ns = v.get("namespace", None)
        else:
            ns = None
        if ns is not None:
            out[gid] = str(ns)
    return out

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

def _is_nonempty_segment_text(x: str) -> bool:
    x = (x or "").strip().lower()
    if not x:
        return False
    if x.endswith(": none.") or x.endswith(": none"):
        return False
    return True




