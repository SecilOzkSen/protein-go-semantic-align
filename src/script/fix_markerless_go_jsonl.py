#!/usr/bin/env python3
"""Remove legacy field markers from a markerless GO JSONL, atomically."""
import argparse
import json
import os
import re
from pathlib import Path

PREFIX = {
    "name": re.compile(r"^\s*name\s*:\s*", re.I),
    "namespace": re.compile(r"^\s*namespace\s*:\s*", re.I),
    "definition": re.compile(r"^\s*definition\s*:\s*", re.I),
    "is_a": re.compile(r"^\s*is(?:_|\s+)a\s*:\s*", re.I),
    "part_of": re.compile(r"^\s*part(?:_|\s+)of\s*:\s*", re.I),
}


def strip_prefix(kind, value):
    text = str(value)
    pattern = PREFIX.get(kind)
    if pattern is None:
        return text, False
    cleaned, n = pattern.subn("", text, count=1)
    return cleaned, bool(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="", help="Defaults to atomic in-place replacement")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if args.output else src
    tmp = dst.with_name(dst.name + ".tmp")
    changed_rows = changed_segments = rows = 0

    with src.open(encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            row_changed = False
            segments = dict(row.get("segments") or {})
            for kind, value in list(segments.items()):
                cleaned, changed = strip_prefix(kind, value)
                if changed:
                    segments[kind] = cleaned
                    changed_segments += 1
                    row_changed = True

            # Keep duplicated top-level fields consistent with segments.
            for kind in ("name", "namespace", "definition"):
                if kind in row:
                    cleaned, changed = strip_prefix(kind, row[kind])
                    if changed:
                        row[kind] = cleaned
                        row_changed = True

            row["segments"] = segments
            order = row.get("segment_order") or ["name", "definition", "is_a"]
            row["text"] = "\n".join(
                str(segments[k]).strip() for k in order
                if k in segments and str(segments[k]).strip()
            )
            row["segment_format"] = "markerless_v1"
            changed_rows += int(row_changed)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(tmp, dst)
    print(f"rows={rows} changed_rows={changed_rows} changed_segments={changed_segments}")
    print(f"output={dst}")


if __name__ == "__main__":
    main()
