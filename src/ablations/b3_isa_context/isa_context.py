"""Audit the existing cached canonical is_a text channel without rebuilding it.

This module never encodes text, pools parent-name embeddings or reads protein
annotations. It follows the canonical loader's is_a text/presence convention:
an explicit nonempty segments.is_a value takes precedence; otherwise parent
names in is_a_parents are joined by '; ' and terminated with punctuation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ISA_CONTEXT_POLICY = "canonical_cached_is_a_text_v1"


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _clean(value):
    return "" if value is None else " ".join(str(value).split())


def _go_id(value):
    if isinstance(value, bool):
        raise ValueError("Invalid canonical GO ID")
    text = str(value).strip()
    if text.upper().startswith("GO:"):
        text = text[3:]
    if not text.isdigit() or int(text) <= 0:
        raise ValueError(f"Invalid canonical GO ID: {value!r}")
    return int(text)


def _canonical_isa(row):
    names = row.get("is_a_parents", []) or []
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list):
        raise ValueError("Canonical is_a_parents must be a name list or string")
    names = [_clean(name) for name in names if _clean(name)]
    joined = "; ".join(names)
    generated = joined + ("" if joined and joined[-1] in ".!?" else ".") if joined else "none."
    segments = row.get("segments")
    explicit = _clean(segments.get("is_a", "")) if isinstance(segments, dict) else ""
    text = explicit or generated
    lower = text.lower()
    if lower.startswith("is-a parents:"):
        raise ValueError("Legacy marker-prefixed is_a context is not accepted by the canonical loader")
    text_present = bool(text) and lower not in {"none", "none."} and not (
        lower.endswith(": none.") or lower.endswith(": none")
    )
    return text, bool(joined) and text_present, names, bool(explicit), explicit != generated if explicit else False


def validate_cached_isa_context(bank_payload, canonical_path, active_go_ids, *, require_any=True):
    """Return a JSON-native audit for the exact existing is_a cache rows.

    The returned identity has no filesystem-dependent paths and is suitable for
    exact train/resume/inference comparison. Direct-parent ID/name agreement is
    checked wherever complete canonical parent records are available; missing
    IDs or missing records are counted explicitly, without deriving new text.
    """
    if not isinstance(bank_payload, dict) or bank_payload.get("format") != "mzsgo_frozen_segment_parent_v1":
        raise ValueError("B3-ISA requires the original frozen segmented GO bank")
    canonical_sha = _sha_file(canonical_path)
    metadata = bank_payload.get("metadata", {})
    if metadata.get("canonical_jsonl_sha256") != canonical_sha:
        raise ValueError("B3-ISA canonical text SHA256 differs from the frozen bank")
    if metadata.get("normalized") is not False:
        raise ValueError("B3-ISA requires the original unnormalized frozen segment embeddings")
    names = list(bank_payload.get("segment_names", []))
    if len(names) != len(set(names)) or "is_a" not in names:
        raise ValueError("B3-ISA requires one explicitly named is_a cache channel")
    active = [_go_id(g) for g in active_go_ids]
    full_ids = [_go_id(g) for g in bank_payload.get("go_ids", [])]
    if not active or len(active) != len(set(active)) or len(full_ids) != len(set(full_ids)):
        raise ValueError("B3-ISA GO IDs must be nonempty and unique")
    lookup = {g: i for i, g in enumerate(full_ids)}
    missing = set(active) - set(lookup)
    if missing:
        raise ValueError(f"B3-ISA active GO IDs missing from frozen bank: {sorted(missing)[:10]}")
    embeddings, presence = bank_payload.get("segment_embs"), bank_payload.get("segment_present")
    if (not isinstance(embeddings, torch.Tensor) or not embeddings.is_floating_point()
            or embeddings.ndim != 3 or embeddings.shape[:2] != (len(full_ids), len(names))):
        raise ValueError("Invalid B3-ISA frozen segment tensor")
    if not isinstance(presence, torch.Tensor) or presence.shape != embeddings.shape[:2]:
        raise ValueError("Invalid B3-ISA segment presence mask")
    if not bool(((presence == 0) | (presence == 1)).all()):
        raise ValueError("B3-ISA presence mask must contain only zero/one values")
    index = torch.tensor([lookup[g] for g in active], dtype=torch.long, device=embeddings.device)
    isa_index = names.index("is_a")
    vectors = embeddings.index_select(0, index)[:, isa_index].detach().float().cpu().contiguous()
    selected_presence = presence.index_select(0, index.to(presence.device))[:, isa_index].bool().cpu()
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("B3-ISA context cache contains nonfinite values")
    records = {}
    with Path(canonical_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "go_id" not in row:
                raise ValueError("Invalid canonical GO text record")
            gid = _go_id(row["go_id"])
            if gid in records:
                raise ValueError(f"Duplicate canonical GO record: {gid}")
            records[gid] = row
    texts, expected_presence = [], []
    explicit_count = override_differs = direct_checked = ids_missing = unresolved_records = 0
    direct_edges_checked = 0
    for gid in active:
        if gid not in records:
            raise ValueError(f"B3-ISA active GO missing from canonical text: {gid}")
        row = records[gid]
        text, present, parent_names, explicit, differs = _canonical_isa(row)
        texts.append([gid, text])
        expected_presence.append(present)
        explicit_count += int(explicit)
        override_differs += int(differs)
        if "is_a_parent_ids" not in row:
            ids_missing += 1
            continue
        ids = row["is_a_parent_ids"] or []
        if not isinstance(ids, list):
            raise ValueError("Canonical is_a_parent_ids must be a list")
        ids = sorted({_go_id(parent) for parent in ids})
        if gid in ids:
            raise ValueError(f"Canonical is_a self-parent for GO {gid}")
        if any(parent not in records or not _clean(records[parent].get("name")) for parent in ids):
            unresolved_records += 1
            continue
        resolved_names = {_clean(records[parent]["name"]) for parent in ids}
        if resolved_names != set(parent_names):
            raise ValueError(f"Canonical direct is_a IDs/names disagree for GO {gid}")
        direct_checked += 1
        direct_edges_checked += len(ids)
    expected = torch.tensor(expected_presence, dtype=torch.bool)
    if not torch.equal(selected_presence, expected):
        mismatch = [g for g, actual, wanted in zip(active, selected_presence.tolist(), expected_presence)
                    if actual != wanted]
        raise ValueError(f"Cached is_a presence differs from canonical is_a text: {mismatch[:10]}")
    count = int(selected_presence.sum())
    if require_any and count == 0:
        raise ValueError("B3-ISA active universe has no usable is_a context; refusing a null context run")
    tensor_digest = hashlib.sha256()
    tensor_digest.update(str((tuple(vectors.shape), str(vectors.dtype))).encode())
    tensor_digest.update(vectors.numpy().tobytes())
    return {
        "policy": ISA_CONTEXT_POLICY,
        "source": "frozen_segment_bank.is_a",
        "semantic_role": "third_learned_GO_text_score_channel",
        "parent_penalty_enabled": False,
        "text_reencoded": False,
        "parent_name_embeddings_averaged": False,
        "canonical_jsonl_sha256": canonical_sha,
        "active_go_ids_sha256": _json_sha(active),
        "canonical_is_a_texts_sha256": _json_sha(texts),
        "is_a_presence_sha256": _json_sha(expected_presence),
        "is_a_float32_vectors_sha256": tensor_digest.hexdigest(),
        "is_a_channel_index": isa_index,
        "n_active_go": len(active),
        "n_active_with_is_a_context": count,
        "n_active_without_is_a_context": len(active) - count,
        "n_explicit_is_a_texts": explicit_count,
        "n_explicit_is_a_texts_differing_from_generated_join": override_differs,
        "n_direct_id_name_records_audited": direct_checked,
        "n_direct_is_a_edges_audited": direct_edges_checked,
        "n_records_without_direct_parent_ids": ids_missing,
        "n_records_with_unresolved_direct_parent_names": unresolved_records,
        "direct_id_audit_policy": "check_complete_records_else_report_unresolved_without_substitution",
        "context_mask_policy": "original_cached_is_a_presence_verified_against_canonical_text",
    }
