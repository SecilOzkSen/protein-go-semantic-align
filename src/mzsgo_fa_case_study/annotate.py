"""Add independently sourced UniProt features after exact sequence verification.

The input FASTA must be the original sequence file used to prepare the benchmark
embeddings. A current UniProt entry is used only as an interpretive annotation,
not as a model input or as evidence that a feature explains the selected GO.
No sequence alignment, isoform substitution, training, or score changes occur.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.error import URLError
from urllib.request import Request, urlopen


ACCESSION = re.compile(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\Z")
FEATURE_TYPES = {"Domain", "Region", "Active site", "Binding site", "DNA binding", "Zinc finger", "Motif"}
MAX_RESPONSE_BYTES = 20 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def accession_from_id(identifier):
    """Accept a canonical accession or an explicit sp|accession|entry header.

    Isoform suffixes and other dataset-specific identifiers are not rewritten.
    """
    if not isinstance(identifier, str):
        return None
    value = identifier.strip()
    if "|" in value:
        parts = value.split("|")
        if len(parts) != 3 or parts[0] not in {"sp", "tr"} or not parts[2]:
            return None
        value = parts[1]
    return value if ACCESSION.fullmatch(value) else None


def read_fasta(path):
    """Index original FASTA IDs and their explicit accession aliases."""
    sequences = {}
    header, chunks = None, []

    def finish():
        if header is None:
            return
        sequence = "".join(chunks).upper()
        require(sequence and re.fullmatch(r"[A-Z*.\-]+", sequence),
                f"Empty or invalid FASTA sequence for {header}")
        aliases = {header}
        accession = accession_from_id(header)
        if accession:
            aliases.add(accession)
        for alias in aliases:
            require(alias not in sequences or sequences[alias] == sequence,
                    f"Ambiguous FASTA identifier with different sequences: {alias}")
            sequences[alias] = sequence

    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                words = line[1:].strip().split()
                require(bool(words), "Empty FASTA header")
                header, chunks = words[0], []
            else:
                require(header is not None, "FASTA sequence encountered before its header")
                chunks.append("".join(line.split()))
    finish()
    require(bool(sequences), "Original benchmark FASTA contains no sequences")
    return sequences


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fetch_record(accession, cache_dir, timeout=30):
    """Fetch or reuse one public record and retain byte-level provenance."""
    require(accession_from_id(accession) == accession, "Invalid UniProt accession")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    record_path = cache_dir / f"{accession}.json"
    metadata_path = cache_dir / f"{accession}.metadata.json"
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    if record_path.is_file():
        require(record_path.stat().st_size <= MAX_RESPONSE_BYTES, "Oversized UniProt cached record")
        raw = record_path.read_bytes()
        record = json.loads(raw)
        cached = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        digest = hashlib.sha256(raw).hexdigest()
        if cached:
            require(cached.get("sha256") == digest and cached.get("source_url") == url,
                    "UniProt cache metadata does not match record bytes or accession")
        audit = {"source_url": url, "cache_path": str(record_path.resolve()),
                 "sha256": digest, "retrieved_at": cached.get("retrieved_at"),
                 "cache_reused": True,
                 "retrieval_timestamp_known": bool(cached.get("retrieved_at"))}
    else:
        request = Request(url, headers={"Accept": "application/json",
                                       "User-Agent": "FA-local-evidence-case-study/1.0"})
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        require(len(raw) <= MAX_RESPONSE_BYTES, "Oversized UniProt response")
        record = json.loads(raw)
        require(isinstance(record, dict), "UniProt response is not a JSON object")
        retrieved_at = datetime.now(timezone.utc).isoformat()
        audit = {"source_url": url, "cache_path": str(record_path.resolve()),
                 "sha256": hashlib.sha256(raw).hexdigest(), "retrieved_at": retrieved_at,
                 "cache_reused": False, "retrieval_timestamp_known": True}
        record_path.write_bytes(raw)
        _write_json(metadata_path, {key: audit[key] for key in ("source_url", "sha256", "retrieved_at")})
    require(isinstance(record, dict), "UniProt record is not a JSON object")
    return record, audit


def _exact_position(position):
    if not isinstance(position, dict) or position.get("modifier") != "EXACT":
        return None
    value = position.get("value")
    return value if type(value) is int and value >= 1 else None


def extract_features(record, protein_length):
    """Keep exact in-range positions; preserve evidence without upgrading it."""
    primary = record.get("primaryAccession", "")
    source = f"https://www.uniprot.org/uniprotkb/{primary}/entry"
    features, skipped = [], {"unrequested_type": 0, "nonexact_or_invalid_coordinates": 0}
    raw_features = record.get("features", [])
    require(isinstance(raw_features, list), "UniProt features must be a list")
    for raw in raw_features:
        if not isinstance(raw, dict) or raw.get("type") not in FEATURE_TYPES:
            skipped["unrequested_type"] += 1
            continue
        location = raw.get("location", {})
        if not isinstance(location, dict):
            skipped["nonexact_or_invalid_coordinates"] += 1
            continue
        start, end = _exact_position(location.get("start")), _exact_position(location.get("end"))
        if start is None or end is None or not (1 <= start <= end <= protein_length):
            skipped["nonexact_or_invalid_coordinates"] += 1
            continue
        # A molecule field can refer to another isoform or processed molecule;
        # reject it rather than assuming canonical sequence coordinates.
        if location.get("sequence") or raw.get("molecule"):
            skipped["nonexact_or_invalid_coordinates"] += 1
            continue
        evidence = [dict(item) for item in raw.get("evidences", []) if isinstance(item, dict)]
        codes = sorted({str(item["evidenceCode"]) for item in evidence if item.get("evidenceCode")})
        kind = ("experimental (ECO:0000269)" if "ECO:0000269" in codes else
                "other recorded evidence" if codes else "not specified")
        description = str(raw.get("description", "")).strip()
        ligand = raw.get("ligand", {})
        ligand_name = ligand.get("name", "") if isinstance(ligand, dict) else ""
        label = description or (str(ligand_name).strip()) or str(raw["type"])
        features.append({"start": start, "end": end, "label": label,
                         "type": raw["type"], "evidence_kind": kind,
                         "evidence_codes": codes, "evidences": evidence,
                         "source": source, "sequence_verified": True,
                         "go_relevance": "not_assessed",
                         "feature_id": raw.get("featureId")})
    return sorted(features, key=lambda item: (item["start"], item["end"], item["type"])), skipped


def annotate_case(case, fasta_sequences, cache_dir, timeout=30):
    result = deepcopy(case)
    # Never retain stale regions when fresh verification fails.
    result["annotations"] = []
    audit = {"sequence_verified": False,
             "verification_policy": "exact original FASTA and UniProt sequence equality",
             "go_feature_relationship": "not_assessed; requires biological interpretation",
             "scores_changed": False}
    result["annotation_audit"] = audit
    protein_id = case.get("protein_id")
    accession = accession_from_id(protein_id)
    if accession is None:
        audit["status"] = "unsupported_accession"
        return result
    audit["requested_accession"] = accession
    sequence = fasta_sequences.get(protein_id, fasta_sequences.get(accession))
    if sequence is None:
        audit["status"] = "missing_original_fasta_sequence"
        return result
    audit["benchmark_sequence_sha256"] = sequence_hash(sequence)
    audit["benchmark_sequence_length"] = len(sequence)
    if type(case.get("protein_length")) is not int or case["protein_length"] != len(sequence):
        audit["status"] = "embedding_sequence_length_mismatch"
        return result
    try:
        record, record_audit = fetch_record(accession, cache_dir, timeout)
        audit["uniprot_record"] = record_audit
        primary = record.get("primaryAccession")
        accessions = {primary, *record.get("secondaryAccessions", [])}
        if not isinstance(primary, str) or not ACCESSION.fullmatch(primary) or accession not in accessions:
            audit["status"] = "uniprot_accession_mismatch"
            return result
        audit["primary_accession"] = primary
        raw_sequence = record.get("sequence", {}).get("value")
        require(isinstance(raw_sequence, str) and raw_sequence,
                "UniProt sequence is missing")
        uniprot_sequence = "".join(raw_sequence.split()).upper()
        audit["uniprot_sequence_sha256"] = sequence_hash(uniprot_sequence)
        audit["uniprot_sequence_length"] = len(uniprot_sequence)
        audit["entry_audit"] = record.get("entryAudit", {})
        if sequence != uniprot_sequence:
            audit["status"] = "uniprot_sequence_mismatch"
            return result
        annotations, skipped = extract_features(record, len(sequence))
        result["annotations"] = annotations
        audit.update({"status": "verified_exact_sequence", "sequence_verified": True,
                      "n_retained_features": len(annotations), "skipped_features": skipped,
                      "entry_type": record.get("entryType")})
    except (URLError, TimeoutError, OSError, ValueError, TypeError, UnicodeError) as error:
        audit.update({"status": "uniprot_annotation_unavailable",
                      "error_type": type(error).__name__, "error": str(error)[:500]})
        result["annotations"] = []
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--fasta", required=True, help="Original benchmark FASTA used for the saved residue embeddings")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    evidence_path, fasta_path, output = map(lambda p: Path(p).expanduser().resolve(),
                                          (args.evidence, args.fasta, args.out_dir))
    require(args.timeout > 0 and args.timeout <= 60, "Timeout must be positive and at most 60 seconds")
    require(evidence_path.is_file() and fasta_path.is_file(), "Evidence JSON and original FASTA must exist")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Annotation output directory must be new or empty")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    require(isinstance(evidence, dict) and evidence.get("format") == "mzsgo_fa_local_evidence_v1",
            "Unsupported local evidence format")
    require(isinstance(evidence.get("cases"), list) and evidence["cases"], "Evidence contains no cases")
    require(all(isinstance(case, dict) for case in evidence["cases"]), "Invalid case entries")
    input_sha, fasta_sha = sha256(evidence_path), sha256(fasta_path)
    sequences = read_fasta(fasta_path)
    cache = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else output / "uniprot_cache"
    result = deepcopy(evidence)
    result["cases"] = [annotate_case(case, sequences, cache, args.timeout) for case in evidence["cases"]]
    result["annotation_provenance"] = {
        "original_evidence": str(evidence_path), "original_evidence_sha256": input_sha,
        "original_fasta": str(fasta_path), "original_fasta_sha256": fasta_sha,
        "annotation_created_at": datetime.now(timezone.utc).isoformat(),
        "source": "UniProt REST API, current records cached with retrieval provenance",
        "coordinate_convention": "1-based inclusive, exact positions only",
        "note": "Interpretive annotations only. Sequence identity does not establish the feature's relevance to the target GO. No scores, ranks, model parameters, or benchmark labels were changed.",
    }
    require(sha256(evidence_path) == input_sha and sha256(fasta_path) == fasta_sha,
            "An annotation input changed during execution")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "evidence.json", result)
    _write_json(output / "annotation_report.json", {
        "provenance": result["annotation_provenance"],
        "cases": [{"case_id": case.get("case_id"), "protein_id": case.get("protein_id"),
                   "go_id": case.get("go_id"), **case["annotation_audit"]} for case in result["cases"]],
    })
    for case in result["cases"]:
        print(f"[ANNOTATION] {case['protein_id']} {case.get('go_id', '')}: "
              f"{case['annotation_audit']['status']}; features={len(case['annotations'])}")
    print(f"[SAVED] {output / 'evidence.json'}")


if __name__ == "__main__":
    main()
