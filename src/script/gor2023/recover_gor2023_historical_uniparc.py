#!/usr/bin/env python3

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests

UNIPARC_SEARCH = "https://rest.uniprot.org/uniparc/search"


def parse_date(x):
    if not x:
        return None
    try:
        return datetime.strptime(x[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return sorted({
            line.strip()
            for line in f
            if line.strip()
        })


def write_fasta_record(f, accession, sequence, upi):
    f.write(f">{accession} historical_uniparc={upi}\n")
    for i in range(0, len(sequence), 80):
        f.write(sequence[i:i + 80] + "\n")


def request_json(session, params, retries=5):
    delay = 2

    for attempt in range(retries):
        try:
            r = session.get(
                UNIPARC_SEARCH,
                params=params,
                timeout=120,
            )

            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", delay))
                print(f"[429] sleeping {wait}s")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue

            if r.status_code >= 500:
                print(
                    f"[{r.status_code}] server error, "
                    f"retry in {delay}s"
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            if r.status_code == 400:
                print("[400]", r.text)
                r.raise_for_status()

            r.raise_for_status()
            return r.json()

        except requests.RequestException as e:
            if attempt == retries - 1:
                raise

            print(f"[WARN] {e}, retry in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)

    raise RuntimeError("UniParc request failed")


UNIPARC_ENTRY = "https://rest.uniprot.org/uniparc"


def fetch_full_uniparc_entry(session, upi, retries=5):
    url = f"{UNIPARC_ENTRY}/{upi}"

    delay = 2

    for attempt in range(retries):
        try:
            r = session.get(
                url,
                params={"format": "json"},
                timeout=120,
            )

            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", delay))
                print(f"[429] sleeping {wait}s")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue

            if r.status_code >= 500:
                print(
                    f"[{r.status_code}] server error, "
                    f"retry in {delay}s"
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            r.raise_for_status()
            return r.json()

        except requests.RequestException:
            if attempt == retries - 1:
                raise

            time.sleep(delay)
            delay = min(delay * 2, 60)

    raise RuntimeError(f"Failed to retrieve UniParc entry {upi}")


def candidate_matches_accession(xref, accession):
    candidates = {
        str(xref.get("id", "")),
        str(xref.get("accession", "")),
        str(xref.get("proteinId", "")),
    }

    return accession in candidates


def xref_date_range(xref):
    first = (
            xref.get("created")
            or xref.get("firstSeen")
            or xref.get("firstSeenDate")
    )

    last = (
            xref.get("last")
            or xref.get("lastUpdated")
            or xref.get("lastSeen")
            or xref.get("lastSeenDate")
    )

    return parse_date(first), parse_date(last)


def extract_sequence(entry):
    seq_obj = entry.get("sequence")

    if isinstance(seq_obj, dict):
        return seq_obj.get("value")

    if isinstance(seq_obj, str):
        return seq_obj

    return None


def get_xrefs(entry):
    for key in (
            "uniParcCrossReferences",
            "crossReferences",
            "databaseCrossReferences",
            "dbReferences",
    ):
        xrefs = entry.get(key)

        if isinstance(xrefs, list):
            return xrefs

    return []


def resolve_accession_at_cutoff(entry, accession, cutoff):
    """
    Returns True if this UniParc sequence has a cross-reference
    showing that the requested accession existed at the cutoff.
    """

    matching = []

    for xref in get_xrefs(entry):
        if not candidate_matches_accession(xref, accession):
            continue

        first, last = xref_date_range(xref)

        matching.append({
            "first": first,
            "last": last,
            "raw": xref,
        })

    for m in matching:
        first = m["first"]
        last = m["last"]

        # We require evidence that the accession existed by cutoff.
        if first and first > cutoff:
            continue

        # If last is absent, treat it as still active.
        if last and last < cutoff:
            continue

        return True, matching

    return False, matching


def query_accession(session, accession):
    """
    Generic UniParc query. The API searches cross-references
    as part of UniParc indexed content.
    """

    params = {
        "query": accession,
        "format": "json",
        "size": 25,
    }

    return request_json(session, params)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_ids",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cutoff",
        type=str,
        default="2023-06-28",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
    )

    args = parser.parse_args()

    cutoff = parse_date(args.cutoff)
    ids = read_ids(args.input_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = args.out_dir / "historical_recovered.fasta"
    audit_path = args.out_dir / "historical_recovery.jsonl"
    unresolved_path = args.out_dir / "unresolved.txt"

    resolved_ids = set()

    # Resume support
    if audit_path.exists():
        with audit_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "RECOVERED":
                        resolved_ids.add(rec["accession"])
                except Exception:
                    pass

    remaining = [
        x for x in ids
        if x not in resolved_ids
    ]

    print("=" * 72)
    print("GOR2023 HISTORICAL UNIPARC RECOVERY")
    print("=" * 72)
    print(f"Cutoff:           {args.cutoff}")
    print(f"Requested:        {len(ids):,}")
    print(f"Already resolved: {len(resolved_ids):,}")
    print(f"Remaining:        {len(remaining):,}")

    session = requests.Session()
    session.headers.update({
        "User-Agent":
            "GOAligneRR-GOR2023-historical-recovery/1.0"
    })

    fasta_mode = "a" if fasta_path.exists() else "w"

    unresolved = []

    with fasta_path.open(
            fasta_mode,
            encoding="utf-8",
    ) as fasta_out, audit_path.open(
        "a",
        encoding="utf-8",
    ) as audit_out:

        for i, accession in enumerate(
                remaining,
                start=1,
        ):
            print(
                f"[{i}/{len(remaining)}] "
                f"{accession}"
            )

            try:
                data = query_accession(
                    session,
                    accession,
                )

                search_entries = data.get("results", [])

                valid_candidates = []

                for search_entry in search_entries:

                    upi = (
                            search_entry.get("uniParcId")
                            or search_entry.get("upi")
                    )

                    if not upi:
                        continue

                    full_entry = fetch_full_uniparc_entry(
                        session,
                        upi,
                    )

                    sequence = extract_sequence(full_entry)

                    if not sequence:
                        continue

                    valid, matching_xrefs = (
                        resolve_accession_at_cutoff(
                            full_entry,
                            accession,
                            cutoff,
                        )
                    )

                    if valid:
                        valid_candidates.append({
                            "entry": full_entry,
                            "sequence": sequence,
                            "matching_xrefs": matching_xrefs,
                        })

                if len(valid_candidates) == 1:
                    cand = valid_candidates[0]

                    entry = cand["entry"]

                    upi = (
                            entry.get("uniParcId")
                            or entry.get("upi")
                            or "UNKNOWN_UPI"
                    )

                    write_fasta_record(
                        fasta_out,
                        accession,
                        cand["sequence"],
                        upi,
                    )

                    record = {
                        "accession": accession,
                        "status": "RECOVERED",
                        "upi": upi,
                        "sequence_length":
                            len(cand["sequence"]),
                        "num_api_results":
                            len(search_entries),
                        "num_valid_candidates":
                            1,
                    }

                elif len(valid_candidates) > 1:
                    record = {
                        "accession": accession,
                        "status":
                            "AMBIGUOUS_MULTIPLE_CANDIDATES",
                        "num_api_results": len(search_entries),
                        "num_valid_candidates":
                            len(valid_candidates),
                    }

                    unresolved.append(accession)

                else:
                    record = {
                        "accession": accession,
                        "status":
                            "NO_VALID_CUTOFF_SEQUENCE",
                        "num_api_results": len(search_entries),
                        "num_valid_candidates":
                            0,
                    }

                    unresolved.append(accession)

                audit_out.write(
                    json.dumps(record) + "\n"
                )
                audit_out.flush()
                fasta_out.flush()

            except Exception as e:
                print(
                    f"[ERROR] {accession}: {e}"
                )

                record = {
                    "accession": accession,
                    "status": "ERROR",
                    "error": str(e),
                }

                audit_out.write(
                    json.dumps(record) + "\n"
                )
                audit_out.flush()

                unresolved.append(accession)

            time.sleep(args.sleep)

    with unresolved_path.open(
            "w",
            encoding="utf-8",
    ) as f:
        for accession in sorted(set(unresolved)):
            f.write(accession + "\n")

    print("\nDone.")
    print(f"FASTA      -> {fasta_path}")
    print(f"Audit      -> {audit_path}")
    print(f"Unresolved -> {unresolved_path}")


if __name__ == "__main__":
    main()