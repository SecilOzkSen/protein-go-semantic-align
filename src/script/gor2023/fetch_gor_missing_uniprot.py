#!/usr/bin/env python3

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests

API = "https://rest.uniprot.org/uniprotkb/search"


def read_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return sorted({
            line.strip()
            for line in f
            if line.strip()
        })


def chunks(items: List[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def request_with_retry(
        session: requests.Session,
        params: dict,
        retries: int = 6,
        timeout: int = 120,
):
    delay = 2

    for attempt in range(retries):
        try:
            r = session.get(
                API,
                params=params,
                timeout=timeout,
            )

            if r.status_code == 400:
                print("[400] UniProt response:")
                print(r.text)
                r.raise_for_status()

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = int(retry_after) if retry_after else delay
                print(f"[429] rate limited, sleeping {wait}s")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue

            if 500 <= r.status_code < 600:
                print(
                    f"[{r.status_code}] server error, "
                    f"retrying in {delay}s"
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            r.raise_for_status()
            return r

        except requests.HTTPError:
            # 4xx errors should not be retried
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise

            if attempt == retries - 1:
                raise

            time.sleep(delay)
            delay = min(delay * 2, 60)

        except requests.RequestException as e:
            if attempt == retries - 1:
                raise

            print(
                f"[WARN] request failed: {e}; "
                f"retrying in {delay}s"
            )
            time.sleep(delay)
            delay = min(delay * 2, 60)

    raise RuntimeError("request failed after retries")


def build_query(ids: List[str]) -> str:
    """
    Search both current primary accessions and secondary accessions.

    Example:
      (accession:P12345 OR sec_acc:P12345 OR ...)
    """

    terms = []

    for acc in ids:
        terms.append(f"accession:{acc}")
        terms.append(f"sec_acc:{acc}")

    return "(" + " OR ".join(terms) + ")"


def fetch_batch_json(
        session: requests.Session,
        ids: List[str],
) -> dict:
    query = build_query(ids)

    params = {
        "query": query,
        "format": "json",
        "size": 500,
        "fields": (
            "accession,id,sequence,"
            "date_sequence_modified,"
            "date_modified,"
            "version"
        ),
    }

    r = request_with_retry(session, params)
    return r.json()


def extract_results(
        data: dict,
        requested: Set[str],
) -> Tuple[Dict[str, dict], Set[str]]:
    """
    Map requested accessions to returned UniProt entries.

    Handles:
      - exact primary accession
      - secondary accession appearing in secondaryAccessions
    """

    mapped = {}
    seen_primary = set()

    for entry in data.get("results", []):
        primary = entry.get("primaryAccession")
        if not primary:
            continue

        seen_primary.add(primary)

        secondary = set(entry.get("secondaryAccessions", []))

        matched_requests = set()

        if primary in requested:
            matched_requests.add(primary)

        matched_requests |= (secondary & requested)

        seq_obj = entry.get("sequence", {})
        sequence = seq_obj.get("value")

        entry_info = entry.get("entryAudit", {})

        metadata = {
            "primary_accession": primary,
            "secondary_accessions": sorted(secondary),
            "entry_type": entry.get("entryType"),
            "sequence": sequence,
            "sequence_length": seq_obj.get("length"),
            "sequence_version": seq_obj.get("version"),
            "sequence_modified": entry_info.get(
                "lastSequenceUpdateDate"
            ),
            "entry_modified": entry_info.get(
                "lastAnnotationUpdateDate"
            ),
            "entry_version": entry_info.get("entryVersion"),
            "uni_prot_id": entry.get("uniProtkbId"),
        }

        for requested_acc in matched_requests:
            mapped[requested_acc] = metadata

    found = set(mapped)

    return mapped, found


def write_fasta(
        mappings: Dict[str, dict],
        path: Path,
):
    with path.open("w", encoding="utf-8") as f:
        for requested_acc in sorted(mappings):
            rec = mappings[requested_acc]

            sequence = rec.get("sequence")
            if not sequence:
                continue

            primary = rec["primary_accession"]

            f.write(
                f">{requested_acc} "
                f"current_primary={primary}\n"
            )

            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i + 80] + "\n")


def write_metadata(
        mappings: Dict[str, dict],
        path: Path,
):
    fields = [
        "requested_accession",
        "primary_accession",
        "entry_type",
        "uni_prot_id",
        "sequence_length",
        "sequence_version",
        "sequence_modified",
        "entry_modified",
        "entry_version",
        "secondary_accessions",
    ]

    with path.open(
            "w",
            encoding="utf-8",
            newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
        )

        writer.writeheader()

        for requested_acc in sorted(mappings):
            rec = mappings[requested_acc]

            writer.writerow({
                "requested_accession": requested_acc,
                "primary_accession": rec["primary_accession"],
                "entry_type": rec.get("entry_type"),
                "uni_prot_id": rec.get("uni_prot_id"),
                "sequence_length": rec.get("sequence_length"),
                "sequence_version": rec.get("sequence_version"),
                "sequence_modified": rec.get(
                    "sequence_modified"
                ),
                "entry_modified": rec.get(
                    "entry_modified"
                ),
                "entry_version": rec.get("entry_version"),
                "secondary_accessions": ",".join(
                    rec.get("secondary_accessions", [])
                ),
            })


def save_ids(ids: Set[str], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for acc in sorted(ids):
            f.write(acc + "\n")


def save_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )


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
        "--batch_size",
        type=int,
        default=100,
        help=(
            "Number of accessions per API query. "
            "100 is conservative and reliable."
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Pause between successful API requests.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_ids = read_ids(args.input_ids)

    print("=" * 72)
    print("GOR2023 MISSING ACCESSION RETRIEVAL")
    print("=" * 72)
    print(f"Input accessions: {len(input_ids):,}")
    print(f"Batch size:       {args.batch_size}")

    mappings: Dict[str, dict] = {}

    checkpoint_path = (
            args.out_dir / "retrieval_checkpoint.json"
    )

    # Resume if an earlier run exists.
    if checkpoint_path.exists():
        with checkpoint_path.open(
                "r",
                encoding="utf-8",
        ) as f:
            old = json.load(f)

        mappings.update(old)

        print(
            f"Resuming with "
            f"{len(mappings):,} already retrieved."
        )

    completed = set(mappings)

    remaining = [
        acc
        for acc in input_ids
        if acc not in completed
    ]

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "GOAligneRR-GOR2023-dataset-preparation/1.0"
        )
    })

    batches = list(
        chunks(remaining, args.batch_size)
    )

    for idx, batch in enumerate(batches, start=1):
        batch_set = set(batch)

        print(
            f"[{idx}/{len(batches)}] "
            f"querying {len(batch):,} accessions..."
        )

        try:
            data = fetch_batch_json(
                session,
                batch,
            )

            batch_map, found = extract_results(
                data,
                batch_set,
            )

            mappings.update(batch_map)

            print(
                f"    found: {len(found):,}/"
                f"{len(batch):,}"
            )

        except Exception as e:
            print(
                f"[ERROR] batch {idx} failed: {e}"
            )
            print(
                "Checkpointing and continuing."
            )

        # Save after every batch.
        save_json(
            mappings,
            checkpoint_path,
        )

        time.sleep(args.sleep)

    all_requested = set(input_ids)
    found_ids = set(mappings)
    not_found = all_requested - found_ids

    fasta_path = (
            args.out_dir
            / "retrieved_current_uniprot.fasta"
    )

    metadata_path = (
            args.out_dir
            / "retrieved_current_uniprot.tsv"
    )

    not_found_path = (
            args.out_dir
            / "not_found.txt"
    )

    write_fasta(
        mappings,
        fasta_path,
    )

    write_metadata(
        mappings,
        metadata_path,
    )

    save_ids(
        not_found,
        not_found_path,
    )

    summary = {
        "requested": len(all_requested),
        "retrieved": len(found_ids),
        "not_found": len(not_found),
        "retrieval_fraction": (
            len(found_ids) / len(all_requested)
            if all_requested
            else 0.0
        ),
    }

    save_json(
        summary,
        args.out_dir / "summary.json",
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Requested : {len(all_requested):,}")
    print(f"Retrieved : {len(found_ids):,}")
    print(f"Not found : {len(not_found):,}")
    print(
        f"Coverage  : "
        f"{100 * summary['retrieval_fraction']:.4f}%"
    )

    print()
    print(f"FASTA    -> {fasta_path}")
    print(f"Metadata -> {metadata_path}")
    print(f"Missing  -> {not_found_path}")


if __name__ == "__main__":
    main()