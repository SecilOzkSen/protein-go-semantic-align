"""Select reproducible, illustrative positive protein--GO case-study pairs.

This CPU-only command reads existing RAW B3-FA benchmark outputs. It does
not run a model, alter a checkpoint, infer negatives, or inspect heatmaps.
The automatic sample contains one protein within a successful retrieval
stratum and one different protein within a difficult retrieval stratum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT_GO_IDS = {"GO:0003674", "GO:0008150", "GO:0005575"}
REQUIRED_FILES = ("metrics.json", "scores_raw.npy", "protein_ids.json",
                  "go_ids.json", "evaluation_data.npz")
FORMATS = {"general": "mzsgo_fa_general_eval_v1",
           "temporal": "mzsgo_fa_temporal_eval_v1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_go(value):
    text = str(value).strip()
    text = text[3:] if text.upper().startswith("GO:") else text
    require(bool(re.fullmatch(r"[0-9]{1,7}", text)),
            f"Invalid GO identifier: {value!r}")
    number = int(text)
    require(number > 0, f"Invalid GO identifier: {value!r}")
    return f"GO:{number:07d}"


def _ids(value, label, go=False):
    require(isinstance(value, list) and bool(value), f"Missing or empty {label}")
    require(all(isinstance(item, str) and item for item in value),
            f"{label} must be nonempty strings")
    require(len(value) == len(set(value)), f"Duplicate {label}")
    if go:
        require(all(item == canonical_go(item) for item in value),
                f"{label} must use canonical GO:0000000 identifiers")
    return value


def _np_ids(archive, key):
    require(key in archive, f"Evaluation archive lacks {key}")
    array = archive[key]
    require(array.ndim == 1 and array.dtype.kind in "US", f"Invalid archive {key}")
    return array.tolist()


def load_evaluation(eval_dir):
    """Validate a saved RAW evaluation and return its ordered arrays.

    ``labels`` has its original evaluation scope; ``label_columns`` maps
    those columns to ``scores``. All ranks use the complete score universe.
    """
    directory = Path(eval_dir).expanduser().resolve()
    require(directory.is_dir(), f"Evaluation directory missing: {directory}")
    manifest_path = directory / "evaluation_manifest.json"
    require(manifest_path.is_file(), "Missing RAW evaluation_manifest.json")
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict), "Evaluation manifest must be an object")
    file_hashes = manifest.get("files")
    require(isinstance(file_hashes, dict), "Evaluation manifest lacks file hashes")
    observed = {}

    def verify(filename):
        path = directory / filename
        require(path.is_file(), f"Missing evaluation file: {filename}")
        expected = file_hashes.get(filename)
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                f"Evaluation manifest lacks SHA256 for {filename}")
        actual = file_sha256(path)
        require(actual == expected, f"Evaluation SHA256 mismatch: {filename}")
        observed[filename] = actual

    for name in REQUIRED_FILES:
        verify(name)
    metrics = read_json(directory / "metrics.json")
    require(isinstance(metrics, dict), "Evaluation metrics must be an object")
    mode = metrics.get("mode")
    require(mode in FORMATS and manifest.get("format") == FORMATS[mode],
            "Use the original general or temporal RAW evaluation, not TPR outputs")
    require(metrics.get("model") == "B3-FA", "Case extraction requires B3-FA evaluation")
    require(metrics.get("true_path_applied", False) is False and
            "propagated" not in metrics and "prediction_propagation" not in metrics,
            "True-path propagated scores are not supported; supply RAW outputs")
    if mode == "temporal":
        require(metrics.get("true_path_applied") is False,
                "Temporal metrics must explicitly record no true-path propagation")
    require(metrics.get("score_transform") == "linear", "Unexpected RAW score transform")
    require(metrics.get("full_train_refit") is False,
            "Expected the fixed validation-selected checkpoint without refit")
    branch = metrics.get("branch")
    require(branch in {"MF", "BP", "CC"} and manifest.get("branch") == branch,
            "Evaluation branch mismatch")
    checkpoint = metrics.get("checkpoint")
    checkpoint_sha = metrics.get("checkpoint_sha256")
    require(isinstance(checkpoint, str) and checkpoint, "Evaluation checkpoint path missing")
    require(isinstance(checkpoint_sha, str) and re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha),
            "Evaluation checkpoint SHA256 missing")
    require(manifest.get("checkpoint_sha256") == checkpoint_sha,
            "Evaluation checkpoint SHA256 differs between metrics and manifest")

    pids = _ids(read_json(directory / "protein_ids.json"), "protein IDs")
    gids = _ids(read_json(directory / "go_ids.json"), "GO IDs", go=True)
    scores = np.load(directory / "scores_raw.npy", mmap_mode="r", allow_pickle=False)
    require(scores.ndim == 2 and scores.shape == (len(pids), len(gids)) and
            scores.dtype == np.float32, "Unexpected RAW score shape or dtype")
    require(all(bool(np.isfinite(scores[row]).all()) for row in range(len(pids))),
            "Nonfinite RAW scores")
    for key, expected in (("n_proteins", len(pids)), ("n_candidates", len(gids))):
        require(metrics.get(key) == expected and manifest.get(key) == expected,
                f"Evaluation count mismatch: {key}")

    with np.load(directory / "evaluation_data.npz", allow_pickle=False) as data:
        require(_np_ids(data, "protein_ids") == pids,
                "Evaluation archive protein order differs from JSON IDs")
        if mode == "general":
            require(_np_ids(data, "go_ids") == gids,
                    "Evaluation archive GO order differs from JSON IDs")
            require("labels" in data, "General evaluation archive lacks labels")
            labels = data["labels"]
            columns = np.arange(len(gids), dtype=np.int64)
            label_gids = gids
        else:
            require(_np_ids(data, "full_go_ids") == gids,
                    "Temporal archive full GO order differs from JSON IDs")
            verify("temporal_go_ids.json")
            label_gids = _ids(read_json(directory / "temporal_go_ids.json"),
                             "temporal GO IDs", go=True)
            require(_np_ids(data, "temporal_go_ids") == label_gids,
                    "Temporal GO order differs between archive and JSON IDs")
            require("labels_temporal" in data and "temporal_columns" in data,
                    "Temporal archive lacks labels or column mapping")
            labels, columns = data["labels_temporal"], data["temporal_columns"]
            require(columns.ndim == 1 and columns.dtype.kind in "iu" and
                    len(columns) == len(label_gids), "Invalid temporal column mapping")
            require(len(set(columns.tolist())) == len(columns) and
                    bool(((columns >= 0) & (columns < len(gids))).all()),
                    "Duplicate or out-of-range temporal columns")
            require([gids[int(i)] for i in columns] == label_gids,
                    "Temporal column mapping disagrees with GO identifiers")
            require(metrics.get("n_temporal_go") == len(label_gids) and
                    manifest.get("n_temporal_go") == len(label_gids),
                    "Temporal GO count mismatch")
        require(labels.ndim == 2 and labels.shape == (len(pids), len(columns)),
                "Evaluation label shape mismatch")
        require(labels.dtype.kind in "buif" and bool(np.isfinite(labels).all()) and
                bool(((labels == 0) | (labels == 1)).all()), "Labels must be binary")
        require(bool((labels.sum(axis=1) > 0).all()),
                "Evaluation contains a protein without a scoped positive annotation")
        labels = labels.astype(bool, copy=False)
        columns = columns.astype(np.int64, copy=False)
    return {"eval_dir": directory, "metrics": metrics, "manifest": manifest,
            "scores": scores, "labels": labels, "label_columns": columns,
            "label_go_ids": label_gids, "protein_ids": pids, "go_ids": gids,
            "source_files_sha256": observed,
            "source_manifest_sha256": file_sha256(manifest_path)}


def stable_ranks(scores):
    """One-based ranks, retaining saved GO order for score ties."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _pair(evaluation, row, column, rank, group):
    return {"protein_id": evaluation["protein_ids"][row],
            "go_id": evaluation["go_ids"][column], "protein_row": int(row),
            "go_column": int(column), "label": 1, "rank": int(rank),
            "score_raw": float(evaluation["scores"][row, column]),
            "selection_group": group}


def choose_cases(evaluation, seed=17, success_max_rank=50, failure_min_rank=500,
                 shortlist_per_group=12, manual_cases=None):
    require(type(seed) is int and seed >= 0, "Seed must be a nonnegative integer")
    require(type(success_max_rank) is int and success_max_rank >= 1,
            "Success rank cutoff must be positive")
    require(type(failure_min_rank) is int and failure_min_rank > success_max_rank,
            "Failure rank cutoff must be greater than the success cutoff")
    require(type(shortlist_per_group) is int and shortlist_per_group > 0,
            "Shortlist size must be positive")
    success_group, failure_group = "within_top_k", "at_or_below_failure_rank"
    groups = {success_group: {}, failure_group: {}}
    pids, gids = evaluation["protein_ids"], evaluation["go_ids"]
    columns = evaluation["label_columns"]
    root_columns = {i for i, gid in enumerate(gids) if gid in ROOT_GO_IDS}
    manual_cases = list(manual_cases or [])
    requested = []
    for raw in manual_cases:
        require(isinstance(raw, str) and "," in raw,
                "Each --case must be PROTEIN_ID,GO:0000000")
        protein, go = raw.rsplit(",", 1)
        requested.append((protein.strip(), canonical_go(go)))
    require(len(requested) == len(set(requested)), "Duplicate manual case pair")
    pid_index, gid_index = {p: i for i, p in enumerate(pids)}, {g: i for i, g in enumerate(gids)}
    scoped_column = {int(column): j for j, column in enumerate(columns)}
    for protein, go in requested:
        require(protein in pid_index and go in gid_index,
                f"Manual case is outside the evaluation universe: {protein}, {go}")
        row, column = pid_index[protein], gid_index[go]
        require(column in scoped_column and evaluation["labels"][row, scoped_column[column]],
                f"Manual case is not a recorded scoped positive: {protein}, {go}")
        require(go not in ROOT_GO_IDS, "Ontology roots are excluded from case studies")

    manual_lookup = {(pid_index[p], gid_index[g]): index for index, (p, g) in enumerate(requested)}
    manual_output = [None] * len(requested)
    eligible_pairs, excluded_roots = 0, 0
    for row in range(len(pids)):
        positive = columns[np.flatnonzero(evaluation["labels"][row])]
        ranks = stable_ranks(evaluation["scores"][row])
        for column in positive:
            column = int(column)
            if column in root_columns:
                excluded_roots += 1
                continue
            eligible_pairs += 1
            rank = int(ranks[column])
            if rank <= success_max_rank:
                groups[success_group].setdefault(row, []).append((column, rank))
            if rank >= failure_min_rank:
                groups[failure_group].setdefault(row, []).append((column, rank))
            if (row, column) in manual_lookup:
                manual_output[manual_lookup[(row, column)]] = _pair(
                    evaluation, row, column, rank, "manual")

    rng = np.random.default_rng(seed)
    ordered_pairs = {}
    for group, per_protein in groups.items():
        rows = sorted(per_protein)
        rows = rng.permutation(np.asarray(rows, dtype=np.int64)).tolist()
        items = []
        for row in rows:
            possibilities = per_protein[row]
            column, rank = possibilities[int(rng.integers(len(possibilities)))]
            items.append(_pair(evaluation, row, column, rank, group))
        ordered_pairs[group] = items

    available = {}
    selected, chosen_proteins = [], set()
    shortlist = []
    for group, items in ordered_pairs.items():
        candidates = items if requested else [p for p in items if p["protein_id"] not in chosen_proteins]
        shortlist.extend(candidates[:shortlist_per_group])
        if candidates:
            available[group] = {"eligible_proteins": len(groups[group]), "status": "available"}
            if not requested:
                selected.append(dict(candidates[0]))
                chosen_proteins.add(candidates[0]["protein_id"])
        else:
            reason = "empty_rank_stratum" if not items else "no_distinct_protein_remaining"
            available[group] = {"eligible_proteins": len(groups[group]),
                                "status": "unavailable", "reason": reason}
    if requested:
        require(all(item is not None for item in manual_output), "A manual case could not be recovered")
        selected = manual_output
    for index, case in enumerate(selected, 1):
        case["case_id"] = f"case_{index:02d}"
    selected_keys = {(c["protein_id"], c["go_id"]): c["case_id"] for c in selected}
    for item in shortlist:
        item["case_id"] = selected_keys.get((item["protein_id"], item["go_id"]), "")
        item["selected"] = bool(item["case_id"])
    existing_shortlist = {(p["protein_id"], p["go_id"]) for p in shortlist}
    for item in selected:
        if (item["protein_id"], item["go_id"]) not in existing_shortlist:
            shortlist.append({**item, "selected": True})

    policy = {
        "method": "manual_recorded_positives" if requested else "protein_stratified_seeded_v1",
        "seed": seed, "success_max_rank": success_max_rank,
        "failure_min_rank": failure_min_rank, "shortlist_per_group": shortlist_per_group,
        "rank_universe": "complete_saved_RAW_score_universe",
        "tie_rule": "stable_saved_GO_order", "excluded_go_ids": sorted(ROOT_GO_IDS),
        "sampling_unit": "protein, then one uniformly sampled eligible positive GO per protein",
        "default_distinct_proteins": True, "heatmaps_inspected_for_selection": False,
        "annotation_region_screening": "not_performed",
        "purpose": "illustrative successful and difficult retrieval cases, not a representative performance estimate",
    }
    counts = {"n_proteins": len(pids), "n_candidates": len(gids),
              "scoped_positive_pairs": int(evaluation["labels"].sum()),
              "eligible_positive_pairs": eligible_pairs, "excluded_root_positive_pairs": excluded_roots,
              "selected_cases": len(selected),
              "stratum_positive_pairs": {group: sum(len(values) for values in items.values())
                                         for group, items in groups.items()}}
    return selected, shortlist, policy, counts, available


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--success-max-rank", type=int, default=50)
    parser.add_argument("--failure-min-rank", type=int, default=500)
    parser.add_argument("--shortlist-per-group", type=int, default=12)
    parser.add_argument("--case", action="append", default=[], dest="manual_cases",
                        metavar="PROTEIN_ID,GO:0000000",
                        help="Explicit recorded-positive pair, repeatable; overrides automatic pair selection")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).expanduser().resolve()
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())),
            f"Output directory must be new or empty: {out}")
    evaluation = load_evaluation(args.eval_dir)
    selected, shortlist, policy, counts, availability = choose_cases(
        evaluation, args.seed, args.success_max_rank, args.failure_min_rank,
        args.shortlist_per_group, args.manual_cases)
    metrics = evaluation["metrics"]
    result = {"format": "mzsgo_fa_case_selection_v1", "eval_dir": str(evaluation["eval_dir"]),
              "mode": metrics["mode"], "branch": metrics["branch"],
              "checkpoint": metrics["checkpoint"], "checkpoint_sha256": metrics["checkpoint_sha256"],
              "source_files_sha256": evaluation["source_files_sha256"],
              "source_manifest_sha256": evaluation["source_manifest_sha256"],
              "selection_policy": policy, "counts": counts, "availability": availability,
              "cases": selected,
              "note": "All selected pairs are observed positive annotations. Unannotated GO terms are not labelled biological negatives. Successful and difficult describe full-universe retrieval rank only; biological region annotations are still to be checked."}
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "Output directory became nonempty during selection")
    (out / "selection.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    fields = ["case_id", "selected", "protein_id", "go_id", "rank", "score_raw",
              "selection_group", "protein_row", "go_column", "label"]
    with (out / "shortlist.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(shortlist)
    print(json.dumps({"selected_cases": selected, "availability": availability,
                      "selection": str(out / "selection.json"),
                      "shortlist": str(out / "shortlist.csv")}, indent=2))
    return result


if __name__ == "__main__":
    main()
