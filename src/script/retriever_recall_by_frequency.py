#!/usr/bin/env python3
"""Audit top-K retrieval recall by GO-term training frequency."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def go_int(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    if s.startswith("GO:"):
        s = s[3:]
    return int(s)


def load_truths(dump_dir):
    p = Path(dump_dir)
    jp = p / "true_go_ids.json"
    if jp.exists():
        with open(jp, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            rows = list(obj.values())
        else:
            rows = obj
        return [[go_int(x) for x in row if go_int(x) >= 0] for row in rows]
    arr = np.load(p / "true_go_ids.npy", mmap_mode="r")
    return [[int(x) for x in row if int(x) >= 0] for row in arr]


def load_retrieved(dump_dir, topk):
    p = Path(dump_dir)
    direct = p / "top_go_ids.int64.npy"
    if direct.exists():
        arr = np.load(direct, mmap_mode="r")[:, :topk]
        return [set(int(x) for x in row if int(x) >= 0) for row in arr]
    eval_ids = np.load(p / "eval_go_ids.npy", mmap_mode="r")
    cols = np.load(p / "top_go_cols.int32.npy", mmap_mode="r")[:, :topk]
    return [set(int(eval_ids[int(c)]) for c in row if int(c) >= 0) for row in cols]


def bin_name(freq):
    if freq == 0: return "unseen_0"
    if freq <= 5: return "1_5"
    if freq <= 10: return "6_10"
    if freq <= 20: return "11_20"
    if freq <= 50: return "21_50"
    if freq <= 100: return "51_100"
    if freq <= 500: return "101_500"
    return "501_plus"


ORDER = ["unseen_0", "1_5", "6_10", "11_20", "21_50", "51_100", "101_500", "501_plus"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dump", required=True)
    ap.add_argument("--eval_dump", required=True)
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    train_truth = load_truths(args.train_dump)
    eval_truth = load_truths(args.eval_dump)
    retrieved = load_retrieved(args.eval_dump, args.topk)
    if len(eval_truth) != len(retrieved):
        raise ValueError(f"row mismatch: truth={len(eval_truth)} retrieved={len(retrieved)}")

    train_freq = Counter(x for row in train_truth for x in set(row))
    true_by_term = Counter()
    hit_by_term = Counter()
    for truth, pred in zip(eval_truth, retrieved):
        for term in set(truth):
            true_by_term[term] += 1
            if term in pred:
                hit_by_term[term] += 1

    terms_by_bin = defaultdict(list)
    for term in true_by_term:
        terms_by_bin[bin_name(train_freq.get(term, 0))].append(term)

    result = {
        "train_proteins": len(train_truth), "eval_proteins": len(eval_truth), "topk": args.topk,
        "bins": {}
    }
    print(f"train proteins={len(train_truth)} eval proteins={len(eval_truth)} topK={args.topk}")
    print(f"{'bin':<12} {'terms':>7} {'truth':>10} {'hits':>10} {'microR':>9} {'macroR':>9}")
    for b in ORDER:
        terms = terms_by_bin.get(b, [])
        truth = sum(true_by_term[t] for t in terms)
        hits = sum(hit_by_term[t] for t in terms)
        micro = hits / truth if truth else float("nan")
        macro = (sum(hit_by_term[t] / true_by_term[t] for t in terms) / len(terms)) if terms else float("nan")
        result["bins"][b] = {
            "n_terms": len(terms), "truth_occurrences": truth, "retrieved_occurrences": hits,
            "micro_recall": micro, "macro_term_recall": macro,
        }
        print(f"{b:<12} {len(terms):>7} {truth:>10} {hits:>10} {micro:>9.4f} {macro:>9.4f}")

    all_truth = sum(true_by_term.values());
    all_hits = sum(hit_by_term.values())
    result["overall_micro_recall"] = all_hits / all_truth
    result["overall_macro_term_recall"] = sum(hit_by_term[t] / true_by_term[t] for t in true_by_term) / len(true_by_term)
    print(f"\noverall micro recall: {result['overall_micro_recall']:.4f}")
    print(f"overall macro term recall: {result['overall_macro_term_recall']:.4f}")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=True)
        print("saved:", args.out_json)


if __name__ == "__main__":
    main()
