"""
Analyse Gene Ontology (GO) term frequencies with true-path propagation
and compute Information Content (IC) for FEW-SHOT / COMMON stratification.

Overview
--------
This script prepares frequency and IC statistics for GO terms based on
the training dataset, following the "true-path rule". Each protein is
annotated not only with its directly assigned GO terms but also with all
ancestor terms (via `is_a` relations) in the ontology DAG. This ensures
that counts reflect biologically valid hierarchical propagation.

IMPORTANT: Zero-shot terms are defined externally (ontology minus training set)
and are NOT computed here anymore.

Steps
-----
1. Load inputs:
   - Protein → GO mapping (GOA, strong-evidence filtered).
   - GO ontology dictionary with term metadata and parents.

2. Build ancestor mapping:
   - Recursively compute all parent terms for each GO term (is_a only).
   - Use memoization for efficiency.

3. Apply true-path propagation:
   - For each protein, expand its GO annotations to include all ancestors.
   - Count how many proteins support each GO term.

4. Compute Information Content (IC):
   - For each GO term `t`, calculate empirical probability:
         p(t) = (n_t + ε) / (N + ε * |V|)
     where n_t = propagated count, N = total proteins,
     |V| = number of GO terms, ε = Laplace smoothing.
   - IC(t) = -log(p(t)), natural log.

5. Stratify terms into FEW-SHOT and COMMON (IC- and frequency-based):
   - FEW-SHOT:
       (FS_LOWER_TRAIN ≤ support ≤ RARE_THRESHOLD)
       OR (IC in [q75, q90) AND support ≤ FS_UPPER_TRAIN)
   - COMMON:
       support ≥ COMMON_MIN_TRAIN
   (Zero-shot is handled in a separate step outside this script.)

6. Save results:
   - Full IC + frequency table (`.tsv`).
   - Count dictionary (`.pkl`).
   - Legacy frequency bins (rare/mid/common) as pickled files.
   - IC-based FEW-SHOT / COMMON sets (pickled) and per-namespace TSVs.

Outputs
-------
- A tab-separated file containing GO_ID, true-path counts, p(t), IC, namespace, and name.
- Pickled Python objects for frequency bands and counts.
- Pickled sets for FEW-SHOT and COMMON (IC-based).
- Per-namespace TSV lists for FEW-SHOT terms.

Notes
-----
- True-path propagation uses only `is_a` relations (consistent with many pipelines).
- Laplace smoothing avoids undefined log values for zero-count terms.
- Frequency bands and IC-based cutoffs can be adjusted via config thresholds.
"""

import pickle
from collections import Counter
import math
import pandas as pd
from functools import lru_cache

from src.config.paths import (
    GOA_PARSED_FILE,          # {protein_id -> [go_ids]}  (TRAIN)
    GO_TERMS_PKL,             # {go_id -> {"name":..., "namespace":..., "is_a":[...], ...}}
    GO_TERM_COUNTS_PKL,       # output: true-path counts (Counter)
    GO_TERM_COUNTS_TSV,       # output: IC table (TSV)
    RARE_COUNT_GO_TERMS_PKL,  # legacy: rare set (by count)
    MIDFREQ_COUNT_GO_TERMS_PKL,
    COMMON_COUNT_GO_TERMS_PKL,
    FEW_SHOT_IC_TERMS_PKL,    # output: FEW-SHOT (IC/freq-based)
    COMMON_IC_GO_TERMS_PKL,   # output: COMMON (IC/freq-based)
    FS_TERMS_PER_NS_IC_BP_TSV,
    FS_TERMS_PER_NS_IC_CC_TSV,
    FS_TERMS_PER_NS_IC_MF_TSV,
)

from src.config.parameters import (
    EPSILON,
    GAMMA_IC_Q75,
    GAMMA_IC_Q90,
    FS_LOWER_TRAIN,       # e.g., 5
    FS_UPPER_TRAIN,       # e.g., 50
    RARE_THRESHOLD,       # e.g., 20
    MID_FREQ_THRESHOLD,   # e.g., 100
    COMMON_MIN_TRAIN,     # e.g., 100 (typically same as MID_FREQ_THRESHOLD)
)


# ==================== HELPERS ==========================

def build_parents_is_a(go_terms: dict) -> dict:
    """
    Extract only `is_a` parent relationships from the GO dictionary.

    Parameters
    ----------
    go_terms : dict
        Mapping {go_id: {"is_a": [...], "name": str, "namespace": str, ...}}

    Returns
    -------
    dict
        Mapping {go_id: set(parent_go_ids)}
    """
    parents = {}
    for gid, meta in go_terms.items():
        parents[gid] = set(meta.get("is_a", []) or [])
    return parents


def make_ancestors_fn_is_a(go_terms: dict):
    """Return a cached function `ancestors_of(node)` that yields all `is_a` ancestors."""
    parents = build_parents_is_a(go_terms)

    @lru_cache(maxsize=None)
    def ancestors_of(node: str) -> set:
        """Recursively collect all ancestors of a node by following its `is_a` parents (cached)."""
        acc = set()
        for p in parents.get(node, set()):
            acc.add(p)
            acc |= ancestors_of(p)
        return acc

    return ancestors_of


# =================== Main pipeline ===================

def analyse_go_term_ic_and_frequency():
    # ---- Load inputs ----
    with open(GOA_PARSED_FILE, 'rb') as f:
        protein2go = pickle.load(f)
    with open(GO_TERMS_PKL, 'rb') as f:
        go_terms = pickle.load(f)

    total_proteins = len(protein2go)
    all_terms = set(go_terms.keys())

    # ---- True-path propagation (`is_a` only) ----
    ancestors_of = make_ancestors_fn_is_a(go_terms)

    tp_counts = Counter()
    for go_list in protein2go.values():
        expanded = set()
        for g in go_list:
            if g in all_terms:
                expanded.add(g)
                expanded |= ancestors_of(g)
        tp_counts.update(expanded)

    # ---- IC calculation ----
    denom = total_proteins + EPSILON * max(1, len(all_terms))
    rows = []
    for g in all_terms:
        n_t = tp_counts.get(g, 0)
        p_t = (n_t + EPSILON) / denom
        ic = -math.log(p_t)
        ns = go_terms[g].get("namespace", "")
        name = go_terms[g].get("name", "")
        rows.append((g, n_t, p_t, ic, ns, name))

    df = pd.DataFrame(rows, columns=["GO_ID", "TP_Protein_Count", "p_t", "IC", "Namespace", "Name"])

    # ---- Namespace-based quantiles ----
    df_bp = df[df["Namespace"] == "biological_process"].copy()
    df_cc = df[df["Namespace"] == "cellular_component"].copy()
    df_mf = df[df["Namespace"] == "molecular_function"].copy()

    def ns_quantiles(sub_df: pd.DataFrame):
        if len(sub_df) == 0:
            return (None, None)
        return (
            sub_df["IC"].quantile(GAMMA_IC_Q75),
            sub_df["IC"].quantile(GAMMA_IC_Q90),
        )

    bp_q75, bp_q90 = ns_quantiles(df_bp)
    cc_q75, cc_q90 = ns_quantiles(df_cc)
    mf_q75, mf_q90 = ns_quantiles(df_mf)

    # ---- FEW-SHOT / COMMON splits (IC- & support-based) ----
    def split_by_ns(sub_df: pd.DataFrame, ns_q75, ns_q90):
        few_shot = set()
        common = set()
        for _, r in sub_df.iterrows():
            gid = r["GO_ID"]
            n = r["TP_Protein_Count"]
            ic = r["IC"]
            # Few-shot:
            #   (5–20 instances) OR (IC in [q75, q90) with support ≤ FS_UPPER_TRAIN)
            if (FS_LOWER_TRAIN <= n <= RARE_THRESHOLD) or (
                ns_q75 is not None and ns_q90 is not None and (ns_q75 <= ic < ns_q90) and (n <= FS_UPPER_TRAIN)
            ):
                few_shot.add(gid)
                continue
            # Common: high support (>= COMMON_MIN_TRAIN)
            if n >= COMMON_MIN_TRAIN:
                common.add(gid)
        return few_shot, common

    few_shot_bp, common_bp = split_by_ns(df_bp, bp_q75, bp_q90)
    few_shot_cc, common_cc = split_by_ns(df_cc, cc_q75, cc_q90)
    few_shot_mf, common_mf = split_by_ns(df_mf, mf_q75, mf_q90)

    FEW_SHOT = few_shot_bp | few_shot_cc | few_shot_mf
    COMMON = common_bp | common_cc | common_mf

    # ---- Legacy frequency buckets (true-path counts only) ----
    rare_terms = set(df[df["TP_Protein_Count"] < RARE_THRESHOLD]["GO_ID"])
    midfreq_terms = set(
        df[(df["TP_Protein_Count"] >= RARE_THRESHOLD) & (df["TP_Protein_Count"] < MID_FREQ_THRESHOLD)]["GO_ID"]
    )
    common_terms_by_count = set(df[df["TP_Protein_Count"] >= MID_FREQ_THRESHOLD]["GO_ID"])

    # ---- Save all results ----
    # Count table
    df.sort_values(by="TP_Protein_Count", ascending=False, inplace=True)
    df.to_csv(GO_TERM_COUNTS_TSV, sep="\t", index=False)
    with open(GO_TERM_COUNTS_PKL, "wb") as f:
        pickle.dump(tp_counts, f)

    # Legacy frequency pickles
    with open(RARE_COUNT_GO_TERMS_PKL, "wb") as f:
        pickle.dump(rare_terms, f)
    with open(MIDFREQ_COUNT_GO_TERMS_PKL, "wb") as f:
        pickle.dump(midfreq_terms, f)
    with open(COMMON_COUNT_GO_TERMS_PKL, "wb") as f:
        pickle.dump(common_terms_by_count, f)

    # IC-based outputs (FEW-SHOT / COMMON only)
    with open(FEW_SHOT_IC_TERMS_PKL, "wb") as f:
        pickle.dump(FEW_SHOT, f)
    with open(COMMON_IC_GO_TERMS_PKL, "wb") as f:
        pickle.dump(COMMON, f)

    # Per-namespace FEW-SHOT reports
    pd.DataFrame({"GO_ID": sorted(few_shot_bp)}).to_csv(FS_TERMS_PER_NS_IC_BP_TSV, sep="\t", index=False)
    pd.DataFrame({"GO_ID": sorted(few_shot_cc)}).to_csv(FS_TERMS_PER_NS_IC_CC_TSV, sep="\t", index=False)
    pd.DataFrame({"GO_ID": sorted(few_shot_mf)}).to_csv(FS_TERMS_PER_NS_IC_MF_TSV, sep="\t", index=False)

    # ---- Console summary ----
    print("=== TRAIN SUMMARY ===")
    print(f"Total proteins (train): {total_proteins}")
    print(f"Total GO terms: {len(all_terms)}")
    print(f"Saved TP+IC table to: {GO_TERM_COUNTS_TSV.name}")

    def ns_print(name, subdf, q75, q90):
        print(f"\n[{name}] terms: {len(subdf)}")
        if len(subdf) > 0:
            print(f"  IC q75={q75:.3f}  q90={q90:.3f}")
            print("  Top-5 most frequent (by TP_Protein_Count):")
            print(subdf.sort_values('TP_Protein_Count', ascending=False)[
                ['GO_ID', 'TP_Protein_Count', 'IC', 'Name']
            ].head(5).to_string(index=False))
            print("  Top-5 highest IC (rarest):")
            print(subdf.sort_values('IC', ascending=False)[
                ['GO_ID', 'TP_Protein_Count', 'IC', 'Name']
            ].head(5).to_string(index=False))

    ns_print("BP", df_bp, bp_q75 or 0.0, bp_q90 or 0.0)
    ns_print("CC", df_cc, cc_q75 or 0.0, cc_q90 or 0.0)
    ns_print("MF", df_mf, mf_q75 or 0.0, mf_q90 or 0.0)

    print("\n=== CLASS SPLITS (IC-based, zero-shot external) ===")
    print(f"FEW_SHOT  total: {len(FEW_SHOT)} (BP={len(few_shot_bp)}, CC={len(few_shot_cc)}, MF={len(few_shot_mf)})")
    print(f"COMMON    total: {len(COMMON)} (BP={len(common_bp)}, CC={len(common_cc)}, MF={len(common_mf)})")

    print("\n=== LEGACY FREQUENCY BUCKETS (true-path counts) ===")
    print(f"Rare   (<{RARE_THRESHOLD}): {len(rare_terms)}")
    print(f"Mid    ([{RARE_THRESHOLD}–{MID_FREQ_THRESHOLD - 1}]): {len(midfreq_terms)}")
    print(f"Common (≥{MID_FREQ_THRESHOLD}): {len(common_terms_by_count)}")

    # Dump head for quick peek + a helper TSV
    print("\nHEAD of IC table:")
    df.to_csv("all.csv", sep="\t", index=False)
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    analyse_go_term_ic_and_frequency()
