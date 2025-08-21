from pathlib import Path

CURRENT_PATH = Path.cwd()
RAW_PATH = CURRENT_PATH.parent / "data" / "raw"
PROCESSED_PATH = CURRENT_PATH.parent / "data" / "processed"

# ============ GO TERM RELATED PATHS =============================

## Raw file paths (Those have been already processed on the data preprocessing step -
            # those are raw only here because those are not processed for training purposes)
GOA_PARSED_FILE = RAW_PATH / "goa_parsed.pkl"
GO_TERMS_PKL = RAW_PATH / "go_basic_obo_terms_v2.pkl" # with "part_of" relationship.

## Processed file paths
GO_TERMS_PATH = PROCESSED_PATH / "go_terms"
GO_TERM_COUNTS_TSV = GO_TERMS_PATH / "go_term_frequency.tsv"
GO_TERM_COUNTS_PKL = GO_TERMS_PATH / "go_term_frequency.pkl"

MIDFREQ_COUNT_GO_TERMS_PKL = PROCESSED_PATH / "count_midfreq_go_terms.pkl" # Count based
COMMON_COUNT_GO_TERMS_PKL = PROCESSED_PATH / "count_common_go_terms.pkl" # Count based
RARE_COUNT_GO_TERMS_PKL = PROCESSED_PATH / "count_rare_go_terms.pkl" # Count based
ZERO_SHOT_IC_TERMS_PKL = PROCESSED_PATH / "ic_zero_shot_terms.pkl" # IC Based
FEW_SHOT_IC_TERMS_PKL = PROCESSED_PATH / "ic_few_shot_terms.pkl" # IC Based
COMMON_IC_GO_TERMS_PKL = PROCESSED_PATH / "ic_common_terms.pkl" # IC Based

#Processed - analysis files
ANALYSIS = PROCESSED_PATH / "analysis"
ANALYSIS_GENERAL = ANALYSIS / "general"
ANALYSIS_PER_NAMESPACE = ANALYSIS / "per_namespace"
ZS_STRICT_MASK_TSV = ANALYSIS_GENERAL / "zero_shot_protein_mask_report.tsv"  # protein based report
##### GO Term analysis per count
ZS_TERMS_PER_NS_COUNT_TSV = ANALYSIS_GENERAL / "zero_shot_terms_per_namespace.tsv"
FS_TERMS_PER_NS_COUNT_TSV = ANALYSIS_GENERAL / "few_shot_terms_per_namespace.tsv"

##### GO Term analysis per IC
ZS_TERMS_PER_NS_IC_BP_TSV = ANALYSIS_PER_NAMESPACE / "ic_zero_shot_BP.tsv"
ZS_TERMS_PER_NS_IC_CC_TSV = ANALYSIS_PER_NAMESPACE / "ic_zero_shot_CC.tsv"
ZS_TERMS_PER_NS_IC_MF_TSV = ANALYSIS_PER_NAMESPACE / "ic_zero_shot_MF.tsv"

FS_TERMS_PER_NS_IC_BP_TSV = ANALYSIS_PER_NAMESPACE / "ic_few_shot_BP.tsv"
FS_TERMS_PER_NS_IC_CC_TSV = ANALYSIS_PER_NAMESPACE / "ic_few_shot_CC.tsv"
FS_TERMS_PER_NS_IC_MF_TSV = ANALYSIS_PER_NAMESPACE / "ic_few_shot_MF.tsv"

def create_data_folders():
    # Create all folders
    for path in [RAW_PATH, PROCESSED_PATH, GO_TERMS_PATH, ANALYSIS, ANALYSIS_GENERAL, ANALYSIS_PER_NAMESPACE]:
        path.mkdir(parents=True, exist_ok=True)

create_data_folders()