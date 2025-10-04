from __future__ import annotations
from pathlib import Path
import os

def _detect_project_root() -> Path:
    """
    Robust root finder:
    1) Env override: PROJECT_ROOT_DIR
    2) Walk up from CWD to find a marker file/folder ('.git', 'pyproject.toml', 'src')
    3) If this file exists (when imported as module), use its parents
    4) Fallback: CWD
    """
    # 1) explicit override
    env = os.getenv("PROJECT_ROOT_DIR")
    if env:
        return Path(env).expanduser().resolve()

    # 2) search upwards from CWD
    cwd = Path.cwd().resolve()
    markers = {".git", "pyproject.toml", "src"}
    cur = cwd
    for _ in range(8):
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    # 3) module file location (if available)
    try:
        here = Path(__file__).resolve()
        cur = here
        for _ in range(8):
            if any((cur / m).exists() for m in markers):
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
    except NameError:
        pass  # __file__ yoksa (notebook), geç

    # 4) fallback
    return cwd

# ---- Root & data layout ------------------------------------------------------

PLATFORM = os.getenv("PLATFORM", "local")  # local, colab, kaggle, ...
PROJECT_ROOT = _detect_project_root()
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SRC_DIR / "data"             # mevcut yapına uyuyor
RAW_PATH = DATA_DIR / "raw"
PROCESSED_PATH = DATA_DIR / "processed"
TRAINING_READY = DATA_DIR / "training_ready"

# ============ GO TERM RELATED PATHS =============================
GOA_PARSED_FILE = RAW_PATH / "goa_parsed.pkl"
GO_TERMS_PKL = RAW_PATH / "go_basic_obo_terms_v2.pkl"  # with 'part_of'

GO_TERMS_PATH = PROCESSED_PATH / "go_terms"
GO_TERM_COUNTS_TSV = GO_TERMS_PATH / "go_term_frequency.tsv"
GO_TERM_COUNTS_PKL = GO_TERMS_PATH / "go_term_frequency.pkl"

# (Legacy count-based, tutuyoruz)
MIDFREQ_COUNT_GO_TERMS_PKL = PROCESSED_PATH / "count_midfreq_go_terms.pkl"
COMMON_COUNT_GO_TERMS_PKL  = PROCESSED_PATH / "count_common_go_terms.pkl"
RARE_COUNT_GO_TERMS_PKL    = PROCESSED_PATH / "count_rare_go_terms.pkl"

# Processed – analysis files
ANALYSIS = PROCESSED_PATH / "analysis"
ANALYSIS_GENERAL = ANALYSIS / "general"
ANALYSIS_PER_NAMESPACE = ANALYSIS / "per_namespace"
ZS_STRICT_MASK_TSV = ANALYSIS_GENERAL / "zero_shot_protein_mask_report.tsv"
ZS_TERMS_PER_NS_COUNT_TSV = ANALYSIS_GENERAL / "zero_shot_terms_per_namespace.tsv"
FS_TERMS_PER_NS_COUNT_TSV = ANALYSIS_GENERAL / "few_shot_terms_per_namespace.tsv"

# IC-based per namespace
ZS_TERMS_PER_NS_IC_BP_TSV = ANALYSIS_PER_NAMESPACE / "ic_zero_shot_BP.tsv"
ZS_TERMS_PER_NS_IC_CC_TSV = ANALYSIS_PER_NAMESPACE / "ic_zero_shot_CC.tsv"
ZS_TERMS_PER_NS_IC_MF_TSV = ANALYSIS_PER_NAMESPACE / "ic_zero_shot_MF.tsv"
FS_TERMS_PER_NS_IC_BP_TSV = ANALYSIS_PER_NAMESPACE / "ic_few_shot_BP.tsv"
FS_TERMS_PER_NS_IC_CC_TSV = ANALYSIS_PER_NAMESPACE / "ic_few_shot_CC.tsv"
FS_TERMS_PER_NS_IC_MF_TSV = ANALYSIS_PER_NAMESPACE / "ic_few_shot_MF.tsv"

# =============== Training ready =============================
ZERO_SHOT_TERMS_ID_ONLY_JSON       = TRAINING_READY / "go_few_zero_common" / "go_zero_shot_id_only.json"
FEW_SHOT_IC_TERMS_ID_ONLY_JSON     = TRAINING_READY / "go_few_zero_common" / "ic_few_shot_terms_id_only.json"
COMMON_IC_GO_TERMS_ID_ONLY_JSON    = TRAINING_READY / "go_few_zero_common" / "ic_common_terms_id_only.json"

# ---- GO indexes by phase -----------------------------
if PLATFORM == "colab":
    _GO_IDX = lambda p: "content" / "drive" / "MyDrive" / "data" / "training_ready" / "go_indexes" / f"phase{p}"
else:
    _GO_IDX = lambda p: TRAINING_READY / "go_indexes" / f"phase{p}"
GO_INDEX = {
    1: {
        "TEXT_EMB": _GO_IDX(1) / "go_text_embeddings.pt",
        "FAISS_IP":  _GO_IDX(1) / "go_faiss_ip.faiss",
        "META":      _GO_IDX(1) / "go_faiss_ip.meta.json",
    },
    2: {
        "TEXT_EMB": _GO_IDX(2) / "go_text_embeddings.pt",
        "FAISS_IP":  _GO_IDX(2) / "go_faiss_ip.faiss",
        "META":      _GO_IDX(2) / "go_faiss_ip.meta.json",
    },
    3: {
        "TEXT_EMB": _GO_IDX(3) / "go_text_embeddings.pt",
        "FAISS_IP":  _GO_IDX(3) / "go_faiss_ip.faiss",
        "META":      _GO_IDX(3) / "go_faiss_ip.meta.json",
    },
    4: {
        "TEXT_EMB": _GO_IDX(4) / "go_text_embeddings.pt",
        "FAISS_IP":  _GO_IDX(4) / "go_faiss_ip.faiss",
        "META":      _GO_IDX(4) / "go_faiss_ip.meta.json",
    },
}

def go_index_paths(phase: int) -> dict[str, Path]:
    if phase not in GO_INDEX:
        raise ValueError(f"Unknown phase: {phase}. Valid: {sorted(GO_INDEX)}")
    return GO_INDEX[phase]

# GO helpers
GO_PARENTS  = TRAINING_READY / "go_dag" / "go_parents.json"
GO_CHILDREN = TRAINING_READY / "go_dag" / "go_children.json"
GO_ANCESTOR_STOPLIST = TRAINING_READY / "go_dag" / "ancestor_stoplist.txt"

# PROTEINS
PID_TO_POSITIVES    = TRAINING_READY / "proteins" / "pid_to_positives.json"
P_SEQ_LEN_LOOKUP    = TRAINING_READY / "proteins" / "seq_len_lookup.pkl"
PROTEIN_TRAIN_IDS   = TRAINING_READY / "proteins" / "protein_train_ids.txt"
PROTEIN_VAL_IDS     = TRAINING_READY / "proteins" / "protein_val_ids.txt"

# GOOGLE DRIVE (opsiyonel) — env ile override edilebilir
# Colab’da: export ESM3B_DRIVE_DIR="/content/drive/MyDrive/esm3b_embeddings"
_ESM3B_DRIVE = os.getenv("ESM3B_DRIVE_DIR")
if _ESM3B_DRIVE:
    GOOGLE_DRIVE_ESM3B = Path(_ESM3B_DRIVE).expanduser().resolve()
else:
    GOOGLE_DRIVE_ESM3B = TRAINING_READY / "esm3b_embeddings"  # repo içi default

GOOGLE_DRIVE_ESM3B_EMBEDDINGS = GOOGLE_DRIVE_ESM3B / "embeddings"  # symlink edebilirsin
GOOGLE_DRIVE_MANIFEST_CACHE  = TRAINING_READY / "manifest_cache" / "esm_manifest.pkl"

# CONFIG
TRAINING_CONFIG = SRC_DIR / "colab.yaml"

def create_data_folders() -> None:
    for path in [
        RAW_PATH, PROCESSED_PATH, GO_TERMS_PATH,
        ANALYSIS, ANALYSIS_GENERAL, ANALYSIS_PER_NAMESPACE,
        TRAINING_READY,
        GOOGLE_DRIVE_ESM3B, GOOGLE_DRIVE_ESM3B_EMBEDDINGS,
        GOOGLE_DRIVE_MANIFEST_CACHE.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)

create_data_folders()
