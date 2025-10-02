# Few shot - zero shot filter parameters
# --- Sterilized go term few zero common choice parameters ---
RARE_THRESHOLD = 20
MID_FREQ_THRESHOLD = 100
EPSILON = 1.0               # Laplace smoothing for p(t)
GAMMA_IC_Q75 = 0.75         # IC quantile split
GAMMA_IC_Q90 = 0.90
K0_MAX_TRAIN = 30           # Max threshold for ZS (IC low)
FS_UPPER_TRAIN = 50         # Upper threshold for FS (IC moderate-high)
FS_LOWER_TRAIN = 5          # FS lower threshold (IC moderate-low)
COMMON_MIN_TRAIN = MID_FREQ_THRESHOLD  # common min threshold (IC high)

# GO Encoder parameters
GO_SPECIAL_TOKENS = ["[GOPATH]", "[PATH]", "[ISA]", "[PART]"]
# phase -> pooling & weights for GO term embeddings
ALLOWED_RELS_FOR_DAG = ["is_a", "part_of"]

