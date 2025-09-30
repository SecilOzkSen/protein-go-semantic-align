# Few shot - zero shot filter parameters
# --- Sterilized go term few zero common choice parameters ---
RARE_THRESHOLD = 20
MID_FREQ_THRESHOLD = 100
EPSILON = 1.0               # Laplace smoothing for p(t)
GAMMA_IC_Q75 = 0.75         # IC quantile split
GAMMA_IC_Q90 = 0.90
K0_MAX_TRAIN = 30           # ZS için eğitimde max destek eşiği
FS_UPPER_TRAIN = 50         # FS için üst sınır (IC orta-yüksek ise)
FS_LOWER_TRAIN = 5          # FS için alt sınır
COMMON_MIN_TRAIN = MID_FREQ_THRESHOLD  # common tabanı (true-path sayımlarıyla)

# GO Encoder parameters
GO_SPECIAL_TOKENS = ["[GOPATH]", "[PATH]", "[ISA]", "[PART]"]
# phase -> pooling & weights (ve tokenizer tarafı için max_len önerisi)
ALLOWED_RELS_FOR_DAG = ["is_a", "part_of"]

