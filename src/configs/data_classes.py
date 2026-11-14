from typing import Tuple, Set, Dict, Optional, Iterable, List, Literal, Any
from dataclasses import dataclass, field
import torch
import numpy as np

@dataclass(frozen=True)
class GOIndex:
    local_go_ids: np.ndarray  # shape: [n_go], global GO id’leri
    global_to_local: Dict[int, int]  # global id -> local index

    @property
    def n_go(self) -> int:
        return int(self.local_go_ids.size)

    @staticmethod
    def from_local_ids(local_go_ids: Iterable[int]) -> "GOIndex":
        arr = np.asarray(list(local_go_ids), dtype=np.int64)
        g2l = {int(g): i for i, g in enumerate(arr)}
        return GOIndex(local_go_ids=arr, global_to_local=g2l)

    def mask_from_globals(self, terms: Set[int]) -> np.ndarray:
        m = np.zeros((self.n_go,), dtype=np.bool_)
        if terms:
            idxs = [self.global_to_local[g] for g in terms if g in self.global_to_local]
            if idxs:
                m[idxs] = True
        return m

    def to_local(self, globals_: Iterable[int]) -> np.ndarray:
        ids = np.fromiter((self.global_to_local.get(int(g), -1) for g in globals_), dtype=np.int64)
        return ids


@dataclass
class FewZeroConfig:
    zero_shot_terms: Set[int]
    few_shot_terms: Set[int]
    min_pos_per_protein: int = 1
    fs_target_ratio: float = 0.30


@dataclass
class LoRAParameters:
    '''
    If you later push r to 32–64, consider use_rslora=True and re-tune lora_alpha (effective scale changes)
    For long GO texts, adapting only the top layers is usually best; widen scope only if metrics stall.
    '''
    adapter_name: Optional[str] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: list([
        "query",
        "value",
        "dense",
        "attention.self.query",
        "attention.self.value",
        "attention.output.dense",
    ]))
    use_rslora: bool = False
    layers_to_transform: Optional[List[int]] = None
    layers_pattern: Optional[str] = None
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "FEATURE_EXTRACTION"


@dataclass
class LoggingConfig:
    log_every: int = 50
    log_lora_hist: bool = False
    probe_eval_every: int = 500
    probe_batch_size: int = 8
    gospec_tau: float = 0.02
    gospec_topk: int = 32


@dataclass
class TrainingContext:
    """Runtime objects and evaluation metadata used by Retriever v2."""

    go_cache: Any
    device: Any = "cpu"
    go_text_store: Any = None
    wandb_run: Any = None
    run_name: Optional[str] = None
    fp16_enabled: bool = True
    logging: Optional[LoggingConfig] = None

    protein_pooling_strategy: str = "mean_attn_gate"
    go_pool_type: str = "mean"
    go_encoder_output_mode: str = "segment_pooled"
    go_segment_representation_mode: str = "segments_only"

    eval_id_list: Optional[List[int]] = None
    eval_seen_go_ids: Optional[List[int]] = None
    eval_unseen_ids: Optional[List[int]] = None
    eval_rare_go_ids: Optional[List[int]] = None
    logger: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "run_name": self.run_name,
            "device": str(self.device),
            "protein_pooling_strategy": self.protein_pooling_strategy,
            "go_pool_type": self.go_pool_type,
            "go_encoder_output_mode": self.go_encoder_output_mode,
            "go_segment_representation_mode": self.go_segment_representation_mode,
        }
        if self.logging is not None:
            d["logging"] = (
                self.logging.__dict__
                if hasattr(self.logging, "__dict__")
                else vars(self.logging)
            )
        return d


@dataclass
class AttrConfig:
    lambda_attr: float = 0.0
    lambda_dag: float = 0.0
    lambda_bce: float = 0.0
    lambda_entropy_alpha: float = 0.05
    lambda_entropy_window: float = 0.01
    topk_per_window: int = 64
    curriculum_epochs: int = 10
    temperature: float = 0.07  # InfoNCE için (DUPLICATE kaldırıldı)
    # teacher loss weight
    lambda_vtrue: float = 0.2
    tau_distill: float = 1.5  # KL temperature for distillation


@dataclass
class TrainerConfig:
    """Configuration for the simplified full-GO Retriever v2 trainer."""

    # Representation dimensions.
    d_h: int = 1024
    d_g: int = 768
    d_z: int = 768

    # Runtime / optimization.
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_epochs: int = 15
    batch_size: int = 4
    eval_batch_size: int = 4
    fp16: bool = True
    warmstart_path: Optional[str] = None

    # Checkpoint selection.
    monitor_metric: str = "oracle_microF@500"
    secondary_monitor_metric: str = "macro_term_recall@500"
    monitor_mode: str = "max"

    # Protein representation: one pooled vector only.
    protein_pooling_strategy: str = "mean_attn_gate"

    # GO representation: frozen text features -> learned segment mix -> projection.
    use_lora: bool = False
    go_pooling: str = "none"
    go_pool_type: str = "mean"
    go_encoder_output_mode: str = "segment_pooled"
    go_segment_representation_mode: str = "segments_only"
    go_segment_alpha: float = 0.05
    go_segment_alpha_warmup_steps: int = 5000
    eval_go_bs: int = 256

    # Similarity / InfoNCE.
    temperature: float = 0.07
    is_logit_scale_constant: bool = True
    lambda_con: float = 1.0

    # Positive Block Rank (PBR).
    # v_ipn = sigmoid((s_in - s_ip + margin) / tau)
    # negatives SUM -> positives MEAN -> proteins MEAN.
    pbr_lambda: float = 0.0
    pbr_margin: float = 0.05
    pbr_tau: float = 0.05

    # Evaluation only. These K values are NOT part of the training objective.
    retrieval_eval_ks: Tuple[int, ...] = (50, 100, 200, 500, 1000)
    positive_rank_quantiles: Tuple[float, ...] = (0.50, 0.75, 0.90, 0.95)

    # Development / W&B diagnostics.
    log_gradient_norms: bool = True
    log_positive_rank_cdf: bool = True
    log_cardinality_metrics: bool = True
    log_go_segment_weights: bool = True