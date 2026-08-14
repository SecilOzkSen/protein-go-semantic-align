from typing import Tuple, Set, Dict, Optional, Iterable, List, Literal, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import torch
import numpy as np

from src.configs.paths import (
    ZERO_SHOT_TERMS_ID_ONLY_JSON,
    FEW_SHOT_IC_TERMS_ID_ONLY_JSON,
    COMMON_IC_GO_TERMS_ID_ONLY_JSON,
    GO_INDEX,                 # {1: {"TEXT_EMB": Path, "FAISS_IP": Path, "META": Path}, ...}
    go_index_paths,           # def go_index_paths(phase:int)->dict[str, Path]
)

@dataclass
class TrainSchedule:
    """
    Phase scheduling and batch mix ratios.
    phase_breaks ex: (5, 12, 25) => [0..4]=phase0, [5..11]=phase1, [12..24]=phase2, [25..]=phase3
    """
    phase_breaks: Tuple[int, int, int] = (5, 12, 25)

    go_cache_paths: List[Path] = field(
        default_factory=lambda: [GO_INDEX[p]["TEXT_EMB"] for p in (1, 2, 3, 4)]
    )
    faiss_paths: List[Path] = field(
        default_factory=lambda: [GO_INDEX[p]["FAISS_IP"] for p in (1, 2, 3, 4)]
    )

    total_phase: int = 4

    # ---- Batch mix ratios per training stage ----
    stageA_mix: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # SCB
    stageB_mix: Tuple[float, float, float] = (0.7, 0.3, 0.0)  # HAB
    stageC_mix: Tuple[float, float, float] = (0.4, 0.4, 0.2)  # EFB
    stageD_mix: Tuple[float, float, float] = (0.4, 0.4, 0.2)  # EFB

    # ---- Attribution Loss ramp ----
    lambda_attr_start: int = 6
    lambda_attr_max: float = 0.2

    # Internal cache for derived #phases
    _n_phases: int = field(default=4, init=False)

    def __post_init__(self):
        b = [int(x) for x in self.phase_breaks if x is not None]
        if len(b) == 0:
            self._n_phases = 1
        elif len(b) == 1:
            self._n_phases = 2
        elif len(b) == 2:
            self._n_phases = 3
        else:
            self._n_phases = 4

    # ---------- helpers ----------
    def phase_for_epoch(self, epoch: int) -> int:
        """Return 0-based phase index for given epoch."""
        e = int(epoch)
        b = list(self.phase_breaks)
        if e < b[0]: return 0
        if e < b[1]: return 1
        if e < b[2]: return 2
        return 3

    def mix_for_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Return (contrastive, dag, attribution) mix for given epoch based on stage mapping."""
        p = self.phase_for_epoch(epoch)
        if p == 0: return self.stageA_mix
        if p == 1: return self.stageB_mix
        if p == 2: return self.stageC_mix
        return self.stageD_mix

    def lambda_attr_for_epoch(self, epoch: int, base_lambda_attr: float) -> float:
        """Ramp attribution weight from 0 to lambda_attr_max starting after lambda_attr_start."""
        if epoch < self.lambda_attr_start:
            return 0.0
        span = max(1, 5)
        t = max(0, epoch - self.lambda_attr_start + 1)
        ramp = min(1.0, t / span) * self.lambda_attr_max
        return min(ramp, base_lambda_attr if base_lambda_attr is not None else self.lambda_attr_max)

    def resolve_go_cache_path(self, phase: int) -> Path:
        if phase >= 0:
            p1 = int(phase) + 1
        else:
            p1 = phase
        return go_index_paths(p1)["TEXT_EMB"]

    def resolve_faiss_path(self, phase: int) -> Optional[Path]:
        p1 = int(phase) + 1
        return go_index_paths(p1)["FAISS_IP"]


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
    faiss_index: Any
    go_cache: Any
    dag_parents: Optional[dict] = None
    dag_children: Optional[dict] = None
    go_namespace_map: Optional[dict] = None
    scheduler: Any = None
    device: Any = "cpu"
    schedule: Any = None
    vres: Any = None
    maybe_refresh_phase_resources: Optional[Callable[[int], None]] = None
    wandb_run: Any = None
    current_phase: Optional[int] = None
    last_refresh_epoch: Any = None
    last_refresh_reason: Any = None
    batch_builder: Any = None
    go_text_store: Any = None
    run_name: str = None
    fp16_enabled: bool = True
    logging: LoggingConfig = None
    use_queue_miner: bool = True
    attribute_loss_enabled: bool = False
    return_alpha: bool = False
    return_slot_attn: bool = False
    fused_bank: Any = None
    protein_pooling_strategy: str = "mean"
    protein_n_slots: int = 0,
    go_pool_type: str = "mean"
    go_encoder_output_mode: str = "pool"
    go_segment_representation_mode: str = "segments_only" # mixed
    eval_id_list: List[int] = None
    logger: Any = None
    eval_seen_go_ids: List[int] = None
    eval_unseen_ids: List[int] = None
    eval_rare_go_ids: List[int] = None


    def to_dict(self):
        d = dict(
            run_name=self.run_name,
            device=self.device,
        )
        # make sure logging is included:
        d["logging"] = self.logging.__dict__ if hasattr(self.logging, "__dict__") else vars(self.logging)
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
    temperature: float = 0.07          # InfoNCE için (DUPLICATE kaldırıldı)
    # teacher loss weight
    lambda_vtrue: float = 0.2
    tau_distill: float = 1.5           # KL temperature for distillation


@dataclass
class TrainerConfig:
    d_h: int = 1024
    d_g: int = 768
    d_z: int = 512
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    lr: float = 2e-4
    lr_lora: float = 5e-5
    use_lora: float = True
    lora_warmup_steps: int = 0
    max_epochs: int = 20,
    cand_chunk_k: int = 8
    pos_chunk_t: int = 128
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    batch_size: int = 64
    eval_batch_size: int = 128
    fp16: bool = True
    monitor_metric: str = "cafa_fmax"
    monitor_mode: str = "max",
    is_logit_scale_constant: bool = False
    go_pooling: str = "masked_mean"  # cls, mean, masked_mean
    eval_go_bs: int = 256
    max_inbatch: int = 64
    eval_cand_chunk_k: int = 64
    warmstart_path: str = None
    go_segment_alpha: float = 0.0
    go_segment_alpha_warmup_steps: int = 10000
    trainable_mode: str = "none"
    local_window_size: int = 64
    local_window_stride: int = 32
    lambda_slot_div: float = 0.0
    multivec_slot_lse_tau: float = 0.00
    multivec_global_residual_init: float = 0.0
    pairwise_margin: float = 0.0
    pairwise_start_step: int = 0
    pairwise_lambda: float = 0.0
    protein_expert_mode: str = "legacy"
    local_slot_aggregation: str = "lse"
    local_slot_lse_tau: float = 0.10
    expert_global_weight: float = 0.50
    expert_fusion_learnable: bool = True
    cardinality_weighting: bool = False
    coverage_lambda: float = 0.0
    coverage_bottom_frac: float = 0.25
    coverage_hard_neg_k: int = 4
    coverage_margin: float = 0.05
    coverage_start_step: int = 2000

@dataclass
class QueueConfig:
    queue_K: int = 65536
    queue_start_step:int = 0
    queue_hard_frac_start: float = 0.0
    queue_hard_frac_end: float = 0.0
    queue_hard_frac_warmup_steps: int = 2000
    queue_weight_start: float = 0.0
    queue_weight_end: float = 0.0
    queue_weight_warmup_steps: int = 2000
    k_hard_queue_start: int = 0
    k_hard_queue_end: int = 0
    k_hard_queue_warmup_steps: int = 2000