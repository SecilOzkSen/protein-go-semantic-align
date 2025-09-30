"""
Simple curriculum scheduler for FAISS-based negative mining.

It interpolates mining knobs over time (step/epoch):
- hard_frac: fraction of hard negatives among k_total
- shortlist_M: FAISS shortlist size before filtering
- k_hard: number of negatives finally used
- hier_max_hops_up/down: ancestor/descendant mask radius on GO DAG
- random_k: number of random-easy negatives to add
- use_inbatch_easy: whether to include in-batch easy negatives

Use:
    cfg = CurriculumConfig(total_steps=10000, hard_frac=(0.2, 0.8), ...)
    sched = CurriculumScheduler(cfg)
    params = sched(step)  # dict with current values
"""
from __future__ import annotations
from src.configs.data_classes import CurriculumConfig
from typing import Dict
import math

def _interp_linear(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _interp_cosine(a: float, b: float, t: float) -> float:
    # smooth start/end
    t2 = (1 - math.cos(math.pi * t)) / 2.0
    return a + (b - a) * t2

class CurriculumScheduler:
    """
    Step-based scheduler that interpolates mining hyperparameters.

    Call with the current global step; returns dict:
    {
      "hard_frac", "shortlist_M", "k_hard",
      "hier_max_hops_up", "hier_max_hops_down",
      "random_k", "use_inbatch_easy",
      "mix_sibling_queue", "allow_siblings"
    }
    """
    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self._interp = _interp_cosine if cfg.mode == "cosine" else _interp_linear

    def __call__(self, step: int) -> Dict[str, object]:
        cfg = self.cfg
        if cfg.total_steps <= 0:
            t = 1.0
        else:
            if step < cfg.warmup:
                t = 0.0
            else:
                denom = max(1, cfg.total_steps - cfg.warmup)
                t = min(1.0, (step - cfg.warmup) / denom)

        def I(a, b):  # int field
            v = self._interp(float(a), float(b), t)
            return max(0, int(round(v)))

        def F(a, b):  # float field in [0,1]
            v = self._interp(float(a), float(b), t)
            return float(min(max(v, 0.0), 1.0))

        hard_frac = F(cfg.hard_frac[0], cfg.hard_frac[1])
        shortlist_M = max(1, I(cfg.shortlist_M[0], cfg.shortlist_M[1]))
        k_hard = max(1, I(cfg.k_hard[0], cfg.k_hard[1]))
        max_up = I(cfg.hier_max_hops_up[0], cfg.hier_max_hops_up[1])
        max_dn = I(cfg.hier_max_hops_down[0], cfg.hier_max_hops_down[1])
        rand_k = I(cfg.random_k[0], cfg.random_k[1])
        inbatch_prob = F(cfg.use_inbatch_easy[0], cfg.use_inbatch_easy[1])
        use_inbatch = inbatch_prob > 0.5

        # yeni: siblings toggle
        allow_sib_prob = F(cfg.allow_siblings_prob[0], cfg.allow_siblings_prob[1])
        allow_siblings = allow_sib_prob > 0.5

        mix = dict(cfg.mix_sibling_queue)
        tot = sum(max(0.0, v) for v in mix.values())
        self.mix_sibling_queue = {k: (max(0.0, v) / tot if tot > 0 else 0.0) for k, v in mix.items()}


        # Güvenlik: shortlist M, k_hard'dan küçük olmasın
        shortlist_M = max(shortlist_M, k_hard)

        return dict(
            hard_frac=hard_frac,
            shortlist_M=shortlist_M,
            k_hard=k_hard,
            hier_max_hops_up=max_up,
            hier_max_hops_down=max_dn,
            random_k=rand_k,
            use_inbatch_easy=use_inbatch,
            mix_sibling_queue=mix,
            allow_siblings=allow_siblings,
        )
