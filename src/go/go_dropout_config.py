from dataclasses import dataclass
import torch

@dataclass
class GoDropoutConfig:
    enabled: bool = True
    p: float = 0.08              # 0.05–0.10 iyi aralık
    pad_id: int = 0              # tokenizer pad id
    protect_ids: tuple[int, ...] = ()  # (bos_id, eos_id, cls_id) varsa ekle

class GoTokenDropout:
    def __init__(self, cfg: GoDropoutConfig):
        self.cfg = cfg

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids: (B, L)
        attention_mask: (B, L) 1/0
        """
        cfg = self.cfg
        if not cfg.enabled or cfg.p <= 0:
            return input_ids, attention_mask

        out_ids = input_ids.clone()
        out_attn = attention_mask.clone()

        valid = out_attn.bool()
        drop = (torch.rand_like(out_attn.float()) < cfg.p) & valid

        # special tokenları koru
        if cfg.protect_ids:
            protect = torch.zeros_like(drop)
            for tid in cfg.protect_ids:
                protect |= (out_ids == tid)
            drop &= ~protect

        out_ids[drop] = cfg.pad_id
        out_attn[drop] = 0
        return out_ids, out_attn