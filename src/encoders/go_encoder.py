from __future__ import annotations
from typing import List, Dict, Optional, Union, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class AttnPool(nn.Module):
    """
    Attention pooling that learns per-token importance and returns a single embedding.
    """

    def __init__(self, hidden_size: int, attn_hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        self.use_mlp = attn_hidden > 0
        if self.use_mlp:
            self.proj1 = nn.Linear(hidden_size, attn_hidden, bias=True)
            self.proj2 = nn.Linear(attn_hidden, 1, bias=False)
        else:
            self.proj = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            H: torch.Tensor,  # [B, L, H]
            attention_mask: torch.Tensor,  # [B, L]
            input_ids: Optional[torch.Tensor] = None,
            token_weight_map: Optional[Dict[int, float]] = None,
            return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.use_mlp:
            x = torch.tanh(self.proj1(H))
            x = self.dropout(x)
            logits = self.proj2(x).squeeze(-1)
        else:
            logits = self.proj(H).squeeze(-1)

        if token_weight_map and input_ids is not None:
            for tid, w in token_weight_map.items():
                if w <= 0:
                    continue
                bias = math.log(float(w))
                logits = logits + (input_ids == tid).float() * bias

        mask = attention_mask == 1
        logits = logits.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(logits, dim=-1)  # [B, L]
        pooled = torch.bmm(attn.unsqueeze(1), H).squeeze(1)  # [B, H]

        if return_attn:
            return pooled, attn
        return pooled


class BioMedBERTEncoder(nn.Module):
    """
    Frozen GO text encoder for Retriever v2.

    Responsibilities:
      - tokenize GO text segments,
      - run BioMedBERT,
      - return pooled and/or token embeddings.

    Deliberately absent:
      - LoRA / PEFT,
      - trainable special-token embeddings,
      - optimizer parameter-group construction,
      - relation-token initialization.

    The retriever freezes this module and learns only downstream GO segment
    mixing plus GO projection.
    """

    def __init__(
            self,
            model_name: str,
            device: Union[str, torch.device],
            max_length: int = 512,
            attention_pooling_strategy: str = "mean",
            attn_hidden: int = 0,
            attn_dropout: float = 0.0,
            gradient_checkpointing: bool = False,
    ):
        super().__init__()

        if attention_pooling_strategy not in {"attn", "mean", "none"}:
            raise ValueError(
                f"Unsupported attention_pooling_strategy: "
                f"{attention_pooling_strategy}. Use 'attn', 'mean', or 'none'."
            )

        if attention_pooling_strategy == "none":
            attention_pooling_strategy = "mean"

        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_length = int(max_length)
        self.pooling_strategy = attention_pooling_strategy

        self.model = AutoModel.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            use_safetensors=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.attn_head: Optional[AttnPool] = None
        if self.pooling_strategy == "attn":
            self.attn_head = AttnPool(
                self.model.config.hidden_size,
                attn_hidden,
                attn_dropout,
            )

        self.to(self.device)

    @staticmethod
    def _masked_mean_pool(
            H: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mf = attention_mask.unsqueeze(-1).float()
        Hf = H.float()
        return (Hf * mf).sum(dim=1) / mf.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _pad_token_batches(
            token_batches: List[torch.Tensor],
            mask_batches: List[torch.Tensor],
            pad_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not token_batches:
            raise ValueError("token_batches is empty.")

        device = token_batches[0].device
        dtype = token_batches[0].dtype
        hidden = token_batches[0].size(-1)
        total_n = sum(x.size(0) for x in token_batches)
        max_len = max(x.size(1) for x in token_batches)

        out_tokens = torch.full(
            (total_n, max_len, hidden),
            fill_value=pad_value,
            device=device,
            dtype=dtype,
        )
        out_mask = torch.zeros(
            (total_n, max_len),
            device=device,
            dtype=mask_batches[0].dtype,
        )

        offset = 0
        for tok, mask in zip(token_batches, mask_batches):
            bsz, cur_len, _ = tok.shape
            out_tokens[offset:offset + bsz, :cur_len] = tok
            out_mask[offset:offset + bsz, :cur_len] = mask
            offset += bsz

        return out_tokens, out_mask

    def _pool(
            self,
            H: torch.Tensor,
            attention_mask: torch.Tensor,
            input_ids: Optional[torch.Tensor] = None,
            return_attn: bool = False,
    ):
        if self.pooling_strategy == "attn":
            if self.attn_head is None:
                raise RuntimeError(
                    "pooling_strategy='attn' but attn_head is missing."
                )
            return self.attn_head(
                H=H,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_weight_map=None,
                return_attn=return_attn,
            )

        pooled = self._masked_mean_pool(H, attention_mask)
        return (pooled, None) if return_attn else pooled

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            output_mode: str = "pooled",
            return_attn: bool = False,
    ):
        if output_mode not in {"pooled", "tokens", "both"}:
            raise ValueError(
                f"Unsupported output_mode: {output_mode}. "
                "Use 'pooled', 'tokens', or 'both'."
            )

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        H = out.last_hidden_state

        if output_mode == "tokens":
            return H

        pooled = self._pool(
            H=H,
            attention_mask=attention_mask,
            input_ids=input_ids,
            return_attn=return_attn,
        )

        if output_mode == "pooled":
            return pooled

        if return_attn:
            pooled_vec, attn = pooled
            return {
                "pooled": pooled_vec,
                "tokens": H,
                "attention_mask": attention_mask,
                "attn": attn,
            }

        return {
            "pooled": pooled,
            "tokens": H,
            "attention_mask": attention_mask,
        }

    @torch.no_grad()
    def encode_texts(
            self,
            go_texts: List[str],
            batch_size: int = 16,
            normalize: bool = False,
            return_attn: bool = False,
            output_mode: str = "pooled",
    ):
        if output_mode not in {"pooled", "tokens", "both"}:
            raise ValueError(
                f"Unsupported output_mode: {output_mode}. "
                "Use 'pooled', 'tokens', or 'both'."
            )

        self.eval()

        pooled_out: List[torch.Tensor] = []
        token_out: List[torch.Tensor] = []
        mask_out: List[torch.Tensor] = []
        attn_out: List[Optional[torch.Tensor]] = []

        for i in range(0, len(go_texts), batch_size):
            batch = go_texts[i:i + batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**toks)
            H = out.last_hidden_state

            if output_mode == "tokens":
                tok = F.normalize(H, dim=-1) if normalize else H
                token_out.append(tok)
                mask_out.append(toks["attention_mask"])
                continue

            pooled = self._pool(
                H=H,
                attention_mask=toks["attention_mask"],
                input_ids=toks.get("input_ids"),
                return_attn=return_attn,
            )

            if return_attn:
                vec, attn = pooled
                attn_out.append(
                    attn.detach().cpu() if attn is not None else None
                )
            else:
                vec = pooled

            if normalize:
                vec = F.normalize(vec, dim=-1)

            pooled_out.append(vec)

            if output_mode == "both":
                tok = F.normalize(H, dim=-1) if normalize else H
                token_out.append(tok)
                mask_out.append(toks["attention_mask"])

        hidden = self.model.config.hidden_size

        if output_mode == "tokens":
            if not token_out:
                return {
                    "tokens": torch.zeros(
                        0, 0, hidden, device=self.device
                    ),
                    "attention_mask": torch.zeros(
                        0, 0, dtype=torch.long, device=self.device
                    ),
                }

            tokens_cat, masks_cat = self._pad_token_batches(
                token_out,
                mask_out,
            )
            return {
                "tokens": tokens_cat,
                "attention_mask": masks_cat,
            }

        if not pooled_out:
            empty = torch.zeros(0, hidden, device=self.device)

            if output_mode == "both":
                result = {
                    "pooled": empty,
                    "tokens": torch.zeros(
                        0, 0, hidden, device=self.device
                    ),
                    "attention_mask": torch.zeros(
                        0, 0, dtype=torch.long, device=self.device
                    ),
                }
                if return_attn:
                    result["attn"] = []
                return result

            return (empty, []) if return_attn else empty

        pooled_cat = torch.cat(pooled_out, dim=0)

        if output_mode == "pooled":
            return (
                (pooled_cat, attn_out)
                if return_attn
                else pooled_cat
            )

        tokens_cat, masks_cat = self._pad_token_batches(
            token_out,
            mask_out,
        )
        result = {
            "pooled": pooled_cat,
            "tokens": tokens_cat,
            "attention_mask": masks_cat,
        }
        if return_attn:
            result["attn"] = attn_out
        return result