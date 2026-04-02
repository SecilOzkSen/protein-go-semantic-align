from __future__ import annotations
from typing import List, Dict, Optional, Union, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel

from src.configs.data_classes import LoRAParameters
from src.configs.parameters import GO_SPECIAL_TOKENS


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
        H: torch.Tensor,                      # [B, L, H]
        attention_mask: torch.Tensor,        # [B, L]
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

        attn = torch.softmax(logits, dim=-1)                 # [B, L]
        pooled = torch.bmm(attn.unsqueeze(1), H).squeeze(1)  # [B, H]

        if return_attn:
            return pooled, attn
        return pooled


class BioMedBERTEncoder(nn.Module):
    """
    Flexible GO text encoder.

    pooling strategy:
        - "attn"
        - "mean"

    output_mode in forward():
        - "pooled"
        - "tokens"
        - "both"

    forward(...):
        pooled -> Tensor [B, H] or Tuple[Tensor[B,H], Tensor[B,L]] if return_attn=True
        tokens -> Tensor [B, L, H]
        both   -> dict with:
                  {
                    "pooled": Tensor[B,H] (or Tuple if return_attn=True is handled internally),
                    "tokens": Tensor[B,L,H],
                    "attention_mask": Tensor[B,L],
                    optionally "attn": Tensor[B,L] or None
                  }

    encode_texts(...):
        pooled -> Tensor [N, H] or (Tensor[N,H], List[attn])
        tokens -> dict with:
                  {
                    "tokens": Tensor[N, Lmax, H],
                    "attention_mask": Tensor[N, Lmax]
                  }
        both   -> dict with:
                  {
                    "pooled": Tensor[N, H],
                    "tokens": Tensor[N, Lmax, H],
                    "attention_mask": Tensor[N, Lmax],
                    optionally "attn": List[Tensor or None]
                  }
    """

    def __init__(
        self,
        model_name: str,
        device: Union[str, torch.device],
        max_length: int = 512,
        attention_pooling_strategy: str = "attn",   # "attn" | "mean"
        attn_hidden: int = 0,
        attn_dropout: float = 0.0,
        special_token_weights: Optional[Dict[str, float]] = None,
        enable_lora: bool = False,
        use_special_tokens: bool = False,
        lora_parameters: Optional[LoRAParameters] = None,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        if attention_pooling_strategy not in {"attn", "mean"}:
            raise ValueError(
                f"Unsupported attention_pooling_strategy: {attention_pooling_strategy}. "
                f"Use 'attn' or 'mean'."
            )

        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_length = max_length
        self.pooling_strategy = attention_pooling_strategy
        self.enable_lora = enable_lora

        self.model = AutoModel.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            use_safetensors=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        old_vocab_size = len(self.tokenizer)
        special_tokens_added = False

        if self.enable_lora and use_special_tokens:
            print("[INFO] LoRA enabled, adding GO special tokens.")
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": list(GO_SPECIAL_TOKENS)}
            )
            special_tokens_added = True

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.attn_head: Optional[AttnPool] = None
        if self.pooling_strategy == "attn":
            print("[INFO] Using attention pooling head with hidden size:", attn_hidden)
            self.attn_head = AttnPool(
                self.model.config.hidden_size,
                attn_hidden,
                attn_dropout,
            )
            for p in self.attn_head.parameters():
                p.requires_grad_(True)

        self._id_weight_map: Dict[int, float] = {}

        if special_token_weights:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": list(special_token_weights.keys())}
            )
            special_tokens_added = True

            for tok, w in special_token_weights.items():
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if tid != self.tokenizer.unk_token_id:
                    self._id_weight_map[tid] = float(w)

        if special_tokens_added:
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.init_relation_token_embeds()
            print("[tok] relation token embeddings initialized")

        print("[INFO] LoRA enabled:", self.enable_lora)
        self.lora_cfg = None

        if self.enable_lora:
            if lora_parameters is None:
                raise ValueError("lora_parameters must be provided when enable_lora=True.")

            self.lora_cfg = LoraConfig(
                r=lora_parameters.lora_r,
                lora_alpha=lora_parameters.lora_alpha,
                lora_dropout=lora_parameters.lora_dropout,
                target_modules=lora_parameters.target_modules,
                layers_to_transform=lora_parameters.layers_to_transform,
                layers_pattern=lora_parameters.layers_pattern,
                bias=lora_parameters.bias,
                use_rslora=lora_parameters.use_rslora,
                task_type=lora_parameters.task_type,
            )
            self.model = get_peft_model(
                self.model,
                self.lora_cfg,
                adapter_name=lora_parameters.adapter_name,
            )

            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            if use_special_tokens and special_tokens_added:
                self._enable_new_token_grad_only(old_vocab_size)

            try:
                self.model.print_trainable_parameters()
            except Exception:
                pass

            assert isinstance(self.model, PeftModel), \
                "[LoRA] get_peft_model failed; adapter not attached."

        self.to(device)

        if use_special_tokens:
            for tok in GO_SPECIAL_TOKENS:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                assert tid != self.tokenizer.unk_token_id, f"{tok} is UNK"

    @staticmethod
    def _masked_mean_pool(H: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mf = attention_mask.unsqueeze(-1).float()
        Hf = H.float()
        pooled = (Hf * mf).sum(dim=1) / mf.sum(dim=1).clamp_min(1.0)
        return pooled

    @staticmethod
    def _pad_token_batches(
        token_batches: List[torch.Tensor],   # list of [Bi, Li, H]
        mask_batches: List[torch.Tensor],    # list of [Bi, Li]
        pad_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(token_batches) == 0:
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
        for tok, m in zip(token_batches, mask_batches):
            bsz, cur_len, _ = tok.shape
            out_tokens[offset:offset + bsz, :cur_len, :] = tok
            out_mask[offset:offset + bsz, :cur_len] = m
            offset += bsz

        return out_tokens, out_mask

    def _pool(
        self,
        H: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if self.pooling_strategy == "attn":
            if self.attn_head is None:
                raise RuntimeError("pooling_strategy='attn' but attn_head is None.")
            return self.attn_head(
                H=H,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_weight_map=self._id_weight_map,
                return_attn=return_attn,
            )

        pooled = self._masked_mean_pool(H, attention_mask)
        if return_attn:
            return pooled, None
        return pooled

    def copy_token_embed(self, new_tok: str, ref_tok: str):
        emb = self.model.get_input_embeddings().weight.data
        new_id = self.tokenizer.convert_tokens_to_ids(new_tok)
        ref_id = self.tokenizer.convert_tokens_to_ids(ref_tok)
        if new_id is None or ref_id is None or new_id < 0 or ref_id < 0:
            raise RuntimeError(f"bad token ids: {new_tok}={new_id}, {ref_tok}={ref_id}")
        emb[new_id].copy_(emb[ref_id])

    def init_relation_token_embeds(self):
        mapping = {
            "[IS_A]": "is",
            "[PART]": "part",
            "[GOPATH]": "relation",
            "[PATH]": "path",
        }
        for new_tok, ref_tok in mapping.items():
            try:
                self.copy_token_embed(new_tok, ref_tok)
            except Exception:
                self.copy_token_embed(new_tok, "the")

    def _enable_new_token_grad_only(self, old_vocab_size: int):
        emb = self.model.base_model.model.embeddings.word_embeddings
        W = emb.weight

        new_vocab = W.shape[0]
        if new_vocab <= old_vocab_size:
            print(f"[DBG] no new tokens to train: old={old_vocab_size} new={new_vocab}")
            return

        W.requires_grad_(True)

        mask = torch.zeros((new_vocab, 1), device=W.device, dtype=W.dtype)
        mask[old_vocab_size:new_vocab] = 1.0

        W.register_hook(lambda g: g * mask.to(device=g.device, dtype=g.dtype))

        print(
            f"[DBG] embedding grad mask enabled. "
            f"old={old_vocab_size}, new={new_vocab}, train_rows={new_vocab - old_vocab_size}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_mode: str = "pooled",   # "pooled" | "tokens" | "both"
        return_attn: bool = False,
    ):
        if output_mode not in {"pooled", "tokens", "both"}:
            raise ValueError(
                f"Unsupported output_mode: {output_mode}. "
                f"Use 'pooled', 'tokens', or 'both'."
            )

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state  # [B, L, H]

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

        # both
        if return_attn:
            pooled_vec, attn = pooled
            return {
                "pooled": pooled_vec,              # [B, H]
                "tokens": H,                       # [B, L, H]
                "attention_mask": attention_mask,  # [B, L]
                "attn": attn,                      # [B, L] or None
            }

        return {
            "pooled": pooled,                     # [B, H]
            "tokens": H,                          # [B, L, H]
            "attention_mask": attention_mask,     # [B, L]
        }

    @torch.no_grad()
    def encode_texts(
        self,
        go_texts: List[str],
        batch_size: int = 16,
        normalize: bool = True,
        return_attn: bool = False,
        output_mode: str = "pooled",   # "pooled" | "tokens" | "both"
    ):
        if output_mode not in {"pooled", "tokens", "both"}:
            raise ValueError(
                f"Unsupported output_mode: {output_mode}. "
                f"Use 'pooled', 'tokens', or 'both'."
            )

        self.model.eval()

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
            H = out.last_hidden_state  # [B,L,H]

            if output_mode == "tokens":
                tok = H
                if normalize:
                    tok = F.normalize(tok, dim=-1)
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
                attn_out.append(attn.detach().cpu() if attn is not None else None)
            else:
                vec = pooled

            if normalize:
                vec = F.normalize(vec, dim=-1)

            pooled_out.append(vec)

            if output_mode == "both":
                tok = H
                if normalize:
                    tok = F.normalize(tok, dim=-1)
                token_out.append(tok)
                mask_out.append(toks["attention_mask"])

        hidden = self.model.config.hidden_size

        if output_mode == "tokens":
            if not token_out:
                empty_tokens = torch.zeros(0, 0, hidden, device=self.device)
                empty_mask = torch.zeros(0, 0, dtype=torch.long, device=self.device)
                return {
                    "tokens": empty_tokens,
                    "attention_mask": empty_mask,
                }

            tokens_cat, masks_cat = self._pad_token_batches(token_out, mask_out)
            return {
                "tokens": tokens_cat,
                "attention_mask": masks_cat,
            }

        if not pooled_out:
            empty = torch.zeros(0, hidden, device=self.device)
            if output_mode == "both":
                empty_tokens = torch.zeros(0, 0, hidden, device=self.device)
                empty_mask = torch.zeros(0, 0, dtype=torch.long, device=self.device)
                result = {
                    "pooled": empty,
                    "tokens": empty_tokens,
                    "attention_mask": empty_mask,
                }
                if return_attn:
                    result["attn"] = []
                return result

            return (empty, []) if return_attn else empty

        pooled_cat = torch.cat(pooled_out, dim=0)

        if output_mode == "pooled":
            return (pooled_cat, attn_out) if return_attn else pooled_cat

        tokens_cat, masks_cat = self._pad_token_batches(token_out, mask_out)
        result = {
            "pooled": pooled_cat,
            "tokens": tokens_cat,
            "attention_mask": masks_cat,
        }
        if return_attn:
            result["attn"] = attn_out
        return result

    def add_special_tokens(self, special_token_weights):
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(special_token_weights.keys())}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        for tok, w in special_token_weights.items():
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid != self.tokenizer.unk_token_id:
                self._id_weight_map[tid] = float(w)

    def set_token_weights(self, token_weight_map: Dict[str, float]):
        for tok, w in token_weight_map.items():
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid != self.tokenizer.unk_token_id:
                self._id_weight_map[tid] = float(w)

    def merge_lora(self):
        if self.enable_lora and isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            self.enable_lora = False

    def param_groups_for_optimizer(
        self,
        lr_emb: float = 1e-3,
        lr_lora: float = 2e-4,
        wd_lora: float = 0.01,
        lr_attn: float = 2e-4,
        wd_attn: float = 0.0,
    ):
        groups = []

        if lr_emb is not None:
            emb_params = []
            for n, p in self.named_parameters():
                if p.requires_grad and "embeddings.word_embeddings.weight" in n:
                    emb_params.append(p)
            if emb_params:
                groups.append({"params": emb_params, "lr": lr_emb, "weight_decay": 0.0})

        if lr_lora is not None:
            lora_params = [p for n, p in self.named_parameters() if p.requires_grad and "lora_" in n]
            if lora_params:
                groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": wd_lora})

        if self.pooling_strategy == "attn" and self.attn_head is not None and lr_attn is not None:
            attn_params = [p for p in self.attn_head.parameters() if p.requires_grad]
            if attn_params:
                groups.append({"params": attn_params, "lr": lr_attn, "weight_decay": wd_attn})

        other = [
            p for n, p in self.named_parameters()
            if p.requires_grad
            and "lora_" not in n
            and "embeddings.word_embeddings.weight" not in n
            and (self.attn_head is None or not n.startswith("attn_head."))
        ]
        if other:
            groups.append({"params": other})

        return groups