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
        - Single-layer (linear) or two-layer MLP (tanh) scoring.
        - Supports attention_mask and optional logit bias for specific token IDs.
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

    def forward(self, H: torch.Tensor,  # [B, L, H]
                attention_mask: torch.Tensor,  # [B, L] (1 = real token, 0 = pad)
                input_ids: torch.Tensor = None,  # [B, L]
                token_weight_map: Optional[Dict[int, float]] = None,  # {token_id: weight (e.g., 0.45)}
                return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.use_mlp:
            x = torch.tanh(self.proj1(H))
            x = self.dropout(x)
            logits = self.proj2(x).squeeze(-1)
        else:
            logits = self.proj(H).squeeze(-1)

        # Optional logit bias using token-specific weights (w in (0, +inf), typical 0.25–1.0)
        if token_weight_map and input_ids is not None:
            # Add log(w) to logits where token matches; w<1 downweights, w>1 upweights
            for tid, w in token_weight_map.items():
                if w <= 0:
                    continue
                bias = math.log(float(w))
                logits = logits + (input_ids == tid).float() * bias

        # Mask out paddings with -inf before softmax
        mask = (attention_mask == 1)
        logits = logits.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(logits, dim=-1)  # [B, L]
        pooled = torch.bmm(attn.unsqueeze(1), H).squeeze(1)  # [B, H]

        if return_attn:
            return pooled, attn
        return pooled


class BioMedBERTEncoder(nn.Module):
    """
        BioMedBERT-based encoder that produces semantic embeddings with learned attention pooling.
        - Inference: encode_texts() -> [N, H]
        - Training: forward() returns pooled embeddings for loss computation
        - Optional LoRA (PEFT): enable_lora=True
    """

    def __init__(self,
                 model_name: str,
                 device: torch.device,
                 max_length: int = 512,
                 use_attention_pool: bool = True,  # Attention pooling head
                 attn_hidden: int = 0,  # 0 = linear scoring; >0 = tanh-MLP hidden size
                 attn_dropout: float = 0.0,
                 special_token_weights: Optional[Dict[str, float]] = None, # Optional token weights to bias attention (e.g., {"[GOPATH]":0.45, "[PATH]":0.45, "[ISA]":0.95, "[PART]":0.85})
                 enable_lora: bool = True,
                 lora_parameters: Optional[LoRAParameters] = None  # LoRA options (optional)
                 ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        # Base model and tokenizer
        self.model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True, trust_remote_code=False, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": list(GO_SPECIAL_TOKENS)})
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Attention pooling head (Optional - nice to use)
        self.use_attention_pool = use_attention_pool
        self.attn_head: Optional[AttnPool] = None
        if self.use_attention_pool:
            self.attn_head = AttnPool(self.model.config.hidden_size, attn_hidden, attn_dropout)
            for p in self.attn_head.parameters():
                p.requires_grad_(True)

        # Optional: add special tokens and keep id->weight map for attention bias
        self._id_weight_map: Dict[int, float] = {}
        if special_token_weights:
            # Ensure tokens exist in the vocab
            self.tokenizer.add_special_tokens({"additional_special_tokens": list(special_token_weights.keys())})
            self.model.resize_token_embeddings(len(self.tokenizer))
            # Build id->weight map
            for tok, w in special_token_weights.items():
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if tid != self.tokenizer.unk_token_id:  # keep even if newly added
                    self._id_weight_map[tid] = float(w)

        # LoRA
        self.enable_lora = enable_lora
        self.lora_cfg = None
        if self.enable_lora:
            self.lora_cfg = LoraConfig(r=lora_parameters.lora_r,
                       lora_alpha=lora_parameters.lora_alpha,
                       lora_dropout=lora_parameters.lora_dropout,
                       target_modules=lora_parameters.target_modules,
                       layers_to_transform=lora_parameters.layers_to_transform,
                       layers_pattern=lora_parameters.layers_pattern,
                       bias=lora_parameters.bias,
                       use_rslora=lora_parameters.use_rslora,
                       task_type=lora_parameters.task_type
                       )
            self.model = get_peft_model(self.model, self.lora_cfg, adapter_name=lora_parameters.adapter_name)
            try:
                self.model.print_trainable_parameters()
            except Exception:
                pass
            assert isinstance(self.model, PeftModel), "[LoRA] get_peft_model failed; adapter not attached."
        self.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """x
            Returns pooled embeddings for a tokenized batch.
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state  # (B, L, H)
        if self.use_attention_pool and self.attn_head is not None:
            return self.attn_head(H=H,
                                  input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_weight_map=self._id_weight_map)
        # fallback: masked mean
        m = attention_mask.unsqueeze(-1).float()
        return (H * m).sum(1) / m.sum(1).clamp(min=1e-6)

    # Inference
    @torch.no_grad()
    def encode_texts(self,
                  go_texts: List[str],
                  batch_size: int = 16,
                  normalize: bool = True,
                  return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
            Encode raw texts into embeddings (optionally also return attention weights per batch).
        """
        self.model.eval()
        embs: List[torch.Tensor] = []
        attn_out: List[torch.Tensor] = []

        for i in range(0, len(go_texts), batch_size):
            batch = go_texts[i: i+batch_size]
            toks = self.tokenizer(batch,
                                  padding=True,
                                  truncation=True,
                                  max_length=self.max_length,
                                  return_tensors="pt").to(self.device)
            H = self.model(**toks).last_hidden_state # [B, L, H]
            if self.use_attention_pool and self.attn_head is not None:
                if return_attn:
                    vec, attn = self.attn_head(H, toks["attention_mask"], toks.get("input_ids"), self._id_weight_map,
                                               return_attn=True)
                    attn_out.append(attn.detach().cpu())
                else:
                    vec = self.attn_head(H, toks["attention_mask"], toks.get("input_ids"), self._id_weight_map)
            if normalize:
                vec = F.normalize(vec, dim=-1)
            embs.append(vec)

        if not embs:
            hidden = self.model.config.hidden_size
            empty = torch.zeros(0, hidden, device=self.device)
            return (empty, []) if return_attn else empty

        emb_cat = torch.cat(embs, dim=0)
        return (emb_cat, attn_out) if return_attn else emb_cat


    def add_special_tokens(self, special_token_weights):
        self.tokenizer.add_special_tokens({"additional_special_tokens": list(special_token_weights.keys())})
        self.model.resize_token_embeddings(len(self.tokenizer))
        for tok, w in special_token_weights.items():
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid != self.tokenizer.unk_token_id:  # keep even if newly added
                self._id_weight_map[tid] = float(w)

    def set_token_weights(self, token_weight_map: Dict[str, float]):
        """
        Update attention bias weights for specific tokens by string.
        """
        for tok, w in token_weight_map.items():
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid != self.tokenizer.unk_token_id:
                self._id_weight_map[tid] = float(w)

    def merge_lora(self):
        """
        Merge LoRA weights into the base model for inference-only deployment.
        (Irreversible; re-load for further LoRA training.)
        """
        if self.enable_lora and isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            self.enable_lora = False

    def param_groups_for_optimizer(
            self,
            lr_emb: float = 1e-3,  # if new tokens were added; set to None to skip
            lr_lora: float = 2e-4,  # LoRA A/B
            wd_lora: float = 0.01,
            lr_attn: float = 2e-4,  # attention pooling head
            wd_attn: float = 0.0,
    ):
        """
        Build parameter groups: word embeddings, LoRA, and attention head.
        Useful for AdamW(..., params=encoder.param_groups_for_optimizer(...)).
        """
        groups = []
        # Word embeddings (often higher LR, no weight decay)
        if lr_emb is not None:
            emb_params = []
            for n, p in self.named_parameters():
                if p.requires_grad and "embeddings.word_embeddings.weight" in n:
                    emb_params.append(p)
            if emb_params:
                groups.append({"params": emb_params, "lr": lr_emb, "weight_decay": 0.0})

        # LoRA params
        if lr_lora is not None:
            lora_params = [p for n, p in self.named_parameters() if p.requires_grad and "lora_" in n]
            if lora_params:
                groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": wd_lora})

        # Attention head params
        if self.use_attention_pool and self.attn_head is not None and lr_attn is not None:
            attn_params = [p for p in self.attn_head.parameters() if p.requires_grad]
            if attn_params:
                groups.append({"params": attn_params, "lr": lr_attn, "weight_decay": wd_attn})

        # (Optional) anything else trainable (normally none if base is frozen with LoRA)
        other = [
            p for n, p in self.named_parameters()
            if p.requires_grad and "lora_" not in n and "embeddings.word_embeddings.weight" not in n
               and (self.attn_head is None or not n.startswith("attn_head."))
        ]
        if other:
            # If you do train any base weights, add them here:
            groups.append({"params": other})

        return groups



