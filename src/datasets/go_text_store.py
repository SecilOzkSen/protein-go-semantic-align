# go_text_store.py

from typing import Dict, List, Mapping, Optional, Iterable
import os
import math
import torch

class GoTextStore:
    """
    Caches tokenized GO texts in memory for fast batch access.
    Expects full_id2text as {phase: {go_id: "text"}}.
    Backward-compatible with your previous usage.

    Args:
      full_id2text: {phase: {go_id(int): text(str)}}
      tokenizer: HF tokenizer (callable)
      phase: active phase
      max_len: tokenizer max_length
      lazy: if True, tokenize on demand per id
      chunk_log: if >0, prints progress every N items during eager/batch tokenize

    New:
      - materialize_tokens_once(): pre-tokenize on main process (batching)
      - batch_tokenize_phase(): internal helper for fast pre-tokenization
      - set_max_len(): change max_len safely (clears cache)
    """

    def __init__(
        self,
        full_id2text: Mapping[int, Mapping[int, str]],
        tokenizer,
        phase: int = 0,
        max_len: int = 256,
        lazy: bool = False,
        chunk_log: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.lazy = bool(lazy)
        self.chunk_log = int(chunk_log)

        # Keep keys as int to avoid mismatches; allow None->"[UNK]"
        self.full_id2text: Dict[int, Dict[int, str]] = {
            int(p): {int(k): (v or "") for k, v in d.items()} for p, d in full_id2text.items()
        }

        # Current phase/text view
        self.phase = int(phase)
        self.id2text = self.full_id2text[self.phase]

        # Token cache: {go_id: {"input_ids": T[L], "attention_mask": T[L]}}
        self.id2tok: Dict[int, Dict[str, torch.Tensor]] = {}

        # Eager or lazy
        if not self.lazy:
            self._tokenize_all()
        else:
            print("[GoTextStore] lazy mode: will tokenize on demand.")

    # ---- pickle safety (avoid mmap/shm explosions with num_workers>0) ----
    def __getstate__(self):
        s = self.__dict__.copy()
        # Drop big tensor cache before forking; workers rebuild lazily if needed
        s['id2tok'] = {}
        return s

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'id2tok' not in self.__dict__ or self.id2tok is None:
            self.id2tok = {}

    # ---- internals ----
    def _encode(self, text: Optional[str]) -> Dict[str, torch.Tensor]:
        txt = text if (text is not None and len(text) > 0) else "[UNK]"
        enc = self.tokenizer(
            txt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",       # batch() kendi stack eder; boylar eşit
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def _tokenize_all(self):
        """Eager, single-thread (safe) encode of current phase."""
        self.id2tok.clear()
        total = len(self.id2text)
        for i, (gid, text) in enumerate(self.id2text.items(), 1):
            self.id2tok[int(gid)] = self._encode(text)
            if self.chunk_log and (i % self.chunk_log == 0):
                print(f"[GoTextStore] tokenized {i}/{total}")
        print("[GoTextStore] Tokenize ended (eager).")

    def _ensure_cached(self, gid: int) -> None:
        gid = int(gid)
        if gid not in self.id2tok:
            if gid not in self.id2text:
                raise KeyError(f"GO id {gid} not found in phase {self.phase}.")
            self.id2tok[gid] = self._encode(self.id2text[gid])

    # ---- fast batch pre-tokenization (recommended on Colab) ----
    @torch.no_grad()
    def batch_tokenize_phase(
        self,
        phase: Optional[int] = None,
        batch_size: int = 512,
        show_progress: bool = True,
    ):
        """
        Batch-pretokenize a given phase (or current phase) on the main process.
        Safer than lazy tokenization with multiple DataLoader workers.

        Stores into self.id2tok if phase == self.phase.
        For other phases, returns a dict you can stash or ignore.
        """
        # Guard rails to avoid worker crashes in Colab
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        ph = self.phase if phase is None else int(phase)
        id2text = self.full_id2text[ph]
        total = len(id2text)
        if total == 0:
            return {} if ph != self.phase else self.id2tok

        items = list(id2text.items())  # [(gid, text), ...]
        steps = math.ceil(total / batch_size)
        rng: Iterable[int] = range(steps)

        if show_progress:
            try:
                from tqdm import tqdm
                rng = tqdm(rng, desc=f"[GoTextStore] pre-tokenizing phase {ph}", ncols=80)
            except Exception:
                pass

        out_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        for step in rng:
            s = step * batch_size
            e = min(total, s + batch_size)
            chunk = items[s:e]
            gids = [int(g) for g, _ in chunk]
            texts = [(t if (t and len(t) > 0) else "[UNK]") for _, t in chunk]

            enc = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",  # fixed length → easy stack
                return_tensors="pt",
                return_attention_mask=True,
            )
            ids = enc["input_ids"]           # [B, L]
            am  = enc["attention_mask"]      # [B, L]

            for i, g in enumerate(gids):
                out_cache[g] = {
                    "input_ids": ids[i].clone().to(dtype=torch.long),
                    "attention_mask": am[i].clone().to(dtype=torch.long),
                }

            if self.chunk_log and (((step + 1) * batch_size) % self.chunk_log == 0):
                done = min((step + 1) * batch_size, total)
                print(f"[GoTextStore] tokenized {done}/{total} (phase {ph})")

        if ph == self.phase:
            # make the cache live for current phase
            self.id2tok = out_cache
            self.lazy = False
        return out_cache

    @torch.no_grad()
    def materialize_tokens_once(
        self,
        phases: Optional[List[int]] = None,
        batch_size: int = 512,
        show_progress: bool = True,
    ):
        """
        One-shot pre-tokenization entry point.
        - If phases=None: pretokenize ONLY current phase into self.id2tok (fast path).
        - If phases=[...]: pretokenize listed phases; current phase goes to self.id2tok.
        """
        if phases is None:
            self.batch_tokenize_phase(self.phase, batch_size=batch_size, show_progress=show_progress)
            print("[GoTextStore] materialized current phase; lazy=False")
            return

        # explicit list of phases
        for ph in phases:
            cache = self.batch_tokenize_phase(ph, batch_size=batch_size, show_progress=show_progress)
            if int(ph) == self.phase:
                self.id2tok = cache
        self.lazy = False
        print(f"[GoTextStore] materialized phases {phases}; active phase={self.phase}")

    # ---- public API (kept compatible) ----
    def tokenize(self):
        """Backwards-compat alias: eager tokenize current phase."""
        self._tokenize_all()

    def update_phase_and_tokenize(self, new_phase: int):
        if int(new_phase) == self.phase:
            return
        self.phase = int(new_phase)
        self.id2text = self.full_id2text[self.phase]
        # Clear previous phase cache to save RAM
        self.id2tok.clear()
        if not self.lazy:
            self._tokenize_all()
        else:
            print(f"[GoTextStore] switched to phase {self.phase} (lazy mode).")

    def has(self, gid: int) -> bool:
        gid = int(gid)
        return (gid in self.id2tok) or (gid in self.id2text)

    def get(self, gid: int) -> Dict[str, torch.Tensor]:
        gid = int(gid)
        if self.lazy and gid not in self.id2tok:
            self._ensure_cached(gid)
        return self.id2tok[gid]

    def batch(self, gids: List[int]) -> Dict[str, torch.Tensor]:
        gids = [int(g) for g in gids]
        if self.lazy:
            for g in gids:
                self._ensure_cached(g)
        input_ids = torch.stack([self.id2tok[g]["input_ids"] for g in gids], dim=0)
        attn_mask = torch.stack([self.id2tok[g]["attention_mask"] for g in gids], dim=0)
        return {"input_ids": input_ids, "attention_mask": attn_mask}

    # ---- utilities ----
    def set_max_len(self, max_len: int):
        """Change max_len safely (clears cache; respects lazy/eager mode)."""
        self.max_len = int(max_len)
        self.id2tok.clear()
        if not self.lazy:
            self._tokenize_all()
