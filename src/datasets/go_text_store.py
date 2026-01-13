
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
            lazy: bool = True,  # keep default True so workers don't explode
            chunk_log: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.lazy = bool(lazy)
        self.chunk_log = int(chunk_log)

        self.full_id2text: Dict[int, Dict[int, str]] = {
            int(p): {int(k): (v or "") for k, v in d.items()} for p, d in full_id2text.items()
        }

        self.phase = int(phase)
        if self.phase not in self.full_id2text:
            raise KeyError(f"phase={self.phase} not found in full_id2text keys={list(self.full_id2text.keys())[:5]}...")

        self.id2text = self.full_id2text[self.phase]

        # {go_id: {"input_ids": [L], "attention_mask": [L]}}
        self.id2tok: Dict[int, Dict[str, torch.Tensor]] = {}

        # IMPORTANT: do not eager tokenize here, that is what caused double work.
        # Main process should call materialize_tokens_once() explicitly.
        if not self.lazy:
            # still support old behavior if someone sets lazy=False intentionally
            self._tokenize_all()
        else:
            print("[GoTextStore] lazy mode: will tokenize on demand (workers). Call materialize_tokens_once() on main.")

        # ---- pickle safety (avoid mmap/shm explosions with num_workers>0) ----

    def shuffle(self, seed: int = 42):
        """Shuffle the id2text mapping for current phase."""
        import random
        go_ids = list(self.id2text.keys())
        texts = [self.id2text[g] for g in go_ids]
        rng = random.Random(seed)
        shuffled = texts[:]
        rng.shuffle(shuffled)
        self.id2text = {
            g: shuffled[i] for i, g in enumerate(go_ids)
        }
        self._tokenize_all()

    def __getstate__(self):
        s = self.__dict__.copy()
        # never ship full cache to workers
        s["id2tok"] = {}
        s["lazy"] = True  # force lazy in workers
        return s

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "id2tok" not in self.__dict__ or self.id2tok is None:
            self.id2tok = {}
        # if cache is empty, worker must be lazy
        if len(self.id2tok) == 0:
            self.lazy = True

        # ---- internals ----

    def _encode(self, text: Optional[str]) -> Dict[str, torch.Tensor]:
        txt = text if (text is not None and len(text) > 0) else "[UNK]"
        enc = self.tokenizer(
            txt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0).to(dtype=torch.long),
            "attention_mask": enc["attention_mask"].squeeze(0).to(dtype=torch.long),
        }

    def _tokenize_all(self):
        """Eager, single-thread encode of current phase (kept for backward compat)."""
        self.id2tok.clear()
        total = len(self.id2text)
        for i, (gid, text) in enumerate(self.id2text.items(), 1):
            self.id2tok[int(gid)] = self._encode(text)
            if self.chunk_log and (i % self.chunk_log == 0):
                print(f"[GoTextStore] tokenized {i}/{total}")
        self.lazy = False
        print("[GoTextStore] Tokenize ended (eager).")

    def _ensure_cached(self, gid: int) -> None:
        gid = int(gid)
        if gid not in self.id2tok:
            if gid not in self.id2text:
                raise KeyError(f"GO id {gid} not found in phase {self.phase}.")
            self.id2tok[gid] = self._encode(self.id2text[gid])

        # ---- fast batch pre-tokenization (main process) ----

    @torch.no_grad()
    def batch_tokenize_phase(
            self,
            phase: Optional[int] = None,
            batch_size: int = 512,
            show_progress: bool = True,
            make_live_if_current: bool = True,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Batch pretokenize a phase on MAIN process.

        If make_live_if_current and ph == self.phase:
          - self.id2tok is replaced
          - self.lazy = False

        Returns cache dict for requested phase.
        """
        # Guard rails
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        ph = self.phase if phase is None else int(phase)
        if ph not in self.full_id2text:
            raise KeyError(f"phase={ph} not found in full_id2text")

        id2text = self.full_id2text[ph]
        total = len(id2text)
        if total == 0:
            return {} if ph != self.phase else self.id2tok

        items = list(id2text.items())
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
                padding="max_length",
                return_tensors="pt",
                return_attention_mask=True,
            )

            ids = enc["input_ids"].to(dtype=torch.long)  # [B,L]
            am = enc["attention_mask"].to(dtype=torch.long)  # [B,L]

            for i, g in enumerate(gids):
                out_cache[g] = {"input_ids": ids[i].clone(), "attention_mask": am[i].clone()}

            if self.chunk_log and (((step + 1) * batch_size) % self.chunk_log == 0):
                done = min((step + 1) * batch_size, total)
                print(f"[GoTextStore] tokenized {done}/{total} (phase {ph})")

        if make_live_if_current and ph == self.phase:
            self.id2tok = out_cache
            self.lazy = False

        return out_cache

    @torch.no_grad()
    def materialize_tokens_once(
            self,
            batch_size: int = 512,
            show_progress: bool = True,
    ):
        """
        MAIN PROCESS ONLY: materialize current phase into self.id2tok exactly once.
        This is the path you should use in your training startup.
        """
        self.batch_tokenize_phase(self.phase, batch_size=batch_size, show_progress=show_progress,
                                  make_live_if_current=True)
        print(f"[GoTextStore] materialized phase {self.phase}; lazy=False")

    # ---- public API ----
    def update_phase(self, new_phase: int, materialize: bool = True, batch_size: int = 512, show_progress: bool = True):
        """
        Phase change:
          - clears cache
          - switches id2text view
          - optionally materializes new phase on main
        """
        new_phase = int(new_phase)
        if new_phase == self.phase:
            return

        if new_phase not in self.full_id2text:
            raise KeyError(f"new_phase={new_phase} not found in full_id2text")

        self.phase = new_phase
        self.id2text = self.full_id2text[self.phase]
        self.id2tok.clear()

        # default: deterministic behavior, materialize now on main
        if materialize:
            self.batch_tokenize_phase(self.phase, batch_size=batch_size, show_progress=show_progress,
                                      make_live_if_current=True)
            print(f"[GoTextStore] switched to phase {self.phase} and materialized; lazy=False")
        else:
            # stay lazy
            self.lazy = True
            print(f"[GoTextStore] switched to phase {self.phase} (lazy=True)")

    # Backward-compat names
    def tokenize(self):
        self._tokenize_all()

    def update_phase_and_tokenize(self, new_phase: int):
        # old behavior was ambiguous based on lazy flag
        # new behavior: switch and materialize deterministically (safe default)
        self.update_phase(new_phase, materialize=True)

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

    def set_max_len(self, max_len: int, materialize: bool = False, batch_size: int = 512, show_progress: bool = True):
        """
        Change max_len safely:
          - clears cache
          - optionally materialize current phase immediately
        """
        self.max_len = int(max_len)
        self.id2tok.clear()
        if materialize:
            self.materialize_tokens_once(batch_size=batch_size, show_progress=show_progress)
        else:
            # keep whatever mode you had, but cache is empty now
            if not self.lazy:
                # if previously eager, you probably want deterministic behavior
                self.materialize_tokens_once(batch_size=batch_size, show_progress=show_progress)
