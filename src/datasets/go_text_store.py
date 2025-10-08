from typing import Dict, List, Mapping, Optional
import torch

class GoTextStore:
    """
    Caches tokenized GO texts in memory for fast batch access.
    Expects full_id2text as {phase: {go_id: "text"}}.
    Backward-compatible with your previous usage.

    New:
      - lazy: if True, tokenize on demand (prevents long startup + huge pickles)
      - chunk_log: if >0, prints progress every N items during eager tokenize
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
        # Ensure dict exists even if pickled without it
        if 'id2tok' not in self.__dict__ or self.id2tok is None:
            self.id2tok = {}

    # ---- internals ----
    def _encode(self, text: Optional[str]) -> Dict[str, torch.Tensor]:
        txt = text if (text is not None and len(text) > 0) else "[UNK]"
        enc = self.tokenizer(
            txt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def _tokenize_all(self):
        self.id2tok.clear()
        total = len(self.id2text)
        for i, (gid, text) in enumerate(self.id2text.items(), 1):
            self.id2tok[int(gid)] = self._encode(text)
            if self.chunk_log and (i % self.chunk_log == 0):
                print(f"[GoTextStore] tokenized {i}/{total}")
        print("Tokenize ended.")

    def _ensure_cached(self, gid: int) -> None:
        gid = int(gid)
        if gid not in self.id2tok:
            if gid not in self.id2text:
                raise KeyError(f"GO id {gid} not found in phase {self.phase}.")
            self.id2tok[gid] = self._encode(self.id2text[gid])

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
