
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
            lazy: bool = True,
            chunk_log: int = 0,
            is_segmented: bool = False,
            full_id2segments: Optional[Mapping[int, Mapping[int, Mapping[str, str]]]] = None,
            full_id2seg_present: Optional[Mapping[int, Mapping[int, Mapping[str, bool]]]] = None,
            segment_max_len: int = 64,
            segment_names: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.segment_max_len = int(segment_max_len)
        self.lazy = bool(lazy)
        self.chunk_log = int(chunk_log)
        self.is_go_segmented = bool(is_segmented)

        self.segment_names = segment_names or ["name", "namespace", "definition", "is_a", "part_of"]

        self.full_id2text: Dict[int, Dict[int, str]] = {
            int(p): {int(k): (v or "") for k, v in d.items()} for p, d in full_id2text.items()
        }

        self.phase = int(phase)
        if self.phase not in self.full_id2text:
            raise KeyError(f"phase={self.phase} not found in full_id2text keys={list(self.full_id2text.keys())[:5]}...")

        self.id2text = self.full_id2text[self.phase]
        self.n_go = len(self.id2text)

        # Full text cache
        self.id2tok: Dict[int, Dict[str, torch.Tensor]] = {}

        # Segment cache
        self.full_id2segments = None
        self.full_id2seg_present = None
        self.id2segments = None
        self.id2seg_present = None
        self.id2seg_tok: Dict[int, Dict[str, torch.Tensor]] = {}

        if self.is_go_segmented:
            if full_id2segments is None or full_id2seg_present is None:
                raise ValueError(
                    "is_segmented=True requires full_id2segments and full_id2seg_present."
                )

            self.full_id2segments = {
                int(p): {
                    int(k): {str(sk): (sv or "") for sk, sv in segs.items()}
                    for k, segs in d.items()
                }
                for p, d in full_id2segments.items()
            }

            self.full_id2seg_present = {
                int(p): {
                    int(k): {str(sk): bool(sv) for sk, sv in pres.items()}
                    for k, pres in d.items()
                }
                for p, d in full_id2seg_present.items()
            }

            if self.phase not in self.full_id2segments:
                raise KeyError(f"phase={self.phase} not found in full_id2segments")

            if self.phase not in self.full_id2seg_present:
                raise KeyError(f"phase={self.phase} not found in full_id2seg_present")

            self.id2segments = self.full_id2segments[self.phase]
            self.id2seg_present = self.full_id2seg_present[self.phase]

        if not self.lazy:
            self._tokenize_all()
        else:
            print("[GoTextStore] lazy mode: will tokenize on demand (workers). Call materialize_tokens_once() on main.")

    def shuffle(self, seed: int = 42):
        """Shuffle GO text/segments for negative-control ablation."""
        import random

        go_ids = list(self.id2text.keys())
        rng = random.Random(seed)

        perm = list(range(len(go_ids)))
        rng.shuffle(perm)

        texts = [self.id2text[g] for g in go_ids]

        self.id2text = {
            g: texts[perm[i]] for i, g in enumerate(go_ids)
        }
        self.full_id2text[self.phase] = self.id2text

        if self.is_go_segmented:
            segs = [self.id2segments[g] for g in go_ids]
            pres = [self.id2seg_present[g] for g in go_ids]

            self.id2segments = {
                g: segs[perm[i]] for i, g in enumerate(go_ids)
            }
            self.id2seg_present = {
                g: pres[perm[i]] for i, g in enumerate(go_ids)
            }

            self.full_id2segments[self.phase] = self.id2segments
            self.full_id2seg_present[self.phase] = self.id2seg_present

        self.id2tok.clear()
        self.id2seg_tok.clear()

        self._tokenize_all()

    def __getstate__(self):
        s = self.__dict__.copy()
        s["id2tok"] = {}
        s["id2seg_tok"] = {}
        s["lazy"] = True
        return s

    def __setstate__(self, state):
        self.__dict__.update(state)

        if "id2tok" not in self.__dict__ or self.id2tok is None:
            self.id2tok = {}

        if "id2seg_tok" not in self.__dict__ or self.id2seg_tok is None:
            self.id2seg_tok = {}

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
        """Eager, single-thread encode of current phase."""
        self.id2tok.clear()
        self.id2seg_tok.clear()

        total = len(self.id2text)

        for i, (gid, text) in enumerate(self.id2text.items(), 1):
            gid = int(gid)

            self.id2tok[gid] = self._encode(text)

            if self.is_go_segmented:
                self.id2seg_tok[gid] = self._encode_segments(gid)

            if self.chunk_log and (i % self.chunk_log == 0):
                print(f"[GoTextStore] tokenized {i}/{total}")

        self.lazy = False
        print("[GoTextStore] Tokenize ended (eager).")

    def _encode_segments(self, gid: int) -> Dict[str, torch.Tensor]:
        if not self.is_go_segmented:
            raise RuntimeError(
                "_encode_segments called but is_go_segmented=False"
            )

        gid = int(gid)

        if self.id2segments is None or self.id2seg_present is None:
            raise RuntimeError("Segment maps are missing.")

        if gid not in self.id2segments:
            raise KeyError(
                f"GO id {gid} not found in id2segments "
                f"for phase {self.phase}"
            )

        segs = self.id2segments[gid]
        present_map = self.id2seg_present.get(gid, {})

        S = len(self.segment_names)
        L = self.segment_max_len

        seg_present = torch.zeros(S, dtype=torch.bool)

        present_indices = []
        present_texts = []

        for segment_idx, segment_name in enumerate(self.segment_names):
            txt = segs.get(segment_name, "")
            is_present = bool(
                present_map.get(segment_name, False)
                and isinstance(txt, str)
                and txt.strip()
            )

            if not is_present:
                continue

            seg_present[segment_idx] = True
            present_indices.append(segment_idx)
            present_texts.append(txt.strip())

        if not present_texts:
            raise RuntimeError(
                f"GO id {gid} has zero present GO segments "
                f"for phase {self.phase}. "
                f"Segments: {self.segment_names}"
            )

        enc = self.tokenizer(
            present_texts,
            truncation=True,
            max_length=L,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )

        present_input_ids = enc["input_ids"].to(dtype=torch.long)
        present_attention_mask = enc["attention_mask"].to(
            dtype=torch.long
        )

        if present_input_ids.shape != (len(present_indices), L):
            raise RuntimeError(
                "Unexpected tokenized segment shape: "
                f"{tuple(present_input_ids.shape)}, expected "
                f"{(len(present_indices), L)}"
            )

        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            raise RuntimeError(
                "Tokenizer must define pad_token_id for segmented GO encoding."
            )

        # Fixed [S, L] output is preserved for batching.
        seg_input_ids = torch.full(
            size=(S, L),
            fill_value=int(pad_token_id),
            dtype=torch.long,
        )

        seg_attention_mask = torch.zeros(
            size=(S, L),
            dtype=torch.long,
        )

        index_tensor = torch.tensor(
            present_indices,
            dtype=torch.long,
        )

        seg_input_ids[index_tensor] = present_input_ids
        seg_attention_mask[index_tensor] = present_attention_mask

        return {
            "seg_input_ids": seg_input_ids,
            "seg_attention_mask": seg_attention_mask,
            "seg_present": seg_present,
        }

    def _ensure_cached(self, gid: int) -> None:
        gid = int(gid)

        if gid not in self.id2tok:
            if gid not in self.id2text:
                raise KeyError(f"GO id {gid} not found in phase {self.phase}.")
            self.id2tok[gid] = self._encode(self.id2text[gid])

        if self.is_go_segmented and gid not in self.id2seg_tok:
            self.id2seg_tok[gid] = self._encode_segments(gid)

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
        out_seg_cache: Dict[int, Dict[str, torch.Tensor]] = {}
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
            # new segment cache
            if self.is_go_segmented:
                seg_texts = []
                seg_present_rows = []

                if self.full_id2segments is None or self.full_id2seg_present is None:
                    raise RuntimeError("Segment maps missing during batch_tokenize_phase.")

                id2segments = self.full_id2segments[ph]
                id2seg_present = self.full_id2seg_present[ph]

                for g in gids:
                    segs = id2segments[int(g)]
                    present = id2seg_present[int(g)]

                    pres = []
                    for sname in self.segment_names:
                        txt = segs.get(sname, "")
                        if not txt:
                            txt = {
                                "name": "Name: none.",
                                "namespace": "Namespace: none.",
                                "definition": "Definition: none.",
                                "is_a": "Is-a parents: none.",
                                "part_of": "Part-of parents: none.",
                            }.get(sname, f"{sname}: none.")

                        seg_texts.append(txt)
                        pres.append(bool(present.get(sname, False)))

                    if not any(pres):
                        pres[0] = True

                    seg_present_rows.append(pres)

                seg_enc = self.tokenizer(
                    seg_texts,
                    truncation=True,
                    max_length=self.segment_max_len,
                    padding="max_length",
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                Bc = len(gids)
                S = len(self.segment_names)
                Ls = self.segment_max_len

                seg_ids = seg_enc["input_ids"].to(dtype=torch.long).view(Bc, S, Ls)
                seg_am = seg_enc["attention_mask"].to(dtype=torch.long).view(Bc, S, Ls)
                seg_present = torch.tensor(seg_present_rows, dtype=torch.bool)  # [Bc,S]

                for i, g in enumerate(gids):
                    out_seg_cache[int(g)] = {
                        "seg_input_ids": seg_ids[i].clone(),
                        "seg_attention_mask": seg_am[i].clone(),
                        "seg_present": seg_present[i].clone(),
                    }

            if self.chunk_log and (((step + 1) * batch_size) % self.chunk_log == 0):
                done = min((step + 1) * batch_size, total)
                print(f"[GoTextStore] tokenized {done}/{total} (phase {ph})")

        if make_live_if_current and ph == self.phase:
            self.id2tok = out_cache
            if self.is_go_segmented:
                self.id2seg_tok = out_seg_cache
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
        new_phase = int(new_phase)
        if new_phase == self.phase:
            return

        if new_phase not in self.full_id2text:
            raise KeyError(f"new_phase={new_phase} not found in full_id2text")

        self.phase = new_phase
        self.id2text = self.full_id2text[self.phase]
        self.id2tok.clear()

        if self.is_go_segmented:
            if self.full_id2segments is None or self.full_id2seg_present is None:
                raise RuntimeError("Segment maps missing while switching phase.")
            if new_phase not in self.full_id2segments:
                raise KeyError(f"new_phase={new_phase} not found in full_id2segments")
            if new_phase not in self.full_id2seg_present:
                raise KeyError(f"new_phase={new_phase} not found in full_id2seg_present")

            self.id2segments = self.full_id2segments[self.phase]
            self.id2seg_present = self.full_id2seg_present[self.phase]
            self.id2seg_tok.clear()

        if materialize:
            self.batch_tokenize_phase(
                self.phase,
                batch_size=batch_size,
                show_progress=show_progress,
                make_live_if_current=True,
            )
            print(f"[GoTextStore] switched to phase {self.phase} and materialized; lazy=False")
        else:
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

        if (
                self.lazy
                or gid not in self.id2tok
                or (self.is_go_segmented and gid not in self.id2seg_tok)
        ):
            self._ensure_cached(gid)

        out = {
            "input_ids": self.id2tok[gid]["input_ids"],
            "attention_mask": self.id2tok[gid]["attention_mask"],
        }

        if self.is_go_segmented:
            out.update({
                "seg_input_ids": self.id2seg_tok[gid]["seg_input_ids"],
                "seg_attention_mask": self.id2seg_tok[gid]["seg_attention_mask"],
                "seg_present": self.id2seg_tok[gid]["seg_present"],
            })

        return out

    def batch(self, gids: List[int]) -> Dict[str, torch.Tensor]:
        gids = [int(g) for g in gids]

        # ensure cache exists, both full and segment if needed
        for g in gids:
            if (
                    self.lazy
                    or g not in self.id2tok
                    or (self.is_go_segmented and g not in self.id2seg_tok)
            ):
                self._ensure_cached(g)

        input_ids = torch.stack([self.id2tok[g]["input_ids"] for g in gids], dim=0)
        attn_mask = torch.stack([self.id2tok[g]["attention_mask"] for g in gids], dim=0)

        output = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }

        if self.is_go_segmented:
            seg_input_ids = torch.stack(
                [self.id2seg_tok[g]["seg_input_ids"] for g in gids],
                dim=0,
            )  # [G,S,Ls]

            seg_attention_mask = torch.stack(
                [self.id2seg_tok[g]["seg_attention_mask"] for g in gids],
                dim=0,
            )  # [G,S,Ls]

            seg_present = torch.stack(
                [self.id2seg_tok[g]["seg_present"] for g in gids],
                dim=0,
            )  # [G,S]

            output.update({
                "seg_input_ids": seg_input_ids,
                "seg_attention_mask": seg_attention_mask,
                "seg_present": seg_present,
                "segment_names": self.segment_names,
            })

        return output

    def set_max_len(self, max_len: int, materialize: bool = False, batch_size: int = 512, show_progress: bool = True):
        self.max_len = int(max_len)
        self.id2tok.clear()

        if self.is_go_segmented:
            self.id2seg_tok.clear()

        if materialize:
            self.materialize_tokens_once(batch_size=batch_size, show_progress=show_progress)
        else:
            if not self.lazy:
                self.materialize_tokens_once(batch_size=batch_size, show_progress=show_progress)
