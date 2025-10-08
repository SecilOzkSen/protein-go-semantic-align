from typing import Dict, List
import torch

class GoTextStore:
    """
    Caches tokenized GO texts in memory for fast batch access.
    Expects an external {go_id: "text"} dict at construction.
    """
    def __init__(self, full_id2text: Dict[int, Dict[int, str]], tokenizer, phase:int=0, max_len: int = 256):
        self.id2tok = {}
        self.full_id2text = full_id2text
        self.id2text = self.full_id2text[phase]
        self.phase = phase
        self.max_len = int(max_len)
        self.tokenizer = tokenizer
        self.tokenize()


    def tokenize(self):
        for gid, text in self.id2text.items():
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            self.id2tok[int(gid)] = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }
        print("Tokenize ended.")

    def update_phase_and_tokenize(self, new_phase:int):
        if new_phase == self.phase:
            return
        self.phase = new_phase
        self.id2text = self.full_id2text[self.phase]
        self.tokenize()

    def has(self, gid: int) -> bool:
        return int(gid) in self.id2tok

    def get(self, gid: int) -> Dict[str, torch.Tensor]:
        return self.id2tok[int(gid)]

    def batch(self, gids: List[int]) -> Dict[str, torch.Tensor]:
        gids = [int(g) for g in gids]
        input_ids = torch.stack([self.id2tok[g]["input_ids"] for g in gids], dim=0)
        attn_mask = torch.stack([self.id2tok[g]["attention_mask"] for g in gids], dim=0)
        return {"input_ids": input_ids, "attention_mask": attn_mask}
