import json
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from src.configs.paths import PID_TO_POSITIVES, PROTEIN_VAL_IDS, PROTEIN_TRAIN_IDS


def split_train_val_from_pid2pos(
        val_frac: float = 0.1,
        random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    with open(PID_TO_POSITIVES, "r", encoding="utf-8") as f:
        p2p = json.load(f)

    ids = list(p2p.keys())

    train_ids, val_ids = train_test_split(
        ids, test_size=val_frac, random_state=random_state
    )
    return list(train_ids), list(val_ids)


if __name__ == '__main__':
    train_ids, val_ids = split_train_val_from_pid2pos()

    PROTEIN_TRAIN_IDS.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    PROTEIN_VAL_IDS.write_text("\n".join(val_ids) + "\n", encoding="utf-8")
