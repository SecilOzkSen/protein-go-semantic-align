import pickle
from src.configs.paths import GOA_PARSED_FILE
from typing import Dict


if __name__ == '__main__':
    with open(GOA_PARSED_FILE, 'rb') as f:
        pid_to_positives = pickle.load(f)
    with open("/Users/secilsen/PhD/protein_function_dataset/data/processed/sequences/sequences_full.pkl", 'rb') as f:
        sequences = pickle.load(f)
    pids = pid_to_positives.keys()
    seq_len_lookup: Dict[str, int] = {}
    for pid in pids:
        if sequences.get(pid, None):
            seq_len_lookup[pid] = len(sequences[pid])
    with open("seq_len_lookup.pkl", 'wb') as f:
        pickle.dump(seq_len_lookup, f)


