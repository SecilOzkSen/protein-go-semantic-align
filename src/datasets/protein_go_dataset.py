import torch
from torch.utils.data import Dataset
import pickle

class ProteinGODataset(Dataset):
    def __init__(self, sequences_file:str, goa_file:str, split:str="train"):
        super().__init__()
        with open(sequences_file, 'rb') as f:
            self.sequences = pickle.load(f)
        with open(goa_file, 'rb') as f:
            self.labels = pickle.load(f)