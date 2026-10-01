"""PyTorch Dataset for SMS keyword extraction (token classification)."""
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SMSKeywordDataset(Dataset):
    def __init__(self, data_path: Path):
        with open(data_path, "rb") as f:
            self.samples = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }
