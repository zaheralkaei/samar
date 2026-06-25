# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:28:25 2025

@author: zaher
"""
import os
import torch
from torch.utils.data import Dataset

class SamarDataset(Dataset):
    def __init__(self, ids_folder, max_len=2048):
        self.paths = [os.path.join(ids_folder, f) for f in os.listdir(ids_folder) if f.endswith(".ids")]
        self.max_len = max_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], "r") as f:
            token_ids = list(map(int, f.read().split()))
        x = token_ids[:-1][:self.max_len]
        y = token_ids[1:][:self.max_len]
        return torch.tensor(x), torch.tensor(y)


