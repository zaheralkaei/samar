# -*- coding: utf-8 -*-
"""
Created on Sat May 10 03:04:51 2025

@author: zaher
"""

# data.py
# Combines dataset.py and precompute_samar_latents.py

import os
import glob
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from core import SAMARInputRepresentation
from vocab_and_tokenizer import SamarTokenizer

tokenizer = SamarTokenizer.load("samar_vocab.pkl")

# === SAMARDataset ===
class SAMARDataset(Dataset):
    def __init__(self, data_dir, context_size=256, max_files=-1, min_chunk_len=8, tokenizer=None):
        self.data_dir = data_dir
        self.context_size = context_size
        self.tokenizer = tokenizer or SamarTokenizer()
        self.min_chunk_len = min_chunk_len

        print(f"Loading XML files from: {data_dir}")
        self.files = sorted(glob.glob(os.path.join(data_dir, "**/*.xml"), recursive=True))
        if max_files > 0:
            self.files = self.files[:max_files]

        print(f"Found {len(self.files)} MusicXML files")
        self.examples = []
        for file in self.files:
            try:
                ir = SAMARInputRepresentation(file)
                events = ir.get_event_sequence()
                description = ir.get_description_tokens()
                token_ids = self.tokenizer.encode(events)
                self.examples.extend([
                    {
                        "tokens": token_ids[i:i + context_size],
                        "file": file,
                        "description": description
                    }
                    for i in range(0, len(token_ids), context_size)
                    if len(token_ids[i:i + context_size]) >= self.min_chunk_len
                ])
            except Exception as e:
                print(f"Failed to process {file}: {e}")

        print(f"Total token chunks prepared: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            'input_ids': torch.tensor(item["tokens"][:-1], dtype=torch.long),
            'labels': torch.tensor(item["tokens"][1:], dtype=torch.long),
            'file': os.path.basename(item["file"]),
            'description': item["description"]
        }

# === SamarLatentDataset ===
class SamarLatentDataset(Dataset):
    def __init__(self, latent_path, context_size=256, tokenizer=None):
        self.context_size = context_size
        self.samples = torch.load(latent_path)
        self.tokenizer = tokenizer or SamarTokenizer.load("samar_vocab.pkl")
        self.samples = [s for s in self.samples if len(s["tokens"]) >= context_size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = item["tokens"][:self.context_size]
        latent = item["latent"]
        return {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(tokens[1:], dtype=torch.long),
            "latent": torch.tensor(latent, dtype=torch.float)
        }

# === Collate Function ===
def samar_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    batch_dict = {'input_ids': input_ids_padded, 'labels': labels_padded}

    if 'latent' in batch[0]:
        latent = [item['latent'] for item in batch]
        latent_padded = pad_sequence(latent, batch_first=True, padding_value=0)
        batch_dict['latent'] = latent_padded

    if 'description' in batch[0]:
        desc_lists = [item['description'] for item in batch]
        desc_ids = [torch.tensor(tokenizer.encode(desc), dtype=torch.long) for desc in desc_lists]
        desc_padded = pad_sequence(desc_ids, batch_first=True, padding_value=tokenizer.get_vocab().pad_id)
        batch_dict['description'] = desc_padded

    return batch_dict

# === DataLoader helper ===
def get_samar_dataloader(data_dir, batch_size=16, context_size=256, max_files=-1, num_workers=0, min_chunk_len=8, drop_last=False):
    dataset = SAMARDataset(data_dir, context_size, max_files, min_chunk_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last, collate_fn=samar_collate_fn)

# === Latent Precomputation Script ===
def precompute_samar_latents(xml_data_dir, vae_model, save_path, batch_size=2, context_size=256):
    dataset = SAMARDataset(xml_data_dir, context_size=context_size, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=samar_collate_fn)

    print(f"Encoding and storing latents from {len(dataset)} sequences...")
    all_latents = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(vae_model.device)
            latents = vae_model.encode_latent(input_ids)
            for i in range(input_ids.size(0)):
                all_latents.append({
                    "tokens": input_ids[i].cpu().tolist(),
                    "latent": latents[i].cpu(),
                    "file": batch["file"][i] if "file" in batch else ""
                })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(all_latents, save_path)
    print(f"✅ Latents saved to: {save_path}")
