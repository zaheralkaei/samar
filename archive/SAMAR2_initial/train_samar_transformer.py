# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 03:19:31 2025

@author: zaher
"""

# === File: train_samar_transformer.py ===
# Script to train Transformer on VAE latents

import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F
from dataset import SamarLatentDataset, samar_collate_fn
from models.samar_transformer import SamarTransformer
from tokenizer import SamarTokenizer
tokenizer = SamarTokenizer.load("samar_vocab.pkl")

CHECKPOINT_DIR = "./checkpoints"
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "samar_transformer_config.json")
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "samar_transformer.pt")

class SamarTransformerTrainer:
    def __init__(self, model: SamarTransformer, latent_path=None, batch_size=16, lr=3e-4, context_size=256, tokenizer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.context_size = context_size
        self.tokenizer = tokenizer  # save it for use in dataset

        if latent_path:
            self.train_dataset = SamarLatentDataset(latent_path, context_size=context_size, tokenizer=self.tokenizer)
            self.val_dataset = SamarLatentDataset(latent_path, context_size=context_size, tokenizer=self.tokenizer)
        else:
            raise ValueError("latent_path must be provided.")

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=samar_collate_fn)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=samar_collate_fn)

    def forward(self, input_ids, latent=None, description=None):
        return self.model(input_ids, tgt=latent, description=description)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['latent'].to(self.device)
        latent = batch.get('latent', None)
        if latent is not None:
            latent = latent.to(self.device)

        description = batch.get('description')
        if description is not None:
            description = description.to(self.device)

        predicted_latent = self.model(input_ids, tgt=latent, description=description)
        predicted_latent = predicted_latent.permute(1, 0, 2)

        L = min(predicted_latent.size(1), labels.size(1))
        predicted_latent = predicted_latent[:, :L, :]
        labels           = labels[:,           :L, :]

        loss = F.mse_loss(predicted_latent, labels)
        print(f"Training Loss: {loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['latent'].to(self.device)
        latent = batch.get('latent', None)
        if latent is not None:
            latent = latent.to(self.device)

        description = batch.get('description')
        if description is not None:
            description = description.to(self.device)

        predicted_latent = self.model(input_ids, tgt=latent, description=description)
        predicted_latent = predicted_latent.permute(1, 0, 2)

        L = min(predicted_latent.size(1), labels.size(1))
        predicted_latent = predicted_latent[:, :L, :]
        labels           = labels[:,           :L, :]

        loss = F.mse_loss(predicted_latent, labels)
        print(f"Validation Loss: {loss.item():.4f}")
        return loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

    def save_model(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), WEIGHTS_PATH)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(self.model.get_config(), f)
        print(f"Model weights and config saved to {CHECKPOINT_DIR}")

    def train(self, num_epochs=10):
        optimizer = self.configure_optimizers()

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss = self.training_step(batch, batch_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss / len(self.train_dataloader):.4f}")

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_dataloader):
                    loss = self.validation_step(batch, batch_idx)
                    val_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(self.val_dataloader):.4f}")

        self.save_model()


def load_trained_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    model = SamarTransformer(**config)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    latent_path = 'latents/latents.pt'

    model = SamarTransformer(
        d_model=256, n_head=4, num_layers=6, dim_feedforward=512,
        dropout=0.1, vocab_size=1129, latent_dim=128
    )

    trainer = SamarTransformerTrainer(model=model, latent_path=latent_path, tokenizer=tokenizer, batch_size=16, lr=3e-4, context_size=128)
    trainer.train(num_epochs=10)