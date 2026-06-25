# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 02:41:00 2025

@author: zaher
"""

# === File: train_samar_vae.py ===
# Script to train the VQ-VAE on tokenized chunks

import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from data import get_samar_dataloader
from models.samar_vae import SamarVQVAE
from vocab_and_tokenizer import SamarTokenizer

# CONFIG
DATA_DIR = "./xml_data"
BATCH_SIZE = 64
CONTEXT_SIZE = 64
MAX_EPOCHS = 10
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
LEARNING_RATE = 5e-4
VOCAB_PATH = "samar_vocab.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD TOKENIZER TO GET pad_id AND vocab size
tokenizer = SamarTokenizer.load(VOCAB_PATH)
vocab_size = len(tokenizer.vocab)
pad_id = tokenizer.vocab.pad_id

# DATALOADER
from data import SamarLatentDataset
from torch.utils.data import DataLoader

LATENT_PATH = "latents/test_latents.pt"

train_dataset = SamarLatentDataset(
    latent_path=LATENT_PATH,
    context_size=CONTEXT_SIZE,
    tokenizer=tokenizer
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODEL
model = SamarVQVAE(d_model=128, n_embed=512, vocab_size=vocab_size, lr=LEARNING_RATE, pad_id=pad_id).to(DEVICE)

# OPTIMIZER
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# LOGGING
writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, "samar_vae"))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
best_loss = float("inf")

# TRAIN LOOP
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        print("Input ID max:", input_ids.max().item())
        print("Vocab size:", model.vocab_size)
        print("Input shape:", input_ids.shape)
        loss, acc = model.compute_loss(input_ids, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    avg_loss = epoch_loss / len(train_loader)
    avg_acc = epoch_acc / len(train_loader)
    print(f"Epoch {epoch}/{MAX_EPOCHS} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc:.4f}")

    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", avg_acc, epoch)

    # Save checkpoint if this is the best so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "samar_vae.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model.get_config(),
            "vocab_path": VOCAB_PATH
        }, checkpoint_path)
        print(f"Saved best model to {checkpoint_path}")

    # Optional: log reconstruction
    with torch.no_grad():
        model.eval()
        sample_input = batch['input_ids'][0:1].to(DEVICE)
        logits = model(sample_input)
        preds = logits.argmax(dim=-1)
        print("\nSample reconstruction:")
        print("Input:", tokenizer.decode(sample_input[0].tolist())[:30])
        print("Output:", tokenizer.decode(preds[0].tolist())[:30])

writer.close()
print("VAE training complete.")
