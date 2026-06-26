# -*- coding: utf-8 -*-
"""
Training script for the autoregressive Transformer decoder.

The transformer takes (latent, description, events) as input and
predicts the next event token. See ``docs/audit-round-3.md`` for the
audit history of this file.
"""

import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from .dataset import SamarLatentDataset, samar_collate_fn
from .models.samar_transformer import SamarTransformer
from .tokenizer import SamarTokenizer
from .constants import AUDIT_TRAINER_VERSION  # noqa: F401  (bump on train-loop changes)

# Resolve paths relative to this file so ``python -m
# samar.train_samar_transformer`` works from any cwd.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)

# === Default configuration ===
# These values are read by ``load_trained_transformer`` /
# ``SamarTransformer.from_pretrained`` and by ``generating.py``.
# Update the live ``SamarVocab`` size before changing these.
DEFAULT_VOCAB_SIZE = 1254  # round-2 vocab extension: was 1129. round-4: +5 instruments (Harp, Drumset, etc.)
DEFAULT_D_MODEL = 256
DEFAULT_N_HEAD = 4
DEFAULT_NUM_LAYERS = 6
DEFAULT_DIM_FEEDFORWARD = 512
DEFAULT_DROPOUT = 0.1
DEFAULT_LATENT_DIM = 128

CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "checkpoints")
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "samar_transformer.pt")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "samar_transformer_config.json")
VOCAB_PATH = os.path.join(_PACKAGE_DIR, "samar_vocab.pkl")


def _load_tokenizer():
    """Load the singleton tokenizer (path resolved relative to the package)."""
    return SamarTokenizer.load(VOCAB_PATH)


class SamarTransformerTrainer:
    """Autoregressive transformer trainer over precomputed latents.

    Loads ``latents.pt`` (precomputed by
    ``python -m samar.precompute_samar_latents``) and trains the
    transformer to map (latent, description, events) -> next-event.

    Parameters
    ----------
    model : SamarTransformer
    latent_path : str
        Path to the precomputed ``latents.pt``.
    batch_size : int
    lr : float
    context_size : int
        Fixed length of input event sequences. Shorter samples are
        padded, longer samples are truncated.
    val_fraction : float
        Fraction of the dataset held out for validation. Default 0.1.
        Set to 0 to keep round-1 behavior (train/val use same data,
        NOT RECOMMENDED).
    gradient_clip : float or None
        Max norm for gradient clipping (default 1.0). Set to None to
        disable.
    warmup_steps : int
        Linear LR warmup steps (default 1000). Set to 0 to disable.
    tokenizer : SamarTokenizer or None
    """

    def __init__(
        self,
        model: SamarTransformer,
        latent_path=None,
        batch_size=16,
        lr=3e-4,
        context_size=256,
        val_fraction=0.1,
        gradient_clip=1.0,
        warmup_steps=1000,
        tokenizer=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.context_size = context_size
        self.gradient_clip = gradient_clip
        self.warmup_steps = warmup_steps
self.best_val = float("inf")  # round-5: track best val_loss for checkpointing
        self.tokenizer = tokenizer or _load_tokenizer()

        if not latent_path:
            raise ValueError("latent_path must be provided.")

        full = SamarLatentDataset(latent_path, context_size=context_size,
                                  tokenizer=self.tokenizer)
        if len(full) == 0:
            raise RuntimeError(
                f"SamarLatentDataset has 0 samples after filtering (latents in "
                f"{latent_path} are all shorter than context_size={context_size}). "
                "Run `python -m samar.precompute_samar_latents` to regenerate them."
            )

        # === Train / val split (round-3 audit T1) ===
        if val_fraction <= 0:
            print("[trainer] WARNING: val_fraction=0, using same data for train and val")
            self.train_dataset = full
            self.val_dataset = full
        else:
            n_val = max(1, int(len(full) * val_fraction))
            n_train = len(full) - n_val
            # Seed for reproducibility
            gen = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset = random_split(
                full, [n_train, n_val], generator=gen
            )
            print(f"[trainer] Split {len(full)} samples into "
                  f"{n_train} train / {n_val} val")

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=samar_collate_fn,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=samar_collate_fn,
        )

    def _lr_lambda(self, step):
        """Linear warmup over ``warmup_steps``, then constant."""
        if self.warmup_steps <= 0:
            return 1.0
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        return 1.0

    def training_step(self, batch):
            """One training step. Returns scalar loss.

            Round-3 audit C1 fixed ``tgt=latent`` -> ``latent=latent``
            (round 1 had only fixed the unused ``forward`` method).

            The architectural shape of this loss comes from the original
            design: the model is trained to map (latent, events) -> a
            per-step 128-dim prediction, and the loss is MSE against the
            ground-truth latent vectors from ``latents.pt``. The model's
            ``sample()`` method treats the 128-dim output as if it were
            vocab logits via ``argmax``; this is a pre-existing design
            inconsistency in the model that should be addressed in a
            separate refactor (a future round-4 audit item).
            """
            input_ids = batch["input_ids"].to(self.device)
            # Round-3 audit: ``batch["labels"]`` is the next-event-id
            # sequence (unused in the current loss). ``batch["latent"]``
            # is the per-step 128-dim VAE target.
            latent = batch.get("latent")
            if latent is None:
                raise RuntimeError(
                    "batch['latent'] is required but missing. Regenerate "
                    "`latents/latents.pt` via "
                    "`python -m samar.precompute_samar_latents`."
                )
            latent = latent.to(self.device)  # [B, T, 128]

            description = batch.get("description")
            if description is not None:
                description = description.to(self.device)

            # Forward pass.
            predicted = self.model(input_ids, latent=latent, description=description)
            # Model returns ``[T, B, dim]`` (PyTorch nn.Transformer output).
            predicted = predicted.permute(1, 0, 2)  # [B, T, dim]

            # Align sequence lengths (defensive: in practice they match).
            L = min(predicted.size(1), latent.size(1))
            predicted = predicted[:, :L, :]
            latent = latent[:, :L, :]

            return F.mse_loss(predicted, latent)

    def validation_step(self, batch):
        """One validation step. Returns scalar loss (no grad)."""
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.device)
            latent = batch.get("latent")
            if latent is None:
                raise RuntimeError("batch['latent'] is required but missing.")
            latent = latent.to(self.device)

            description = batch.get("description")
            if description is not None:
                description = description.to(self.device)

            predicted = self.model(input_ids, latent=latent, description=description)
            predicted = predicted.permute(1, 0, 2)
            L = min(predicted.size(1), latent.size(1))
            predicted = predicted[:, :L, :]
            latent = latent[:, :L, :]
            return F.mse_loss(predicted, latent)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=self._lr_lambda)
        return optimizer, scheduler

    def save_model(self):
        """Save weights + config JSON. The JSON is regenerated every
        training run so it always matches the vocab/checkpoint state.
        It's gitignored -- see ``.gitignore``."""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), WEIGHTS_PATH)
        config = self.model.get_config()
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[trainer] Saved weights -> {WEIGHTS_PATH}")
        print(f"[trainer] Saved config  -> {CONFIG_PATH}")

    def train(self, num_epochs=10, log_every_n_batches=50):
        optimizer, scheduler = self.configure_optimizers()

        for epoch in range(num_epochs):
            # === Train ===
            self.model.train()
            train_loss_sum = 0.0
            train_loss_count = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss = self.training_step(batch)
                optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.gradient_clip)
                optimizer.step()
                scheduler.step()
                train_loss_sum += loss.item()
                train_loss_count += 1
                if (batch_idx + 1) % log_every_n_batches == 0:
                    print(f"  [epoch {epoch+1} batch {batch_idx+1}] "
                          f"loss={loss.item():.4f}")

            train_avg = train_loss_sum / max(1, train_loss_count)

            # === Validation ===
            self.model.eval()
            val_loss_sum = 0.0
            val_loss_count = 0
            with torch.no_grad():
                for batch in self.val_dataloader:
                    val_loss = self.validation_step(batch)
                    val_loss_sum += val_loss.item()
                    val_loss_count += 1
            val_avg = val_loss_sum / max(1, val_loss_count)

            print(f"Epoch {epoch+1}/{num_epochs}  "
                  f"train_loss={train_avg:.4f}  val_loss={val_avg:.4f}")

            # Round-5 fix: save best-val checkpoint every epoch so we
            # never lose training progress to a crash. Previously the
            # model was only saved at the END of all epochs, which
            # meant any mid-training failure lost everything.
            if val_avg < best_val:
                best_val = val_avg
                self.save_model()
                print(f"[trainer] New best val_loss={val_avg:.4f}")

            self.save_model()  # Final save even if not the best


def load_trained_transformer():
    """Load the trained transformer with warm-started missing layers.

    Uses :meth:`SamarTransformer.from_pretrained` so a checkpoint saved
    before ``description_embedding`` / ``pos_embedding`` were added
    still loads cleanly (those layers are warm-started from existing
    weights -- see the audit-round-1 finding #4 notes and
    ``SamarTransformer.from_pretrained``).

    Returns a model in ``eval()`` mode on the auto-selected device.
    """
    from .models.samar_transformer import SamarTransformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model, report = SamarTransformer.from_pretrained(
        WEIGHTS_PATH, config=config, device=str(device)
    )
    if report["missing"]:
        print(f"[load_trained_transformer] warm-started missing layers: {report['missing']}")
    if report["unexpected"]:
        print(f"[load_trained_transformer] ignored unexpected keys: {report['unexpected']}")
    model.eval()
    return model


if __name__ == "__main__":
    latent_path = os.path.join(_REPO_ROOT, "latents", "latents.pt")

    model = SamarTransformer(
        d_model=DEFAULT_D_MODEL,
        n_head=DEFAULT_N_HEAD,
        num_layers=DEFAULT_NUM_LAYERS,
        dim_feedforward=DEFAULT_DIM_FEEDFORWARD,
        dropout=DEFAULT_DROPOUT,
        vocab_size=DEFAULT_VOCAB_SIZE,
        latent_dim=DEFAULT_LATENT_DIM,
    )

    trainer = SamarTransformerTrainer(
        model=model,
        latent_path=latent_path,
        tokenizer=_load_tokenizer(),
        batch_size=16,
        lr=3e-4,
        context_size=256,
        val_fraction=0.1,    # round-3 audit T1: actual train/val split
        gradient_clip=1.0,   # round-3 audit T5
        warmup_steps=1000,   # round-3 audit T7
    )
    trainer.train(num_epochs=10)