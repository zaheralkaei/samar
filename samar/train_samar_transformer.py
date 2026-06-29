# -*- coding: utf-8 -*-
"""
Training script for the SAMAR Transformer (FIGARO-aligned).

Round 18: Rewritten to match FIGARO figaro-expert architecture.
- Description-only encoder (no VQ-VAE latent)
- bar_ids + position_ids structural embeddings
- Separate encoder/decoder layer counts
- Reduced model size and warmup for small dataset
"""

import os
import json

# Round-21: force all BLAS/MKL/OpenMP threads to use all 8 CPU cores.
# Without this the trainer only used 4 threads (~6% CPU efficiency on
# the 8-core ai-laptop), making each batch take ~120s instead of ~7s.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import torch
torch.set_num_threads(8)

from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from .dataset import SamarLatentDataset, samar_collate_fn
from .models.samar_transformer import SamarTransformer
from .tokenizer import SamarTokenizer

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)

# === Default configuration (round 18) ===
# Round-20: vocab grew from 1254 to 1265 after adding Tie_Start,
# Tie_Stop, Dot_0/1/2, Tuplet_3/5/6/7/12, Chord_On tokens. The old
# model checkpoint (output_layer = 1254) is incompatible with this
# new vocab; training a fresh transformer is required.
DEFAULT_VOCAB_SIZE = 1265
DEFAULT_D_MODEL = 256
DEFAULT_N_HEAD = 4
DEFAULT_NUM_ENCODER_LAYERS = 2
DEFAULT_NUM_DECODER_LAYERS = 4
DEFAULT_DIM_FEEDFORWARD = 512
DEFAULT_DROPOUT = 0.1

CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "checkpoints")
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "samar_transformer.pt")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "samar_transformer_config.json")
VOCAB_PATH = os.path.join(_PACKAGE_DIR, "samar_vocab.pkl")


def safe_cross_entropy(logits, labels, ignore_index=0):
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction='sum',
    )
    n_valid = (labels != ignore_index).sum().clamp_min(1)
    return loss / n_valid


def _load_tokenizer():
    return SamarTokenizer.load(VOCAB_PATH)


class SamarTransformerTrainer:
    def __init__(self, model, latent_path=None, batch_size=16, lr=1e-4,
                 context_size=256, val_fraction=0.1, gradient_clip=1.0,
                 warmup_steps=100, tokenizer=None, weights_path=None,
                 config_path=None, round_tag=18):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.context_size = context_size
        self.gradient_clip = gradient_clip
        self.warmup_steps = warmup_steps
        self.tokenizer = tokenizer or _load_tokenizer()
        self.weights_path = weights_path or os.path.join(CHECKPOINT_DIR, "samar_transformer.pt")
        self.config_path = config_path or os.path.join(CHECKPOINT_DIR, "samar_transformer_config.json")
        self.round_tag = round_tag

        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.initial_best_val = float("inf")

        if not latent_path:
            raise ValueError("latent_path must be provided.")

        full = SamarLatentDataset(
            latent_path, context_size=context_size, tokenizer=self.tokenizer
        )
        if len(full) == 0:
            raise RuntimeError(
                f"SamarLatentDataset has 0 samples from {latent_path}. "
                "Run `python -m samar.precompute_samar_latents` first."
            )

        if val_fraction <= 0:
            print("[trainer] WARNING: val_fraction=0, using same data for train and val")
            self.train_dataset = full
            self.val_dataset = full
        else:
            n_val = max(1, int(len(full) * val_fraction))
            n_train = len(full) - n_val
            gen = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset = random_split(
                full, [n_train, n_val], generator=gen
            )
            print(f"[trainer] Split {len(full)} samples into "
                  f"{n_train} train / {n_val} val", flush=True)

        # Round-21: num_workers=4 for parallel data loading. The
        # dataloader was the bottleneck on the full 12,378-sample
        # dataset (each batch waited ~80s for collate to finish even
        # with 8 OMP threads doing the matmul work). 4 workers is
        # enough to keep the GPU/CPU fed; 8 caused memory pressure
        # on the 8-core ai-laptop.
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=samar_collate_fn, num_workers=4,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=samar_collate_fn, num_workers=4,
        )

    def _lr_lambda(self, step):
        """Sqrt decay schedule (FIGARO pattern): warmup then 1/sqrt(step/warmup)."""
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, 1.0 / (max(step, 1) / self.warmup_steps) ** 0.5)

    def training_step(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        bar_ids = batch["bar_ids"].to(self.device)
        position_ids = batch["position_ids"].to(self.device)
        description = batch.get("description")
        desc_bar_ids = batch.get("desc_bar_ids")
        if description is not None:
            description = description.to(self.device)
        if desc_bar_ids is not None:
            desc_bar_ids = desc_bar_ids.to(self.device)

        logits = self.model(
            input_ids, description=description, bar_ids=bar_ids,
            position_ids=position_ids, desc_bar_ids=desc_bar_ids,
        )
        logits = logits.permute(1, 0, 2)  # [B, T, vocab]
        loss = safe_cross_entropy(logits, labels, ignore_index=0)
        return loss

    def validation_step(self, batch):
        with torch.no_grad():
            return self.training_step(batch)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = LambdaLR(optimizer, lr_lambda=self._lr_lambda)
        self.optimizer = optimizer
        self.scheduler = scheduler
        return optimizer, scheduler

    def save_model(self, epoch=None, best_val=None, is_best=False):
        os.makedirs(os.path.dirname(self.weights_path) or ".", exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "context_size": self.context_size,
            "round": self.round_tag,
        }
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, self.weights_path)
        config = self.model.get_config()
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[trainer] Saved weights -> {self.weights_path}")
        print(f"[trainer] Saved config  -> {self.config_path}")
        if is_best:
            best_path = self.weights_path.replace(".pt", "_best.pt")
            torch.save(state, best_path)
            print(f"[trainer] Saved best   -> {best_path}")

    def train(self, num_epochs=10, log_every_n_batches=10):
        optimizer, scheduler = self.configure_optimizers()

        # Restore optimizer/scheduler from resume
        if hasattr(self, "_resume_optimizer_state") and self._resume_optimizer_state is not None:
            optimizer.load_state_dict(self._resume_optimizer_state)
            self.optimizer = optimizer
            print("[trainer] Restored optimizer state from resume", flush=True)
        if hasattr(self, "_resume_scheduler_state") and self._resume_scheduler_state is not None:
            scheduler.load_state_dict(self._resume_scheduler_state)
            self.scheduler = scheduler
            print(f"[trainer] Restored scheduler state from resume "
                  f"(last_epoch={scheduler.last_epoch})", flush=True)

        best_val = self.initial_best_val

        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_loss_count = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss = self.training_step(batch)
                optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.gradient_clip
                    )
                optimizer.step()
                scheduler.step()
                train_loss_sum += loss.item()
                train_loss_count += 1
                if (batch_idx + 1) % log_every_n_batches == 0:
                    print(f"  [epoch {epoch+1} batch {batch_idx+1}] "
                          f"loss={loss.item():.4f}", flush=True)

            train_avg = train_loss_sum / max(1, train_loss_count)

            self.model.eval()
            val_loss_sum = 0.0
            val_loss_count = 0
            with torch.no_grad():
                for batch in self.val_dataloader:
                    val_loss = self.validation_step(batch)
                    val_loss_sum += val_loss.item()
                    val_loss_count += 1
            val_avg = val_loss_sum / max(1, val_loss_count)

            print(f"Epoch {epoch+1}/{self.start_epoch + num_epochs}  "
                  f"train_loss={train_avg:.4f}  val_loss={val_avg:.4f}",
                  flush=True)

            if val_avg < best_val:
                best_val = val_avg
                print(f"[trainer] New best val_loss={val_avg:.4f}", flush=True)
                self.save_model(epoch=epoch + 1, best_val=best_val, is_best=True)
            else:
                self.save_model(epoch=epoch + 1, best_val=best_val)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--latent-path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Where to save the trained weights. Default: "
                             "checkpoints/samar_transformer.pt")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Where to save the model config. Default: "
                             "checkpoints/samar_transformer_config.json")
    parser.add_argument("--round", type=int, default=18,
                        help="Round tag written into checkpoint metadata "
                             "(helps distinguish runs).")
    args = parser.parse_args()

    if args.latent_path:
        latent_path = args.latent_path
    else:
        latent_path = os.path.join(_REPO_ROOT, "latents", "latents.pt")

    model = SamarTransformer(
        d_model=DEFAULT_D_MODEL,
        n_head=DEFAULT_N_HEAD,
        num_encoder_layers=DEFAULT_NUM_ENCODER_LAYERS,
        num_decoder_layers=DEFAULT_NUM_DECODER_LAYERS,
        dim_feedforward=DEFAULT_DIM_FEEDFORWARD,
        dropout=DEFAULT_DROPOUT,
        vocab_size=DEFAULT_VOCAB_SIZE,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Total parameters: {total_params:,}")

    _resume_loaded = None
    if args.resume:
        print(f"[trainer] Resuming from checkpoint: {args.resume}", flush=True)
        resume_cfg_path = args.resume.replace(".pt", "_config.json")
        if os.path.exists(resume_cfg_path):
            with open(resume_cfg_path) as f_cfg:
                resume_cfg = json.load(f_cfg)
        else:
            resume_cfg = model.get_config()

        model, report = SamarTransformer.from_pretrained(
            args.resume, config=resume_cfg, device="cpu",
        )
        if report["missing"]:
            print(f"[trainer] missing: {report['missing']}", flush=True)
        if report["unexpected"]:
            print(f"[trainer] unexpected: {report['unexpected']}", flush=True)

        loaded = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            _resume_loaded = {
                "optimizer_state_dict": loaded.get("optimizer_state_dict"),
                "scheduler_state_dict": loaded.get("scheduler_state_dict"),
                "epoch": loaded.get("epoch", 0),
                "best_val": loaded.get("best_val", float("inf")),
            }

    trainer = SamarTransformerTrainer(
        model=model,
        latent_path=latent_path,
        tokenizer=_load_tokenizer(),
        batch_size=args.batch_size,
        lr=args.lr,
        context_size=args.context_size,
        val_fraction=0.1,
        gradient_clip=1.0,
        warmup_steps=args.warmup_steps,
        weights_path=args.weights_path,
        config_path=args.config_path,
        round_tag=args.round,
    )

    if _resume_loaded is not None:
        trainer.start_epoch = _resume_loaded["epoch"] or 0
        trainer.initial_best_val = _resume_loaded["best_val"] or float("inf")
        trainer._resume_optimizer_state = _resume_loaded["optimizer_state_dict"]
        trainer._resume_scheduler_state = _resume_loaded["scheduler_state_dict"]
        print(f"[trainer] Resuming from epoch {trainer.start_epoch}, "
              f"best_val={trainer.initial_best_val:.4f}", flush=True)

    trainer.train(num_epochs=args.num_epochs)
