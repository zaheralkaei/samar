# -*- coding: utf-8 -*-
"""
Training script for the autoregressive Transformer decoder.

The transformer takes (latent, description, events) as input and
predicts the next event token. See ``docs/audit-round-3.md`` for the
audit history of this file.

Round-17: extended ``save_model`` / ``--resume`` to save and restore
the full training state (model + optimizer + scheduler + epoch +
best_val). Previously ``--resume`` only restored model weights, so
the optimizer momentum reset to zero and the LR schedule re-warmed
up over the first 1000 steps of the resumed run. The round-17
extension enables true continue-training where the resumed run picks
up exactly where the previous run left off.

Backward compatibility: if the checkpoint file contains only
``state_dict`` (the round-15/16 format, no optimizer/scheduler),
``--resume`` falls back to warm-start-only behavior (weights load,
fresh optimizer/scheduler). The trainer prints a notice when this
fallback fires.
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

def safe_cross_entropy(logits, labels, ignore_index=0):
    """Cross-entropy loss that's safe against all-ignored-label batches.

    PyTorch's ``F.cross_entropy`` with ``reduction='mean'`` returns NaN
    if **all** labels equal ``ignore_index`` (no gradients to average
    over). With our real data this never fires, but a future regression
    (e.g. a dataset that accidentally pads everything) would propagate
    NaN through the optimizer and corrupt the model. Use
    ``reduction='sum'`` and divide by max(1, count) instead.

    Round-14 audit.
    """
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction='sum',
    )
    n_valid = (labels != ignore_index).sum().clamp_min(1)
    return loss / n_valid



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
    context_size=512,
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
        self.tokenizer = tokenizer or _load_tokenizer()

        # Round-17: optimizer and scheduler are created lazily in
        # ``configure_optimizers()`` (called by ``train()``). When the
        # trainer is resumed, the loaded optimizer/scheduler state is
        # assigned here before ``train()`` is called so ``save_model``
        # picks up the restored state.
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0   # round-17: set by --resume to skip already-done epochs
        self.initial_best_val = float("inf")  # round-17: best_val restored from checkpoint

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
                  f"{n_train} train / {n_val} val", flush=True)

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

            Round-8 architectural fix: loss is now CrossEntropy on
            next-event-id prediction (proper LM-style training), not
            MSE against latents. The model's ``output_layer`` projects
            to vocab_size directly so we can compute the token
            distribution at each step.

            The ``labels`` field is the standard autoregressive shift:
            ``labels[t] = input_ids[t+1]`` for t in [0, T-1]. We
            compare ``logits[t]`` (predicted distribution at step t)
            against ``labels[t]`` (the actual next token).
            """
            input_ids = batch["input_ids"].to(self.device)
            latent = batch.get("latent")
            description = batch.get("description")
            if description is not None:
                description = description.to(self.device)
            # Latent is used as decoder input style conditioning
            # (the VAE latent space is still useful as a "piece style"
            # signal). It is no longer the loss target.
            if latent is not None:
                latent = latent.to(self.device)

            # Forward pass.
            logits = self.model(input_ids, latent=latent, description=description)
            # Model returns ``[T, B, vocab_size]`` (PyTorch nn.Transformer output).
            logits = logits.permute(1, 0, 2)  # [B, T, vocab_size]

            # Build the autoregressive labels: predict input_ids[t+1] from
            # logits[t]. The model's `labels` field already encodes this
            # shift (see SamarLatentDataset). Pad positions use pad_id (0).
            labels = batch.get("labels")
            if labels is None:
                raise RuntimeError(
                    "batch['labels'] is required but missing. The round-8 "
                    "trainer needs the next-event-id targets."
                )
            labels = labels.to(self.device)  # [B, T]

            # Cross-entropy over the full sequence (ignore_index=0 is pad_id).
            # Use safe_cross_entropy (round-14 audit) to avoid NaN if a
            # batch has all-pad labels (F.cross_entropy mean-reduction
            # returns NaN in that case).
            loss = safe_cross_entropy(logits, labels, ignore_index=0)
            return loss

    def validation_step(self, batch):
        """One validation step. Returns scalar loss (no grad)."""
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.device)
            latent = batch.get("latent")
            description = batch.get("description")
            if description is not None:
                description = description.to(self.device)
            if latent is not None:
                latent = latent.to(self.device)

            logits = self.model(input_ids, latent=latent, description=description)
            logits = logits.permute(1, 0, 2)

            labels = batch.get("labels")
            if labels is None:
                raise RuntimeError("batch['labels'] is required but missing.")
            labels = labels.to(self.device)

            loss = safe_cross_entropy(logits, labels, ignore_index=0)
            return loss

    def configure_optimizers(self):
        """Build Adam + LambdaLR scheduler. Stores both on ``self`` so
        ``save_model`` and ``--resume`` can round-trip them.

        Round-17: previously returned (optimizer, scheduler) without
        attaching to self, so resume could not restore their state.
        """
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=self._lr_lambda)
        self.optimizer = optimizer
        self.scheduler = scheduler
        return optimizer, scheduler

    def save_model(self, epoch=None, best_val=None):
        """Save full training state to ``WEIGHTS_PATH``.

        Round-17: extended from weights-only to a full state dict
        containing ``model_state_dict``, ``optimizer_state_dict``,
        ``scheduler_state_dict``, ``epoch``, ``best_val``, and
        ``lr``. The JSON config is regenerated every save from
        ``model.get_config()`` so it always matches the model shape.

        Backward compat: old checkpoints that contain only
        ``state_dict()`` keys (round-15/16 format) still load
        cleanly -- ``load_pretrained`` in ``--resume`` detects the
        old format and falls back to warm-start.

        The .pt is now over 35MB; the JSON is gitignored (see
        ``.gitignore``).
        """
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        # Round-17: full state. If optimizer/scheduler haven't been
        # built yet (i.e. save_model was called before train()),
        # store None placeholders so the dict shape is consistent.
        state = {
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "context_size": self.context_size,
            "round": 17,
        }
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
            # ``last_epoch`` is the canonical resume position for
            # LambdaLR. PyTorch's LambdaLR uses last_epoch to offset
            # the lr_lambda step counter so the schedule continues
            # smoothly across resumes.
            state["scheduler_last_epoch"] = self.scheduler.last_epoch
        torch.save(state, WEIGHTS_PATH)
        config = self.model.get_config()
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[trainer] Saved weights -> {WEIGHTS_PATH}")
        print(f"[trainer] Saved config  -> {CONFIG_PATH}")

    def load_pretrained_state(self, path):
        """Load a checkpoint produced by ``save_model`` (round-17+).

        Returns a dict with keys:
          - 'state_dict' : model weights (always present)
          - 'optimizer_state_dict' : or None if old-format checkpoint
          - 'scheduler_state_dict' : or None
          - 'epoch' : int (0 if old format)
          - 'best_val' : float (inf if old format)
          - 'lr' : float (None if old format; uses trainer default otherwise)
          - 'is_old_format' : bool

        Round-17: distinguishes between the round-15/16 weight-only
        format and the round-17 full state format. The caller
        (``__main__``) decides whether to restore optimizer/scheduler
        state or fall back to warm-start.
        """
        loaded = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            # Round-17+ full state
            return {
                "state_dict": loaded["model_state_dict"],
                "optimizer_state_dict": loaded.get("optimizer_state_dict"),
                "scheduler_state_dict": loaded.get("scheduler_state_dict"),
                "scheduler_last_epoch": loaded.get("scheduler_last_epoch", 0),
                "epoch": loaded.get("epoch", 0),
                "best_val": loaded.get("best_val", float("inf")),
                "lr": loaded.get("lr"),
                "is_old_format": False,
            }
        # Old format: bare state_dict from round-15/16
        return {
            "state_dict": loaded,
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "scheduler_last_epoch": 0,
            "epoch": 0,
            "best_val": float("inf"),
            "lr": None,
            "is_old_format": True,
        }

    def train(self, num_epochs=10, log_every_n_batches=50):
        """Train for ``num_epochs`` epochs.

        Round-17: if ``self.start_epoch > 0`` (set by --resume), the
        epoch loop starts from there so we don't re-do work. The
        scheduler's ``last_epoch`` is restored from the checkpoint so
        the LR schedule continues smoothly (no double warmup).
        """
        optimizer, scheduler = self.configure_optimizers()

        # Round-17: if --resume populated self.optimizer / self.scheduler,
        # restore the loaded state. configure_optimizers() just
        # attached fresh ones; we overwrite with the loaded ones below.
        if hasattr(self, "_resume_optimizer_state") and self._resume_optimizer_state is not None:
            optimizer.load_state_dict(self._resume_optimizer_state)
            self.optimizer = optimizer
            print(f"[trainer] Restored optimizer state from resume", flush=True)
        if hasattr(self, "_resume_scheduler_state") and self._resume_scheduler_state is not None:
            scheduler.load_state_dict(self._resume_scheduler_state)
            self.scheduler = scheduler
            print(f"[trainer] Restored scheduler state from resume "
                  f"(last_epoch={scheduler.last_epoch})", flush=True)
        elif hasattr(self, "_resume_scheduler_last_epoch"):
            # Old-format resume: advance scheduler.last_epoch to the
            # checkpoint's epoch count so the LR continues from there.
            scheduler.last_epoch = self._resume_scheduler_last_epoch
            self.scheduler = scheduler

        best_val = self.initial_best_val  # round-17: restored from checkpoint

        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
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
                          f"loss={loss.item():.4f}", flush=True)

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

            print(f"Epoch {epoch+1}/{self.start_epoch + num_epochs}  "
                  f"train_loss={train_avg:.4f}  val_loss={val_avg:.4f}",
                  flush=True)

            # Round-5 fix: save best-val checkpoint every epoch so we
            # never lose training progress to a crash. Previously the
            # model was only saved at the END of all epochs, which
            # meant any mid-training failure lost everything.
            if val_avg < best_val:
                best_val = val_avg
                self.save_model(epoch=epoch + 1, best_val=best_val)
                print(f"[trainer] New best val_loss={val_avg:.4f}", flush=True)
            else:
                # Even if not best, save the latest state so a
                # resume picks up the most recent epoch. Round-17.
                self.save_model(epoch=epoch + 1, best_val=best_val)


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-epochs", type=int, default=10,
        help="Number of training epochs (round-9: 50 recommended).",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--context-size", type=int, default=512,
        help=(
            "Length of input event sequences. Must match the "
            "checkpoint's max_len (default 512). Shorter samples are "
            "padded, longer samples are truncated."
        ),
    )
    parser.add_argument(
        "--latent-path", type=str, default=None,
        help=(
            "Path to precomputed latents.pt. Default: "
            "latents/latents.pt (Arabic MusicXML). Pass "
            "latents/midi_latents.pt to train on Western classical MIDI."
        ),
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help=(
            "Path to a checkpoint .pt to continue training from. Round-17: "
            "restores model weights, optimizer state (Adam momentum), "
            "scheduler state (LR position), epoch counter, and best_val. "
            "Old-format (round-15/16) checkpoints that only contain "
            "weights fall back to warm-start-only behavior."
        ),
    )
    args = parser.parse_args()

    if args.latent_path:
        latent_path = args.latent_path
    else:
        latent_path = os.path.join(_REPO_ROOT, "latents", "latents.pt")

    # Round-14: build the model with the same max_len as the CLI
    # context_size, so the positional embedding capacity matches.
    model = SamarTransformer(
        d_model=DEFAULT_D_MODEL,
        n_head=DEFAULT_N_HEAD,
        num_layers=DEFAULT_NUM_LAYERS,
        dim_feedforward=DEFAULT_DIM_FEEDFORWARD,
        dropout=DEFAULT_DROPOUT,
        vocab_size=DEFAULT_VOCAB_SIZE,
        latent_dim=DEFAULT_LATENT_DIM,
        max_len=args.context_size,
    )

    # Round-15 / Round-17: warm-start or full-resume from an existing
    # checkpoint if --resume is provided.
    if args.resume:
        print(f"[trainer] Resuming from checkpoint: {args.resume}", flush=True)
        # Use the trainer's load_pretrained_state helper so we can
        # detect old-format vs new-format checkpoints in one place.
        # We have to instantiate the trainer first to get the helper,
        # but the trainer needs the model + latent_path, so do the
        # minimal setup here, call load_pretrained_state, then build
        # the trainer with the restored start_epoch / state.
        # Actually simpler: build the trainer after we know the
        # checkpoint format, but the trainer needs latent_path +
        # model. Construct a stub trainer to call the helper:
        stub_tokenizer = _load_tokenizer()
        stub_trainer = SamarTransformerTrainer(
            model=model,
            latent_path=latent_path,
            tokenizer=stub_tokenizer,
            batch_size=16,
            lr=args.lr,
            context_size=args.context_size,
            val_fraction=0.1,
            gradient_clip=1.0,
            warmup_steps=1000,
        )
        loaded = stub_trainer.load_pretrained_state(args.resume)
        if loaded["is_old_format"]:
            print(f"[trainer] Old-format checkpoint (weights only). "
                  f"Falling back to warm-start; optimizer/scheduler will "
                  f"reset and LR will re-warmup. Re-save with the new "
                  f"trainer (round-17) to enable full resume.",
                  flush=True)
        else:
            print(f"[trainer] Full-state checkpoint detected: "
                  f"epoch={loaded['epoch']}, best_val={loaded['best_val']:.4f}",
                  flush=True)
        # Load the model weights via SamarTransformer.from_pretrained
        # so we get the standard missing-keys report.
        resume_cfg_path = args.resume.replace(".pt", "_config.json")
        if os.path.exists(resume_cfg_path):
            with open(resume_cfg_path) as f_cfg:
                resume_cfg = json.load(f_cfg)
        else:
            print(f"[trainer] WARNING: no config file at {resume_cfg_path}, "
                  f"using default config", flush=True)
            resume_cfg = None
        model, report = SamarTransformer.from_pretrained(
            args.resume, config=resume_cfg, device="cpu",
        )
        if report["missing"]:
            print(f"[trainer] warm-started missing: {report['missing']}", flush=True)
        if report["unexpected"]:
            print(f"[trainer] ignored unexpected: {report['unexpected']}", flush=True)
        # Round-17: carry the loaded state forward to the trainer.
        # We rebuild the trainer (discarding stub_trainer) but pass
        # the loaded state via attributes the trainer expects.
        _resume_loaded = loaded
    else:
        _resume_loaded = None

    trainer = SamarTransformerTrainer(
        model=model,
        latent_path=latent_path,
        tokenizer=_load_tokenizer(),
        batch_size=16,
        lr=args.lr,
        context_size=args.context_size,
        val_fraction=0.1,    # round-3 audit T1: actual train/val split
        gradient_clip=1.0,   # round-3 audit T5
        warmup_steps=1000,   # round-3 audit T7
    )

    if _resume_loaded is not None:
        # Round-17: wire the loaded state into the trainer. The
        # trainer's train() picks up these attributes.
        trainer.start_epoch = _resume_loaded["epoch"]
        trainer.initial_best_val = _resume_loaded["best_val"]
        trainer._resume_optimizer_state = _resume_loaded["optimizer_state_dict"]
        trainer._resume_scheduler_state = _resume_loaded["scheduler_state_dict"]
        trainer._resume_scheduler_last_epoch = _resume_loaded["scheduler_last_epoch"]
        print(f"[trainer] Resuming from epoch {trainer.start_epoch}, "
              f"best_val={trainer.initial_best_val:.4f}", flush=True)

    # Round-16 R16-A: removed duplicate trainer.train() call.
    # Round-14's trainer-hardening commit (5be1370) accidentally added a
    # second trainer.train() call here on top of round-9's original. Result:
    # --num-epochs 10 was actually running 20 epochs AND resetting the LR
    # schedule mid-training. Round-15 claimed to fix this but the fix did
    # not actually land in the file. Verify with: grep -c 'trainer\.train('
    # samar/train_samar_transformer.py -> must be 1.
    trainer.train(num_epochs=args.num_epochs)