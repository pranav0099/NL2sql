"""
NL2SQL — Phase 4: Transformer Training Loop
Run: python dl/trainer.py

Complete training pipeline with early stopping, model checkpointing,
learning-rate scheduling (ReduceLROnPlateau), gradient clipping,
and loss-curve visualisation.

**COMPLETE - Step 5 done**
"""

import sys
from pathlib import Path
import argparse
import time
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe on headless servers
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRAD_CLIP,
    EARLY_STOP_PATIENCE,
    DL_MODEL_PATH,
)
from dl.model import create_model
from dl.dataset import load_vocab, get_dataloaders
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# TRAINER CLASS
# ============================================================================

class TransformerTrainer:
    """
    Training manager for the NL2SQL Transformer.

    Encapsulates the optimiser, loss criterion, learning-rate scheduler,
    and the full train / validate / checkpoint loop with early stopping.
    """

    def __init__(
        self,
        model,
        nl_vocab: Dict,
        sql_vocab: Dict,
        learning_rate: float = LEARNING_RATE,
        grad_clip: float = GRAD_CLIP,
    ):
        """
        Args:
            model: NL2SQLTransformer instance (already on DEVICE).
            nl_vocab: NL vocabulary dict.
            sql_vocab: SQL vocabulary dict.
            learning_rate: Initial learning rate for Adam.
            grad_clip: Maximum gradient norm for clipping.
        """
        self.model = model.to(DEVICE)
        self.nl_vocab = nl_vocab
        self.sql_vocab = sql_vocab
        self.pad_idx = sql_vocab["tok2idx"]["<pad>"]
        self.grad_clip = grad_clip

        # ── Loss (cross-entropy with label smoothing, ignoring PAD) ──────
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_idx,
            label_smoothing=0.1,
        )

        # ── Optimiser (Adam with Transformer-paper betas) ────────────────
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # ── LR scheduler ────────────────────────────────────────────────
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

        # ── Tracking ────────────────────────────────────────────────────
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.train_losses = []
        self.val_losses = []

        logger.info("TransformerTrainer initialised")
        logger.info(f"  LR: {learning_rate}")
        logger.info(f"  Grad clip: {grad_clip}")
        logger.info(f"  Label smoothing: 0.1")
        logger.info(f"  PAD idx: {self.pad_idx}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Execute one full training epoch.

        Args:
            train_loader: Training DataLoader.

        Returns:
            Average training loss over all batches.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        tgt_vocab_size = self.sql_vocab["size"]

        for batch_idx, batch in enumerate(train_loader):
            try:
                src = batch["src"].to(DEVICE)
                tgt = batch["tgt"].to(DEVICE)

                # Teacher-forced inputs / outputs
                tgt_input = tgt[:, :-1]   # all except last token
                tgt_output = tgt[:, 1:]   # all except SOS (first token)

                # Forward
                logits = self.model(src, tgt_input)

                # Loss
                loss = self.criterion(
                    logits.reshape(-1, tgt_vocab_size),
                    tgt_output.reshape(-1),
                )

                # Backward
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

                # Progress logging
                if (batch_idx + 1) % 100 == 0:
                    logger.info(
                        f"  Batch {batch_idx + 1}/{num_batches}  "
                        f"batch_loss={loss.item():.4f}"
                    )

            except RuntimeError as exc:
                logger.error(f"RuntimeError in batch {batch_idx}: {exc}")
                self.optimizer.zero_grad()
                continue

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate model on validation set.

        Args:
            val_loader: Validation DataLoader.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        tgt_vocab_size = self.sql_vocab["size"]

        for batch in val_loader:
            try:
                src = batch["src"].to(DEVICE)
                tgt = batch["tgt"].to(DEVICE)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, tgt_vocab_size),
                    tgt_output.reshape(-1),
                )
                total_loss += loss.item()

            except RuntimeError as exc:
                logger.error(f"RuntimeError during evaluation: {exc}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"  Validation loss: {avg_loss:.4f}")
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = NUM_EPOCHS,
        save_path: str = DL_MODEL_PATH,
        patience: int = EARLY_STOP_PATIENCE,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping and checkpointing.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            num_epochs: Maximum number of epochs.
            save_path: Where to save best model checkpoint.
            patience: Stop if val loss doesn't improve for this many epochs.

        Returns:
            Training history dictionary.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("STARTING TRANSFORMER TRAINING")
        logger.info("=" * 80)
        logger.info(f"  Train samples: {len(train_loader.dataset)}")
        logger.info(f"  Val samples  : {len(val_loader.dataset)}")
        logger.info(f"  Max epochs   : {num_epochs}")
        logger.info(f"  Patience     : {patience}")
        logger.info(f"  Device       : {DEVICE}")

        best_epoch = 0
        training_start = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.evaluate(val_loader)

            # Record
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # LR scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                logger.info(f"  LR reduced: {current_lr:.2e} → {new_lr:.2e}")

            epoch_time = time.time() - epoch_start

            # Check improvement
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                best_epoch = epoch

                self.save_checkpoint(str(save_path), epoch, val_loss)
                logger.info(
                    f"  ✓ Best model saved (improved by {improvement:.4f})"
                )
            else:
                self.epochs_no_improve += 1
                logger.info(
                    f"  ✗ No improvement for "
                    f"{self.epochs_no_improve}/{patience} epochs"
                )

            # Epoch summary (print for console)
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"LR: {new_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Early stopping
            if self.epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        total_time = time.time() - training_start

        # Reload best checkpoint
        if save_path.exists():
            self.load_checkpoint(str(save_path))
            logger.info("Best checkpoint reloaded")

        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "best_epoch": best_epoch,
            "total_epochs": len(self.val_losses),
            "stopped_early": self.epochs_no_improve >= patience,
            "total_time_seconds": total_time,
        }

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Best val loss : {self.best_val_loss:.4f}")
        logger.info(f"  Best epoch    : {best_epoch}")
        logger.info(f"  Total epochs  : {history['total_epochs']}")
        logger.info(f"  Total time    : {total_time / 60:.1f} min")
        logger.info("=" * 80)

        return history

    def save_checkpoint(
        self, path: str, epoch: int, val_loss: float
    ) -> None:
        """
        Save model checkpoint including optimiser state and config.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "nl_vocab_size": self.nl_vocab["size"],
            "sql_vocab_size": self.sql_vocab["size"],
            "config": {
                "embed_dim": self.model.embed_dim,
                "num_heads": self.model.num_heads,
                "num_encoder_layers": self.model.num_encoder_layers,
                "num_decoder_layers": self.model.num_decoder_layers,
                "ffn_dim": self.model.ffn_dim,
                "dropout": self.model.dropout_rate,
            },
        }

        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved → {path}  (epoch {epoch}, val_loss {val_loss:.4f})")

    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint into current model and optimiser.

        Args:
            path: Path to .pt checkpoint.

        Returns:
            Epoch number from checkpoint.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            logger.error(f"Checkpoint not found: {path}")
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as exc:
                logger.warning(f"Could not restore optimiser state: {exc}")

        epoch = checkpoint.get("epoch", 0)
        val_loss = checkpoint.get("val_loss", float("inf"))
        self.best_val_loss = val_loss

        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"  Epoch: {epoch}")
        logger.info(f"  Val loss: {val_loss:.4f}")

        return epoch

    def plot_losses(self, save_path: Optional[str] = None) -> None:
        """
        Plot training vs. validation loss curves.
        """
        if not self.train_losses or not self.val_losses:
            logger.warning("No loss data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, "b-o", label="Train Loss", linewidth=2, markersize=4)
        ax.plot(epochs, self.val_losses, "r-o", label="Validation Loss", linewidth=2, markersize=4)

        best_idx = self.val_losses.index(min(self.val_losses))
        best_epoch = best_idx + 1
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("NL2SQL Transformer Training Curves", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150, bbox_inches="tight")
            logger.info(f"Loss plot saved → {out}")
        else:
            plt.show()

        plt.close(fig)


# ============================================================================
# STANDALONE TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NL2SQL Phase 4 — Transformer Training")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples for quick testing (None = all)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("NL2SQL Phase 4 — Transformer Training")
    print("=" * 60)
    print(f"Device     : {DEVICE}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"LR         : {args.lr}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Resume     : {args.resume}")

    try:
        # Vocabulary
        nl_vocab, sql_vocab = load_vocab("data/processed/vocab.json")

        # DataLoaders
        train_loader, val_loader = get_dataloaders(
            "data/processed/train.json",
            "data/processed/val.json",
            nl_vocab, sql_vocab,
            batch_size=args.batch_size,
        )

        # Truncate for quick testing
        if args.max_samples and args.max_samples < len(train_loader.dataset):
            logger.info(f"Truncating training set to {args.max_samples} samples")
            train_loader.dataset.records = train_loader.dataset.records[:args.max_samples]

        # Model & Trainer
        model = create_model(nl_vocab, sql_vocab)
        trainer = TransformerTrainer(model, nl_vocab, sql_vocab, learning_rate=args.lr)

        # Resume
        if args.resume and Path(DL_MODEL_PATH).exists():
            trainer.load_checkpoint(str(DL_MODEL_PATH))

        # Train
        history = trainer.train(train_loader, val_loader, num_epochs=args.epochs)

        # Plot
        trainer.plot_losses("logs/training_curve.png")

        print("\nTraining complete.")
        print(f"Best val loss : {history['best_val_loss']:.4f}")
        print(f"Best epoch    : {history['best_epoch']}")
        print(f"Stopped early : {history['stopped_early']}")
        print(f"Model saved   : {DL_MODEL_PATH}")
        print("\nNext: python dl/generator.py")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("\n[WARN] Training interrupted")
        sys.exit(130)

    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        print(f"\n[FAIL] Training failed: {exc}")
        sys.exit(1)

    # ── HOW TO RUN PHASE 4 ──────────────────────────
    print("""
IMPORTANT: Training takes time. Use these flags to test quickly first:

Step 1 — Quick test (5 epochs, 1000 samples):
  python dl/trainer.py --epochs 5 --max_samples 1000

Step 2 — Verify model works:
  python dl/generator.py

Step 3 — Full training (when ready):
  python dl/trainer.py --epochs 30

Step 4 — Resume if interrupted:
  python dl/trainer.py --epochs 30 --resume

Expected output during training:
  Epoch  1/30 | Train: 4.2341 | Val: 3.8921 | LR: 1e-4
  Epoch  2/30 | Train: 3.1205 | Val: 2.9431 | LR: 1e-4
  Epoch  5/30 | Train: 1.8432 | Val: 1.9123 | LR: 1e-4
  ...
  Epoch 15/30 | Train: 0.4231 | Val: 0.6821 | LR: 5e-5
  Early stopping triggered at epoch 20

Expected BLEU after full training: 0.65 - 0.80
Expected exact match: 35% - 55%
""")

    # ── WHAT PHASE 5 NEEDS FROM PHASE 4 ─────────────
    print("""
Phase 5 (Memory) imports from Phase 4 like this:

  from dl.generator import SQLGenerator
  generator = SQLGenerator()
  result = generator.generate(
      question   = "show customers from mumbai",
      nlp_result = nlp_pipeline.process(question)
  )
  sql        = result["sql"]
  confidence = result["confidence"]
  fallback   = result["fallback_used"]
""")

