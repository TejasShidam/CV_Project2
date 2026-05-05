from __future__ import annotations

import logging
import math

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from anpr.config import Settings
from anpr.data.dataset import build_dataloader
from anpr.models.anpr_model import ANPRModel
from anpr.tokenizer import PlateTokenizer
from anpr.training.losses import cross_entropy_sequence_loss
from anpr.training.metrics import character_accuracy, full_plate_accuracy
from anpr.utils.io import ensure_dir, write_jsonl

logger = logging.getLogger(__name__)


def train_model(
    settings: Settings,
    epochs: int = 30,
    lr: float = 5e-4,
    patience: int = 7,
    max_grad_norm: float = 1.0,
    warmup_epochs: int = 3,
    resume: bool = False,
) -> None:
    """Train the ANPR model with cosine LR schedule, gradient clipping, and early stopping.

    Args:
        settings:       Project settings (paths, device, etc.)
        epochs:         Maximum training epochs.
        lr:             Peak learning rate (after warmup).
        patience:       Early-stopping patience in epochs.
        max_grad_norm:  Max gradient norm for clipping (prevents mode collapse).
        warmup_epochs:  Linear LR warm-up before cosine schedule kicks in.
    """
    tokenizer = PlateTokenizer(settings.charset)
    train_loader = build_dataloader(
        csv_path=settings.artifact_dir / "splits" / "train.csv",
        tokenizer=tokenizer,
        image_size=settings.image_size,
        max_label_length=settings.max_label_length,
        training=True,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )
    val_loader = build_dataloader(
        csv_path=settings.artifact_dir / "splits" / "val.csv",
        tokenizer=tokenizer,
        image_size=settings.image_size,
        max_label_length=settings.max_label_length,
        training=False,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )

    device = torch.device(settings.device)
    model = ANPRModel(
        vocab_size=tokenizer.vocab_size,
        max_label_length=settings.max_label_length,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
    ).to(device)

    if resume and settings.model_path.exists():
        print(f"Resuming from checkpoint: {settings.model_path}")
        checkpoint = torch.load(settings.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing kicks in after warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)

    best_full_acc = -1.0
    best_char_acc = -1.0
    patience_counter = 0
    checkpoint_dir = ensure_dir(settings.model_path.parent)
    _ = checkpoint_dir
    metrics_file = settings.logs_dir / "train_metrics.jsonl"

    print(f"Training for up to {epochs} epochs | patience={patience} | device={device}")
    print(f"Backbone: ResNet18 (pretrained) | Transformer decoder | lr={lr}")

    for epoch in range(1, epochs + 1):
        # --- Linear warm-up ---
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * warmup_factor

        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            logits = model(image, label)
            loss = cross_entropy_sequence_loss(logits, label, tokenizer.pad_id)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping — prevents exploding gradients / mode collapse
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            train_loss += float(loss.item())
            num_batches += 1

        avg_train_loss = train_loss / max(1, num_batches)

        # Advance scheduler after warmup
        if epoch > warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # --- Validation ---
        model.eval()
        pred_texts: list[str] = []
        true_texts: list[str] = []

        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                pred_tokens = model.greedy_decode(image)
                pred_texts.extend(tokenizer.decode(row) for row in pred_tokens)
                true_texts.extend(batch["label_text"])

        char_acc = character_accuracy(pred_texts, true_texts)
        full_acc = full_plate_accuracy(pred_texts, true_texts)

        improved = False
        if full_acc > best_full_acc:
            improved = True
            best_full_acc = full_acc
            best_char_acc = char_acc
        elif full_acc == best_full_acc and char_acc > best_char_acc:
            improved = True
            best_char_acc = char_acc

        if improved:
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tokenizer_charset": settings.charset,
                    "max_label_length": settings.max_label_length,
                    "epoch": epoch,
                },
                settings.model_path,
            )
        else:
            patience_counter += 1

        payload = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_char_acc": round(char_acc, 5),
            "val_full_acc": round(full_acc, 5),
            "best_val_full_acc": round(best_full_acc, 5),
            "lr": round(current_lr, 8),
        }
        logger.info(payload)
        write_jsonl(metrics_file, payload)

        print(
            f"Epoch {epoch:3d}/{epochs} | loss={avg_train_loss:.4f} | "
            f"char_acc={char_acc:.4f} | full_acc={full_acc:.4f} | "
            f"best_full={best_full_acc:.4f} | best_char={best_char_acc:.4f} | lr={current_lr:.2e}"
            + (" ✓ saved" if improved else f" (patience {patience_counter}/{patience})")
        )

        # --- Early stopping ---
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    print(f"\nTraining complete. Best full plate accuracy: {best_full_acc:.4f} | Best char accuracy: {best_char_acc:.4f}")
    print(f"Checkpoint saved to: {settings.model_path}")
