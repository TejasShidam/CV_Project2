from __future__ import annotations

from pathlib import Path

import torch

from anpr.config import Settings
from anpr.data.dataset import build_dataloader
from anpr.models.anpr_model import ANPRModel
from anpr.tokenizer import PlateTokenizer
from anpr.training.metrics import character_accuracy, full_plate_accuracy, advanced_metrics


def evaluate_model(
    settings: Settings,
    split: str = "test",
    csv_override: Path | None = None,
) -> dict[str, float]:
    tokenizer = PlateTokenizer(settings.charset)

    loader = build_dataloader(
        csv_path=csv_override or (settings.artifact_dir / "splits" / f"{split}.csv"),
        tokenizer=tokenizer,
        image_size=settings.image_size,
        max_label_length=settings.max_label_length,
        training=False,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )

    checkpoint = torch.load(settings.model_path, map_location=settings.device)

    model = ANPRModel(
        vocab_size=tokenizer.vocab_size,
        max_label_length=settings.max_label_length,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
    ).to(settings.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pred_texts: list[str] = []
    true_texts: list[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(settings.device)
            pred_tokens = model.greedy_decode(images)
            pred_texts.extend(tokenizer.decode(row) for row in pred_tokens)
            true_texts.extend(batch["label_text"])

    print("\n--- Evaluation Prediction Samples ---")
    match_count = 0
    for i, (pred, truth) in enumerate(zip(pred_texts, true_texts)):
        if i < 10:  # Print first 10
            match_str = "SUCCESS" if pred == truth else "FAIL"
            print(f"Sample {i+1} | Ground Truth: {truth:10} | Prediction: {pred:10} | {match_str}")
        if pred == truth:
            match_count += 1
    print(f"Total Exact Matches: {match_count} / {len(true_texts)}")
    print("-------------------------------------\n")

    adv_metrics = advanced_metrics(pred_texts, true_texts)
    
    metrics = {
        "character_accuracy": character_accuracy(pred_texts, true_texts),
        "full_plate_accuracy": full_plate_accuracy(pred_texts, true_texts),
        **adv_metrics
    }
    return metrics
