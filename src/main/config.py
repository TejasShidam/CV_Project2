from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_root: Path
    image_dir: Path
    annotation_dir: Path
    train_split: float
    val_split: float
    test_split: float
    seed: int
    device: str
    batch_size: int
    num_workers: int
    image_size: int
    max_label_length: int
    model_path: Path
    artifact_dir: Path
    logs_dir: Path
    charset: str
    loss_type: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_root = Path(os.getenv("ANPR_DATA_ROOT", "Dataset"))
    image_dir = Path(
        os.getenv(
            "ANPR_IMAGE_DIR",
            str(data_root / "in" / "in"),
        )
    )
    annotation_dir = Path(
        os.getenv(
            "ANPR_ANNOTATION_DIR",
            str(data_root / "Annotations" / "Annotations"),
        )
    )
    train_split = float(os.getenv("ANPR_TRAIN_SPLIT", "0.70"))
    val_split = float(os.getenv("ANPR_VAL_SPLIT", "0.15"))
    test_split = float(os.getenv("ANPR_TEST_SPLIT", "0.15"))

    if abs((train_split + val_split + test_split) - 1.0) > 1e-6:
        raise ValueError("Train/val/test splits must sum to 1.0")

    artifact_dir = Path("artifacts")
    logs_dir = artifact_dir / "logs"

    return Settings(
        data_root=data_root,
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=int(os.getenv("ANPR_SEED", "42")),
        device=os.getenv("ANPR_DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu"),
        batch_size=int(os.getenv("ANPR_BATCH_SIZE", "16")),
        num_workers=int(os.getenv("ANPR_NUM_WORKERS", "0")),
        image_size=int(os.getenv("ANPR_IMAGE_SIZE", "224")),
        max_label_length=int(os.getenv("ANPR_MAX_LABEL_LEN", "16")),
        model_path=Path(
            os.getenv("ANPR_MODEL_PATH", str(artifact_dir / "checkpoints" / "best_model.pt"))
        ),
        artifact_dir=artifact_dir,
        logs_dir=logs_dir,
        charset=os.getenv("ANPR_CHARSET", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
        loss_type=os.getenv("ANPR_LOSS_TYPE", "ce"),
    )
