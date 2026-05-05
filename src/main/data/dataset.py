from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from anpr.cv.plate_localization import localize_plate_region
from anpr.data.transforms import build_eval_transforms, build_train_transforms
from anpr.tokenizer import PlateTokenizer


class PlateDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PlateTokenizer,
        image_size: int,
        max_label_length: int,
        training: bool,
        use_cv_localizer: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_label_length = max_label_length
        self.transforms = (
            build_train_transforms(image_size) if training else build_eval_transforms(image_size)
        )
        self.use_cv_localizer = use_cv_localizer

    def __len__(self) -> int:
        return len(self.df)

    def _safe_crop(self, image: np.ndarray, row: pd.Series) -> np.ndarray:
        h, w = image.shape[:2]
        xmin = int(max(0, min(w - 1, row["xmin"])))
        ymin = int(max(0, min(h - 1, row["ymin"])))
        xmax = int(max(0, min(w, row["xmax"])))
        ymax = int(max(0, min(h, row["ymax"])))

        if xmax <= xmin or ymax <= ymin:
            return image

        return image[ymin:ymax, xmin:xmax]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self._safe_crop(image, row)

        if self.use_cv_localizer:
            image = localize_plate_region(image)

        transformed = self.transforms(image=image)
        image_tensor = torch.tensor(transformed["image"], dtype=torch.float32).permute(2, 0, 1)

        text = str(row["plate_text"])
        label = self.tokenizer.encode(text, self.max_label_length)

        return {
            "image": image_tensor,
            "label": label,
            "label_text": text,
        }


def _collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[str]]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    texts = [str(item["label_text"]) for item in batch]
    return {"image": images, "label": labels, "label_text": texts}


def build_dataloader(
    csv_path: Path,
    tokenizer: PlateTokenizer,
    image_size: int,
    max_label_length: int,
    training: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    df = pd.read_csv(csv_path)
    dataset = PlateDataset(
        df=df,
        tokenizer=tokenizer,
        image_size=image_size,
        max_label_length=max_label_length,
        training=training,
        use_cv_localizer=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )
