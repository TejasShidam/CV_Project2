from __future__ import annotations

import albumentations as A


def build_train_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Rotate(limit=10, p=0.5),
            A.Normalize(),
        ]
    )


def build_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
            A.Normalize(),
        ]
    )
