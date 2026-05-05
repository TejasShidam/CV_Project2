from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anpr.config import get_settings
from anpr.data.split import split_dataframe
from anpr.data.verify import verify_annotations
from anpr.utils.io import ensure_dir


def main() -> None:
    settings = get_settings()

    df = verify_annotations(settings.annotation_dir, settings.image_dir, settings.data_root)

    metadata_dir = ensure_dir(settings.artifact_dir)
    splits_dir = ensure_dir(settings.artifact_dir / "splits")

    df.to_csv(metadata_dir / "metadata.csv", index=False)

    valid_df = df[df["status"] == "ok"].copy().reset_index(drop=True)
    train_df, val_df, test_df = split_dataframe(
        valid_df,
        train_split=settings.train_split,
        val_split=settings.val_split,
        test_split=settings.test_split,
        seed=settings.seed,
    )

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"Verified annotations: {len(df)}")
    print(f"Valid samples: {len(valid_df)}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")


if __name__ == "__main__":
    main()
