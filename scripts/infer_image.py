from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anpr.config import get_settings
from anpr.inference.pipeline import ANPRInferenceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ANPR on a single image")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    settings = get_settings()
    engine = ANPRInferenceEngine(
        model_path=settings.model_path,
        charset=settings.charset,
        image_size=settings.image_size,
        max_label_length=settings.max_label_length,
        device=settings.device,
    )

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    result = engine.process_frame(image)
    print(result)


if __name__ == "__main__":
    main()
