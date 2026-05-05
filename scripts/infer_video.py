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
    parser = argparse.ArgumentParser(description="Run ANPR on video/webcam")
    parser.add_argument("--source", type=str, default="0")
    args = parser.parse_args()

    settings = get_settings()
    engine = ANPRInferenceEngine(
        model_path=settings.model_path,
        charset=settings.charset,
        image_size=settings.image_size,
        max_label_length=settings.max_label_length,
        device=settings.device,
    )

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pred = engine.process_frame(frame)
        text = f"{pred['plate_text']} | ID:{pred['track_id']} | {pred['zone']}"
        x1, y1, x2, y2 = pred["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("ANPR", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
