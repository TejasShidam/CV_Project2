from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anpr.config import get_settings
from anpr.training.train import train_model
from anpr.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ANPR model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.logs_dir / "train.log")
    train_model(settings=settings, epochs=args.epochs, lr=args.lr, resume=args.resume)


if __name__ == "__main__":
    main()
