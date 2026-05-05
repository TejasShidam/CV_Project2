from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anpr.config import get_settings
from anpr.training.evaluate import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ANPR model")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    settings = get_settings()
    csv_override = Path(args.csv) if args.csv else None
    metrics = evaluate_model(settings, split=args.split, csv_override=csv_override)
    print(metrics)


if __name__ == "__main__":
    main()
