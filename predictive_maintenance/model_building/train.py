#!/usr/bin/env python3
"""train.py - CI/CD training job: tune, track (MLflow), select, evaluate and save the best model.

Usage:
    python model_building/train.py            # full training run
    python model_building/train.py --check    # reload the saved model and verify its metrics
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_utils as mu  # noqa: E402

BASE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify the saved model instead of training")
    args = parser.parse_args()
    if args.check:
        mu.verify_saved_model(BASE)
    else:
        mu.run_pipeline(BASE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
