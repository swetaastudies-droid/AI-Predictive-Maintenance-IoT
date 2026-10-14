"""Simple command-line entry point for model training and evaluation."""

import argparse
from src.train import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-Powered Predictive Maintenance project"
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional path to the AI4I CSV. Defaults to data/raw/ai4i2020.csv.",
    )
    args = parser.parse_args()

    if args.data:
        train_and_evaluate(args.data)
    else:
        train_and_evaluate()


if __name__ == "__main__":
    main()
