"""Load a saved model and predict failure/no failure for a new CSV file."""

from pathlib import Path
import argparse

import joblib
import pandas as pd

from src.data import PROJECT_ROOT
from src.features import build_features

MODEL_PATH = PROJECT_ROOT / "models" / "failure_model.joblib"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "predictions" / "predictions.csv"


def predict_file(input_path: str | Path, model_path: str | Path = MODEL_PATH) -> pd.DataFrame:
    """Predict machine-failure labels and probabilities for rows in a CSV."""
    input_path = Path(input_path)
    model_path = Path(model_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run `python -m src.train` first."
        )

    raw = pd.read_csv(input_path)
    X = build_features(raw)
    model = joblib.load(model_path)

    predicted = model.predict(X)
    probability = model.predict_proba(X)[:, 1]

    result = raw.copy()
    result["predicted_failure"] = predicted.astype(int)
    result["predicted_status"] = result["predicted_failure"].map(
        {0: "No failure predicted", 1: "Failure predicted"}
    )
    result["failure_probability"] = probability.round(4)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(result[["predicted_status", "failure_probability"]].head(10).to_string(index=False))
    print(f"\nPredictions saved to: {OUTPUT_PATH}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict machine failure from a CSV of sensor measurements."
    )
    parser.add_argument(
        "input_csv",
        help="CSV containing Type, temperature, RPM, torque and tool-wear columns",
    )
    args = parser.parse_args()
    predict_file(args.input_csv)


if __name__ == "__main__":
    main()
