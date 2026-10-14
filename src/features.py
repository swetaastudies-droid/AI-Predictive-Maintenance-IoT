"""Feature engineering and target-leakage prevention."""

import numpy as np
import pandas as pd

TARGET_COLUMN = "Machine failure"

# Only measurements available before the target is known should be model inputs.
BASE_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model inputs from raw sensor measurements.

    Failure-mode labels and the overall target are deliberately excluded from
    features because they would leak the answer into the model.
    """
    missing = [column for column in BASE_FEATURES if column not in df.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))

    X = df[BASE_FEATURES].copy()

    # Convert numeric sensor fields safely.
    numeric_columns = BASE_FEATURES[1:]
    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    # Simple domain-inspired features. These are proxies, not physical diagnoses.
    X["Temperature difference [K]"] = (
        X["Process temperature [K]"] - X["Air temperature [K]"]
    )
    X["Torque-speed proxy"] = (
        X["Torque [Nm]"] * X["Rotational speed [rpm]"]
    )
    X["Wear per 1000 rpm"] = (
        X["Tool wear [min]"] / X["Rotational speed [rpm]"].replace(0, np.nan)
    ) * 1000

    # Leave missing values for the model pipeline's imputer.
    return X
