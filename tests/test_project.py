"""Small tests for feature engineering and data validation."""

import pandas as pd
from src.features import build_features


def test_feature_engineering_adds_expected_columns():
    raw = pd.DataFrame(
        {
            "Type": ["L"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [100],
            "Machine failure": [0],
            "TWF": [0],
        }
    )
    features = build_features(raw)
    assert "Temperature difference [K]" in features.columns
    assert "Torque-speed proxy" in features.columns
    assert "Machine failure" not in features.columns
    assert "TWF" not in features.columns


def test_feature_engineering_calculates_temperature_difference():
    raw = pd.DataFrame(
        {
            "Type": ["M"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [312.0],
            "Rotational speed [rpm]": [1200],
            "Torque [Nm]": [30.0],
            "Tool wear [min]": [50],
        }
    )
    features = build_features(raw)
    assert features.loc[0, "Temperature difference [K]"] == 12.0
