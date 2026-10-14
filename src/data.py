"""Dataset loading and basic validation for the AI4I predictive-maintenance project."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

# Source names used by the UCI AI4I 2020 dataset.
REQUIRED_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the CSV, check its required columns, and return a clean copy."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the UCI AI4I 2020 dataset and save its CSV as "
            "'data/raw/ai4i2020.csv'. See README.md for the official link."
        )

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing)
            + "\nCheck that you downloaded the original AI4I 2020 CSV."
        )

    # Remove duplicate rows and rows missing any required modelling values.
    df = df.drop_duplicates().dropna(subset=REQUIRED_COLUMNS).copy()

    # Keep only valid binary target rows.
    df["Machine failure"] = pd.to_numeric(df["Machine failure"], errors="coerce")
    df = df[df["Machine failure"].isin([0, 1])].copy()
    df["Machine failure"] = df["Machine failure"].astype(int)

    if df.empty:
        raise ValueError("No valid rows remain after basic data validation.")

    return df.reset_index(drop=True)
