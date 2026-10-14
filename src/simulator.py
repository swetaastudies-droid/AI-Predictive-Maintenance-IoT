"""Generate a virtual sensor stream and score it with the trained model.

This is a software simulation, not a live connection to physical IoT hardware.
The scenario names describe how readings are generated; they are not ground-truth
failure labels. The classifier supplies the predicted label and probability.
"""

from pathlib import Path
import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import PROJECT_ROOT
from src.features import build_features

MODEL_PATH = PROJECT_ROOT / "models" / "failure_model.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "simulation"


def generate_virtual_stream(steps_per_scenario: int = 60, seed: int = 42) -> pd.DataFrame:
    """Create artificial sensor readings for three scenarios.

    Readings are intentionally illustrative and should not be interpreted as
    a validated physical model of any particular machine.
    """
    if steps_per_scenario < 2:
        raise ValueError("steps_per_scenario must be at least 2.")

    rng = np.random.default_rng(seed)
    rows = []

    scenarios = ["Healthy operation", "Gradual degradation", "Sudden stress"]

    for scenario in scenarios:
        for step in range(steps_per_scenario):
            progress = step / (steps_per_scenario - 1)

            # Start with plausible illustrative baseline values.
            air_temp = 300.0 + rng.normal(0, 0.35)
            process_temp = 310.0 + rng.normal(0, 0.40)
            rpm = 1500.0 + rng.normal(0, 25)
            torque = 40.0 + rng.normal(0, 1.8)
            tool_wear = 35.0 + progress * 100 + rng.normal(0, 2.0)

            if scenario == "Gradual degradation":
                # Illustrative slow drift: warmer operation, more torque and wear.
                air_temp += 1.8 * progress
                process_temp += 3.0 * progress
                rpm -= 100.0 * progress
                torque += 10.0 * progress
                tool_wear += 70.0 * progress

            elif scenario == "Sudden stress":
                # Keep early readings near baseline, then add a sudden disturbance.
                if progress >= 0.65:
                    stress = (progress - 0.65) / 0.35
                    air_temp += 1.5 * stress
                    process_temp += 3.5 * stress
                    rpm -= 180.0 * stress
                    torque += 18.0 * stress
                    tool_wear += 15.0 * stress

            rows.append({
                "scenario": scenario,
                "step": step + 1,
                "simulated_time_minutes": step,
                "Type": rng.choice(["L", "M", "H"], p=[0.6, 0.3, 0.1]),
                "Air temperature [K]": max(270.0, air_temp),
                "Process temperature [K]": max(270.0, process_temp),
                "Rotational speed [rpm]": max(1.0, rpm),
                "Torque [Nm]": max(0.1, torque),
                "Tool wear [min]": max(0.0, tool_wear),
            })

    return pd.DataFrame(rows)


def run_simulation(steps_per_scenario: int = 60) -> pd.DataFrame:
    """Generate readings, predict failure risk, and save CSVs and charts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `python -m src.train` first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stream = generate_virtual_stream(steps_per_scenario=steps_per_scenario)

    # Use the same feature engineering and fitted model pipeline as training.
    model = joblib.load(MODEL_PATH)
    X = build_features(stream)
    stream["predicted_failure"] = model.predict(X).astype(int)
    stream["failure_probability"] = model.predict_proba(X)[:, 1]
    stream["predicted_status"] = stream["predicted_failure"].map({
        0: "No failure predicted",
        1: "Failure predicted",
    })

    # Save every generated sensor reading and corresponding prediction.
    csv_path = OUTPUT_DIR / "virtual_sensor_stream.csv"
    stream.to_csv(csv_path, index=False)

    # Graph 1: probability trend over simulated time for each scenario.
    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=stream,
        x="step",
        y="failure_probability",
        hue="scenario",
        errorbar=None,
    )
    plt.title("Model-Estimated Failure Probability During Virtual Simulation")
    plt.xlabel("Simulation step")
    plt.ylabel("Estimated failure probability")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_probability_trend.png", dpi=160)
    plt.close()

    # Graph 2: selected sensor trends, separated by scenario.
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    sensor_columns = [
        ("Process temperature [K]", "Process temperature (K)"),
        ("Rotational speed [rpm]", "Rotational speed (RPM)"),
        ("Torque [Nm]", "Torque (Nm)"),
    ]
    for ax, (column, label) in zip(axes, sensor_columns):
        sns.lineplot(
            data=stream,
            x="step",
            y=column,
            hue="scenario",
            errorbar=None,
            ax=ax,
        )
        ax.set_ylabel(label)
        ax.legend(title="Scenario")
    axes[-1].set_xlabel("Simulation step")
    fig.suptitle("Virtual Sensor Readings by Scenario")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sensor_trends.png", dpi=160)
    plt.close(fig)

    # Graph 3: mean predicted probability for each generated scenario.
    summary = (
        stream.groupby("scenario", as_index=False)["failure_probability"]
        .mean()
        .sort_values("failure_probability", ascending=False)
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=summary,
        x="scenario",
        y="failure_probability",
    )
    plt.title("Average Model-Estimated Failure Probability by Scenario")
    plt.xlabel("Virtual scenario")
    plt.ylabel("Average estimated probability")
    plt.ylim(0, 1)
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scenario_probability_comparison.png", dpi=160)
    plt.close()

    # Compact summary: model outputs only; scenario is not a true failure label.
    summary["interpretation_note"] = (
        "Model estimate on artificial readings; not a confirmed failure rate."
    )
    summary.to_csv(OUTPUT_DIR / "scenario_summary.csv", index=False)

    print("Virtual simulation completed.")
    print(f"Rows generated: {len(stream)}")
    print(f"CSV: {csv_path}")
    print(f"Charts and summary: {OUTPUT_DIR}")
    print("\nPredicted class counts by scenario:")
    print(pd.crosstab(stream["scenario"], stream["predicted_status"]))
    return stream





def test_historical_failures(sample_size=20):
    """Test the model on failure-labelled dataset records."""
    import pandas as pd
    import joblib

    from src.data import PROJECT_ROOT
    from src.features import build_features

    dataset_path = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
    model_path = PROJECT_ROOT / "models" / "failure_model.joblib"
    output_dir = PROJECT_ROOT / "outputs" / "simulation"

    # Check that required files exist.
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python -m src.train first."
        )

    df = pd.read_csv(dataset_path)
    model = joblib.load(model_path)

    # Select records labelled as failures.
    failures = df[df["Machine failure"] == 1].head(sample_size).copy()

    if failures.empty:
        raise ValueError("No failure-labelled records found.")

    actual = failures["Machine failure"].astype(int)
    X = build_features(failures)

    failures["predicted_failure"] = model.predict(X).astype(int)
    failures["failure_probability"] = (
        model.predict_proba(X)[:, 1].round(4)
    )
    failures["prediction_correct"] = (
        failures["predicted_failure"] == actual
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "historical_failure_test.csv"
    failures.to_csv(output_path, index=False)

    detected = int(
        (
            (actual == 1)
            & (failures["predicted_failure"] == 1)
        ).sum()
    )

    print("\nHistorical failure diagnostic")
    print(f"Failure records examined: {len(failures)}")
    print(f"Failures predicted correctly: {detected}")
    print(f"Results saved to: {output_path}")
    print(
        failures[
            [
                "Machine failure",
                "predicted_failure",
                "failure_probability",
                "prediction_correct",
            ]
        ].to_string(index=False)
    )
def main():
    parser = argparse.ArgumentParser(
        description="Run virtual sensor simulation and diagnostic tests."
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Number of time steps per virtual scenario.",
    )

    parser.add_argument(
        "--historical-test",
        action="store_true",
        help="Test the model on labelled historical failure records.",
    )

    args = parser.parse_args()

    if args.historical_test:
        test_historical_failures()
    else:
        run_simulation(args.steps)


if __name__ == "__main__":
    main()
