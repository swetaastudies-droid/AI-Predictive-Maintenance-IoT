"""Train a Random Forest classifier, evaluate it, and save artefacts."""

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")  # Allows chart saving in headless environments.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import DEFAULT_DATA_PATH, PROJECT_ROOT, load_dataset
from src.features import TARGET_COLUMN, build_features

MODEL_PATH = PROJECT_ROOT / "models" / "failure_model.joblib"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create preprocessing + classifier as one reusable pipeline."""
    categorical = ["Type"]
    numeric = [column for column in X.columns if column not in categorical]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def save_feature_importance(model: Pipeline, output_path: Path) -> None:
    """Save a feature-importance chart using transformed feature names."""
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]
    try:
        names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
        importance_df = pd.DataFrame(
            {"feature": names, "importance": importances}
        ).sort_values("importance", ascending=False).head(15)

        plt.figure(figsize=(10, 7))
        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
    except (AttributeError, ValueError) as exc:
        print(f"Could not create feature-importance chart: {exc}")


def train_and_evaluate(data_path: str | Path = DEFAULT_DATA_PATH) -> dict:
    """Train and evaluate the model; return the saved evaluation metrics."""
    df = load_dataset(data_path)
    X = build_features(df)
    y = df[TARGET_COLUMN]

    if y.nunique() < 2:
        raise ValueError("Training requires examples from both failure classes (0 and 1).")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(X_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset_rows_after_cleaning": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "failure_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_failure_class": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall_failure_class": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_failure_class": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    # ROC-AUC needs both classes in the test split; stratification normally ensures this.
    if y_test.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_probability))

    # Create output folders and persist the model and evaluation results.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    with (METRICS_DIR / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_test, y_pred, target_names=["No failure", "Failure"], zero_division=0
    )
    (METRICS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    # Confusion matrix chart.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No failure", "Failure"],
        yticklabels=["No failure", "Failure"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Count of labels in the full cleaned dataset.
    label_counts = y.value_counts().reindex([0, 1], fill_value=0)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["No failure", "Failure"], y=label_counts.values)
    plt.title("Failure vs No-Failure Records")
    plt.xlabel("Machine status")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_label_counts.png", dpi=160)
    plt.close()

    save_feature_importance(model, FIGURES_DIR / "feature_importance.png")

    # Save test predictions so the model's results can be inspected.
    prediction_output = X_test.copy()
    prediction_output["actual_failure"] = y_test
    prediction_output["predicted_failure"] = y_pred
    prediction_output["failure_probability"] = y_probability
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_output.to_csv(predictions_dir / "test_predictions.csv", index=False)

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_DIR / 'model_metrics.json'}")
    print(f"Charts saved to: {FIGURES_DIR}")
    print("\nEvaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("\nClassification report:\n")
    print(report)

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
