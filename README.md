# AI-Powered Predictive Maintenance System for IoT Devices

A beginner-friendly machine-learning project that estimates whether an industrial machine is likely to report a failure from its operating measurements. It uses the public AI4I 2020 Predictive Maintenance dataset and does not require physical IoT hardware.

## Important limitations
- AI4I 2020 is a synthetic industrial-style dataset, not live readings from a real factory.
- The classifier predicts the dataset's `Machine failure` label for a row; it does not predict an exact failure time or remaining useful life.
- Model probabilities and alert thresholds are decision-support signals, not guarantees.
- Do not use this demonstration alone to make safety-critical maintenance decisions.

## Project structure
- `data/raw/`: place the downloaded CSV here.
- `src/data.py`: load and validate source data.
- `src/features.py`: engineer model features and prevent target leakage.
- `src/train.py`: train, evaluate, save model, metrics and charts.
- `src/predict.py`: predict failure/no failure for new rows.
- `main.py`: simple command-line entry point.
- `outputs/`: generated evaluation results and plots.
- `models/`: saved trained pipeline.

## Setup
Python 3.12 is recommended.

### Windows PowerShell
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
Download the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Place the CSV at:
`data/raw/ai4i2020.csv`

The expected source columns include `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, and `Machine failure`.

## Train and evaluate
From the project root:
```bash
python -m src.train
```
This creates:
- `models/failure_model.joblib`
- `outputs/metrics/model_metrics.json`
- `outputs/metrics/classification_report.txt`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/feature_importance.png`
- `outputs/figures/failure_label_counts.png`

The dataset is imbalanced, so accuracy alone is not sufficient. Review precision, recall, F1-score and the confusion matrix.

## Predict on new rows
Prepare a CSV with the six input columns shown below (the source column names are required):
`Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`

Then run:
```bash
python -m src.predict path/to/new_readings.csv
```
The output is written to `outputs/predictions/predictions.csv`.

## Tests
```bash
pytest -q
```


## Virtual simulation
After training the model, generate three artificial sensor scenarios:
```bash
python -m src.simulator
```
Optional: choose the number of time steps per scenario:
```bash
python -m src.simulator --steps 100
```
Outputs are saved in `outputs/simulation/`: the generated sensor stream, estimated failure-probability trend, sensor trends, scenario comparison and scenario summary.

The scenarios are illustrative software-generated readings, not actual hardware telemetry or ground-truth breakdown events. Treat probabilities as model estimates only.
