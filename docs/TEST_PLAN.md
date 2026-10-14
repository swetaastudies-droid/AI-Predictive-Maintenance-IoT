# Test Plan

## Automated checks
Run `python -m pytest -q`.

- Feature engineering adds the expected derived features.
- The target label is not included in model inputs.
- Temperature difference is calculated correctly.

## Training verification
1. Confirm `data/raw/ai4i2020.csv` exists.
2. Run `python -m src.train`.
3. Confirm `models/failure_model.joblib` is created.
4. Confirm metrics and plots are created under `outputs/`.
5. Check that the printed classification report contains both classes.

## Prediction verification
1. Prepare a CSV with the six required measurement columns.
2. Run `python -m src.predict path/to/new_readings.csv`.
3. Confirm `outputs/predictions/predictions.csv` contains predicted labels and failure probabilities.

## Limitation
These tests validate basic software behaviour. They do not establish safety or real-world predictive performance.
