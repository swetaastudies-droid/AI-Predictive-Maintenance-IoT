# Project Report: AI-Powered Predictive Maintenance

## Problem statement
Unexpected equipment failure can cause downtime and maintenance costs. This project demonstrates how machine operating measurements can be used to classify whether a record is associated with machine failure.

## Objectives
- Load and validate a public predictive-maintenance dataset.
- Engineer useful features from sensor measurements.
- Train a supervised machine-learning classifier.
- Evaluate the model using accuracy, precision, recall, F1-score and a confusion matrix.
- Save the trained model and use it to score new records.

## Dataset
The project uses the UCI AI4I 2020 Predictive Maintenance Dataset. It is a synthetic industrial-style dataset and should not be described as live readings from an actual factory.

## Methodology
1. Load and validate the CSV.
2. Separate input measurements from the target label.
3. Create derived features.
4. Split data into training and test sets.
5. Fit preprocessing and a Random Forest classifier.
6. Evaluate on the held-out test set.
7. Save the model, metrics, charts and test predictions.

## Evaluation
Insert the values from `outputs/metrics/model_metrics.json` after running the project. Do not publish invented or placeholder metrics as measured results.

## Limitations
The model predicts the dataset's failure label for a row. It does not predict exact time-to-failure, remaining useful life, or guarantee that a physical machine will fail. Real-world use would require representative operational data, validation, monitoring and qualified engineering oversight.
