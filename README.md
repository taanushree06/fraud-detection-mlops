# Credit Card Fraud Detection - MLOps Project

![CI](https://github.com/taanushree06/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)

## Objective
Classify credit card transactions as FRAUDULENT or LEGITIMATE. The project handles class imbalance, compares three models, optimizes the decision threshold, tracks experiments with MLflow, tests the code with pytest, runs CI with GitHub Actions and serves predictions through a Streamlit app.

## Dataset
ULB / Kaggle "Credit Card Fraud Detection" dataset (European cardholders, 2013).
- 284,807 transactions, 31 columns
- Features: `Time`, `V1`-`V28` (anonymized PCA components), `Amount`. Target: `Class` (1 = fraud)
- 492 frauds vs 284,315 legitimate (0.17% fraud, about 578:1 imbalance)
- No missing values, 1,081 duplicate rows (dropped before splitting)

## ML approach
1. Data validation (`src/data_validation.py`): required columns, types, missing/infinite values, negative amounts, target values, class balance. Training stops if a critical check fails.
2. EDA (`src/eda.py`): charts saved to `reports/`.
3. Preprocessing (`src/preprocessing.py`): duplicates removed, stratified 60/20/20 train/validation/test split, StandardScaler on `Time` and `Amount` fitted on training data only.
4. Class imbalance: class weights (`class_weight` for Logistic Regression and Random Forest, `scale_pos_weight` for XGBoost). No resampling, so there is no leakage risk.
5. Models (`src/train.py`): Logistic Regression, Random Forest, XGBoost. Each is a pipeline of scaler + model.
6. Evaluation (`src/evaluate.py`): precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix.
7. Threshold optimization (`src/threshold.py`): 19 thresholds evaluated on validation data, the best F2 score (recall weighted twice as much as precision) is chosen, then checked once on the test set.

## Results
Validation set, threshold 0.5:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.0588 | 0.8511 | 0.1100 | 0.9711 | 0.6922 |
| Random Forest | 0.8861 | 0.7447 | 0.8092 | 0.9490 | 0.7836 |
| XGBoost | 0.9211 | 0.7447 | 0.8235 | 0.9698 | 0.8011 |

XGBoost was selected (best PR-AUC). Chosen threshold: **0.3**.
Test set at threshold 0.3: precision 0.940, recall 0.821, ROC-AUC 0.970, PR-AUC 0.842 (78 frauds caught, 5 false alarms, 17 missed).

Fraud is rare, so accuracy is misleading (always predicting "legitimate" gives 99.8%). Recall shows how many frauds are caught, precision shows how many alerts are real, and PR-AUC summarizes both.

## MLflow
Every model run logs parameters and metrics. The final model is registered as `fraud-detection-model` with the alias `production`.

python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

Open http://127.0.0.1:5000

## Saved model
`models/fraud_pipeline.joblib` contains preprocessing + model + threshold in one file. The app loads only this file.

## Application
Streamlit app (`app/app.py`) with 30 input fields named after the dataset columns, example buttons, prediction, fraud probability and the threshold used. A Monitoring tab shows prediction counts and probabilities logged to `logs/predictions.csv`.

## Run locally

pip install -r requirements.txt
python -m streamlit run app/app.py


## Retrain from scratch

pip install -r requirements-train.txt
python src/download_data.py
python src/data_validation.py
python src/train.py
python src/threshold.py
python src/register_model.py


## Tests and CI

python -m pytest -v

33 tests cover data validation, preprocessing, the saved model, input handling and monitoring. GitHub Actions (`.github/workflows/ci.yml`) runs them on every push.

## Project structure

src/ data_validation, preprocessing, train, evaluate, threshold, register_model, predict, monitoring
app/ Streamlit app
tests/ pytest tests and a 500-row sample dataset
models/ final saved pipeline and threshold
reports/ EDA charts, comparison table, threshold analysis
.github/ CI workflow

