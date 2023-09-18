import json

import pandas as pd
from sklearn.model_selection import train_test_split

from metrics_and_plots import save_metrics, save_predictions, save_roc_curve
from model import evaluate_model, train_model
from utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def load_hyperparameters(hyperparameter_file):
    # Load hyperparameters from the JSON file
    with open(hyperparameter_file, "r") as json_file:
        hyperparameters = json.load(json_file)
    return hyperparameters


def main():
    X, y = load_data(PROCESSED_DATASET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    # Load hyperparameters from the JSON file
    hyperparameters = load_hyperparameters("rfc_best_params.json")
    model = train_model(X_train, y_train, hyperparameters)
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

    print("====================Test Set Metrics==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")

    save_metrics(metrics)
    save_predictions(y_test, y_pred)
    save_roc_curve(y_test, y_pred_proba)


if __name__ == "__main__":
    main()
