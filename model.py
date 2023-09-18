import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils_and_constants import RFC_FOREST_DEPTH


def train_model(
    X_train,
    y_train,
    rfc_params={"max_depth": RFC_FOREST_DEPTH, "n_estimators": 5, "random_state": 1993},
):
    model = RandomForestClassifier(**rfc_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, float_precision=4):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    metrics = json.loads(
        json.dumps(metrics), parse_float=lambda x: round(float(x), float_precision)
    )

    return metrics, y_pred, y_proba
