from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Gaussian Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}


def build_models() -> dict[str, Pipeline]:
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(random_state=RANDOM_STATE),
    }
    return {
        name: Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])
        for name, estimator in estimators.items()
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = (dataset.target == 0).astype(int).rename("target")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    test_data = x_test.copy()
    test_data["target"] = y_test
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    metrics_by_model: dict[str, dict[str, float]] = {}
    model_files: dict[str, str] = {}
    for display_name, pipeline in build_models().items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        scores = (
            pipeline.predict_proba(x_test)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else pipeline.decision_function(x_test)
        )
        metrics_by_model[display_name] = {
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "AUC": float(roc_auc_score(y_test, scores)),
            "Precision": float(precision_score(y_test, predictions, zero_division=0)),
            "Recall": float(recall_score(y_test, predictions, zero_division=0)),
            "F1": float(f1_score(y_test, predictions, zero_division=0)),
            "MCC": float(matthews_corrcoef(y_test, predictions)),
        }
        filename = f"{MODEL_NAMES[display_name]}.joblib"
        joblib.dump(pipeline, MODEL_DIR / filename)
        model_files[display_name] = filename

    metrics = {
        "dataset_name": "Wisconsin Diagnostic Breast Cancer (UCI)",
        "dataset_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "source": "scikit-learn load_breast_cancer; original dataset from UCI",
        "instances": int(len(features)),
        "feature_count": int(features.shape[1]),
        "train_instances": int(len(x_train)),
        "test_instances": int(len(x_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "positive_class": "Malignant (target=1)",
        "metrics": metrics_by_model,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "dataset_name": metrics["dataset_name"],
        "feature_names": list(features.columns),
        "model_files": model_files,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Dataset: {metrics['dataset_name']} ({len(features)} rows, {features.shape[1]} features)")
    print(f"Train/test rows: {len(x_train)}/{len(x_test)}")
    print(pd.DataFrame(metrics_by_model).T.round(4).to_string())


if __name__ == "__main__":
    main()
