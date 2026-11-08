from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


TARGET_COLUMN = "target"


def validate_data(data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if data.empty:
        raise ValueError("The uploaded CSV has no rows.")
    if data.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")

    required = [*feature_names, TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    features = data.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("Feature columns must contain numeric, non-missing values.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature columns must contain only finite numbers.")

    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("The target column must contain only 0 (benign) and 1 (malignant).")
    if target.nunique() < 2:
        raise ValueError("The test CSV must contain both target classes to calculate AUC.")

    return pd.concat([features, target.rename(TARGET_COLUMN)], axis=1)


def calculate_metrics(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    predictions = model.predict(features)
    scores = (
        model.predict_proba(features)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(features)
    )
    return {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": float(roc_auc_score(target, scores)),
        "Precision": float(precision_score(target, predictions, zero_division=0)),
        "Recall": float(recall_score(target, predictions, zero_division=0)),
        "F1": float(f1_score(target, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predictions)),
    }
