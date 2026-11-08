from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from evaluation import TARGET_COLUMN, calculate_metrics, validate_data


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
LABELS = ["Benign", "Malignant"]


@st.cache_resource
def load_artifacts() -> tuple[dict, list[str], dict]:
    with (MODEL_DIR / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    models = {
        model_name: joblib.load(MODEL_DIR / filename)
        for model_name, filename in metadata["model_files"].items()
    }
    return models, metadata["feature_names"], metadata


st.set_page_config(page_title="Breast Cancer Classifier Lab", page_icon="🧪", layout="wide")
st.title("Breast Cancer Classifier Lab")
st.caption(
    "Compare five saved classifiers on labeled test data from the UCI Wisconsin "
    "Diagnostic Breast Cancer dataset. Models are loaded as trained; this app never retrains them."
)

models, feature_names, metadata = load_artifacts()
model_name = st.selectbox("Choose a classifier", list(models))
uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])

if uploaded_file is None:
    test_path = ROOT / "test_data.csv"
    st.info("Showing the bundled holdout test set. Upload another labeled CSV to evaluate it instead.")
    data = pd.read_csv(test_path)
else:
    try:
        data = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()

try:
    validated = validate_data(data, feature_names)
except ValueError as error:
    st.error(str(error))
    st.stop()

features = validated.loc[:, feature_names]
target = validated[TARGET_COLUMN].astype(int)
model = models[model_name]
metrics = calculate_metrics(model, features, target)
predictions = model.predict(features)

st.subheader(f"Evaluation: {model_name}")
metric_columns = st.columns(6)
for column, (metric, value) in zip(metric_columns, metrics.items()):
    column.metric(metric, f"{value:.4f}")
st.write(f"Evaluated **{len(validated):,}** rows.")

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    st.pyplot(figure)
    plt.close(figure)

with right:
    st.markdown("#### Classification report")
    report = classification_report(
        target,
        predictions,
        labels=[0, 1],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")

with st.expander("Dataset and model details"):
    st.write(f"Dataset: {metadata['dataset_name']}")
    st.write(f"Features required: {len(feature_names)}")
    st.write("Target encoding: 0 = benign, 1 = malignant")
    st.dataframe(validated.head(), width="stretch")
