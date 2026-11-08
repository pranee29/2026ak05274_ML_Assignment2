# Breast Cancer Classifier

## a. Problem statement

Train and compare five supervised classification models for predicting whether a
breast mass is malignant or benign. The saved models are evaluated on a held-out
test set, and a Streamlit app lets users compare a selected model on any
labeled CSV with the required features. The app loads trained pipelines and
does not retrain on uploaded data.

## b. Dataset description

- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic); accessed through scikit-learn's bundled copy of the original UCI data.
- **Rows:** 569 (455 training, 114 held-out test)
- **Features:** 30 numeric measurements computed from digitized fine needle aspirate images.
- **Target:** `target` — 0 = benign, 1 = malignant (binary classification).
- **Test data:** `test_data.csv` contains the stratified held-out test rows and target. The test rows are never used to fit the models.
- **Split:** 80/20 stratified train/test split, `random_state=42`.

## c. GitHub Repository Link

**Repository:** [https://github.com/pranee29/ML_Assignment2_2026ak05274](https://github.com/pranee29/ML_Assignment2_2026ak05274)

The project files in this repository include the app, training source,
requirements, saved model pipelines, test data, and this README.

## d. Models used

All metrics are measured on the same held-out test set. Precision, recall, and
F1 use malignant (`target=1`) as the positive class. AUC uses predicted
malignancy probabilities, or the decision score for SVM; MCC is the Matthews
correlation coefficient.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong performance, with the highest AUC (0.9960) and high precision (0.9750); it correctly detects 92.86% of malignant cases. |
| Decision Tree | The weakest overall holdout results (0.9298 accuracy and 0.8492 MCC); precision and recall are balanced at 0.9048. |
| KNN | Good performance (0.9561 accuracy, 0.9825 AUC); precision is high, but recall (0.9048) is lower than the linear and SVM models. |
| Naive Bayes | High ranking ability (0.9891 AUC), but its 0.8571 recall and 0.9211 accuracy are the lowest among these models. |
| Support Vector Machine | Best overall balance: highest accuracy (0.9737), precision (1.0000), F1 (0.9630), and MCC (0.9442), with 0.9286 recall. |
| **Overall, Winner for your dataset?** | **Support Vector Machine**, based on the strongest accuracy, precision, F1, and MCC. Logistic Regression has the highest AUC (0.9960), so it is a close alternative when ranking quality is the priority. |