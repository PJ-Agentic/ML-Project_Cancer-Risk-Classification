# Cancer Risk Level Prediction (Low / Medium / High) + Probability Scores

This project builds machine learning models to classify cancer risk levels (**Low**, **Medium**, **High**) using demographic, lifestyle, genetic, and environmental factors. The goal is to demonstrate a clear, methodical machine learning workflow—starting from a simple, interpretable baseline model and progressing to more complex models.

I’m coming at this as a Program/Project Manager learning Python more seriously, so the emphasis is on a clean, repeatable workflow and readable code.

---

## Problem Statement

Given a dataset of hospital patients with multiple risk-related attributes, predict each patient’s **Cancer Risk Level**:
- Low
- Medium
- High

In addition to the class label, produce a **percentage score** per patient. In this repo, the percentage is implemented as:

> **Cancer risk % = P(High Risk) × 100**

This provides an intuitive score for sorting/prioritization (e.g., “which patients look most likely to be High risk?”).

---

## Dataset Notes / Target Definition

The spreadsheet includes columns **A through U**. Two columns are explicitly called out:

- `Overall_Risk_Score` (Column R)
- `Risk_Level` (Column U)

**Constraint (per assignment):**
- `Overall_Risk_Score` is **not used** as a feature.
- `Risk_Level` is the **label** we predict, so it is not used as a feature either.

So:
- **Target (y):** `Risk_Level`
- **Features (X):** all other columns (excluding `Overall_Risk_Score`)

---
## Tools/Libraries
- Python
- pandas
- numpy
- sickit-learn
---
## Methodology 

### 1) Load + basic checks
- Read the Excel file into pandas
- Inspect column types and missing values

Reasoning: No model is trustworthy if the data is messy or inconsistent.

### 2) Preprocessing pipeline 
Built with scikit-learn `Pipeline` + `ColumnTransformer`:
- Numeric features: median imputation + scaling
- Categorical features: most-frequent imputation + one-hot encoding

Reasoning: This avoids training/serving mismatch and keeps preprocessing consistent.

### 3) Baseline model 
**Logistic Regression (multiclass)**

Reasoning:
- Strong baseline
- Fast and stable
- Easier to reason about than many black-box models

### 4) Evaluation
Using a stratified train/test split and reporting:
- Accuracy
- Macro F1 (more fair for multi-class imbalance)
- Log Loss (i.e. quality of probability estimates)

Reasoning: Accuracy alone can be misleading when class balance is uneven.

### 5) Output scoring
For each patient:
- `predicted_risk_level`
- `prob_Low`, `prob_Medium`, `prob_High`
- `cancer_risk_percent_high`

---

## Results / Summary

Running the baseline Logistic Regression pipeline typically yields strong performance on this dataset. The exact metrics may vary based on the random split and parameters, but the script prints:

- Accuracy
- Macro F1
- Log Loss
- Classification report

---

## Project Structure

<img width="343" height="253" alt="image" src="https://github.com/user-attachments/assets/0926f487-bdd0-4a46-9318-589a634083e3" />

---

## Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train + score
```bash
python src/train_and_score.py
```

### 3) Output
A CSV will be created at:
`outputs/patient_cancer_risk_predictions.csv`

---

## Notes / Next Improvements

- Add k-fold cross-validation and hyperparameter tuning
- Add model comparison (Random Forest / Gradient Boosting)
- Add probability calibration checks (reliability curves)
- Add feature importance and leakage sanity checks
- Add a short data dictionary (what each column means)
- Package this as a small CLI Tool or Streamlit App

