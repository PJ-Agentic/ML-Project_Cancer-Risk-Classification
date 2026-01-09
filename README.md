# ML Project_Cancer Risk Classification
Cancer Risk Prediction (Python+scikit-learn). Builds baseline and advanced ML classifiers to predict patient cancer risk level (Low/Medium/High) from demographic, lifestyle, genetic, and environmental factors, and outputs per-patient probability (“risk %”) scores. Includes clean preprocessing pipeline, evaluation, and reproducible inference script.

# Cancer Risk Level Prediction (Low / Medium / High) + Probability Scores

Welcome!! This project builds machine learning models to classify cancer risk levels (by Low, Medium, High) using demographic, lifestyle, genetic, and environmental factors. The goal is to showcase a clear, systematic ML workflow, beginning from a simple and explicable baseline model which then progresses to slightly more complex models.

I am approaching this as a Program/Project Manager learning Python more seriously, and so I’m intentionally keeping the workflow structured, readable, and reproducible.

---

## Problem Statement

Given a dataset of hospital patients with multiple risk-related attributes, predict each patient’s _ Cancer Risk Level_:

- **Low**
- **Medium**
- **High**

In addition to the class label, produce a **percentage score** per patient. In this repo, the percentage is implemented as:

> **Risk % = P(High Risk) × 100**

This makes the output easier to interpret and sort (e.g., “Who looks most likely to be High risk?”).

---

## Dataset Notes/Target Definition

The spreadsheet contains columns **A through U** (21 columns in total). Two columns are specifically called out:

- `Overall_Risk_Score` (Column R)
- `Risk_Level` (Column U)

**Important constraints used for data input considerations:**
- Do **not** use `Overall_Risk_Score` as an input feature.
- Do **not** use `Risk_Level` as an input feature.

In this project:

- **Target (y):** `Risk_Level` (Low/Medium/High)
- **Dropped from features (X):** `Overall_Risk_Score

So we are predicting `Risk_Level` using all other patient factors, while explicitly avoiding leakage from the existing `Overall_Risk_Score`.

---

## Methodology:

### 1) Data Loading + Basic Checks
- Load Excel into pandas
- Identify categorical vs numeric columns
- Check missing values

### 2) Preprocessing Pipeline 
Using scikit-learn `Pipeline` + `ColumnTransformer`:

- Numeric:
  - median imputation
  - scaling (for models like Logistic Regression)
- Categorical:
  - most-frequent imputation
  - one-hot encoding

Reasoning: This prevents training/serving skew and makes it easy to re-run.

### 3) Baseline Model
**Logistic Regression (multiclass)**

Reasoning:
- It’s a strong baseline
- Fast
- Coefficients are easier to reason about than many black-box models

### 4) More Complex Model (Optional Comparison)
**Random Forest**

Reasoning:
- Captures non-linear relationships
- Handles mixed feature interactions well

### 5) Evaluation
On a stratified train/test split, report:
- Accuracy
- Macro F1 (better for multi-class balance)
- Log Loss (quality of probabilities)

Reasoning: Accuracy alone can hide poor performance on minority classes.

### 6) Inference Outputs
For every patient, output:
- `predicted_risk_level`
- `prob_Low`, `prob_Medium`, `prob_High`
- `cancer_risk_percent_high` = `prob_High * 100`

---

## Results / Summary:

Using a standard 80/20 stratified split with a baseline Logistic Regression pipeline, it'safe to assume the following based on observation:

- **Accuracy:** ~0.89
- **Macro F1:** ~0.885
- **Log Loss:** ~0.303

## Project Structure: 
.
├── data/
│   └── cancer-risk-factors.xlsx
├── notebooks/
│   └── exploration.ipynb
├── src/
│   └── train_and_score.py
├── outputs/
│   └── patient_cancer_risk_predictions.csv
├── requirements.txt
└── README.md


