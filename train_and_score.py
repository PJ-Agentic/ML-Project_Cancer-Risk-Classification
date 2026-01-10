\
"""
Train + score a multi-class cancer risk model (Low/Medium/High) and output per-patient probabilities.

Constraints (per assignment):
- Do NOT use Overall_Risk_Score (Column R) as an input feature.
- Do NOT use Risk_Level (Column U) as an input feature (it's the target label).

Output:
- predicted_risk_level
- prob_<class> for each class
- cancer_risk_percent_high = P(High) * 100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report


def build_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """Build a preprocessing + model pipeline."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
    )

    return Pipeline(steps=[("prep", preprocess), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and score cancer risk model.")
    parser.add_argument("--input", type=str, default="data/cancer-risk-factors.xlsx", help="Path to input Excel file")
    parser.add_argument("--output", type=str, default="outputs/patient_cancer_risk_predictions.csv", help="Path to output CSV")
    parser.add_argument("--id-col", type=str, default="Patient_ID", help="Patient identifier column to keep in output (if present)")
    parser.add_argument("--target", type=str, default="Risk_Level", help="Target label column (Low/Medium/High)")
    parser.add_argument("--drop-feature", action="append", default=["Overall_Risk_Score"],
                        help="Feature columns to drop (can be provided multiple times). Defaults to Overall_Risk_Score.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(in_path)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Available columns: {list(df.columns)}")

    # Build features
    drop_cols = [c for c in args.drop_feature if c in df.columns]
    X = df.drop(columns=drop_cols + [args.target])
    y = df[args.target]

    # Identify categorical vs numeric
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    clf = build_pipeline(cat_cols=cat_cols, num_cols=num_cols)

    # Train/test split (stratified where possible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if y.nunique() > 1 else None,
    )

    clf.fit(X_train, y_train)

    # Evaluate
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    classes = clf.named_steps["model"].classes_

    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    ll = log_loss(y_test, proba, labels=classes)

    print("\n--- Test Metrics (Baseline Logistic Regression) ---")
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(f1m, 4))
    print("Log Loss:", round(ll, 4))
    print("\nClassification Report:\n", classification_report(y_test, pred))

    # Refit on full dataset for final scoring
    clf.fit(X, y)

    all_proba = clf.predict_proba(X)
    all_pred = clf.predict(X)
    classes = clf.named_steps["model"].classes_

    proba_df = pd.DataFrame(all_proba, columns=[f"prob_{c}" for c in classes])

    # Risk percent definition: P(High)*100 (only if 'High' exists)
    if "High" in classes:
        risk_percent_high = (proba_df["prob_High"] * 100).round(2)
    else:
        # Fallback: use the max probability class as a % if 'High' label isn't present
        risk_percent_high = (proba_df.max(axis=1) * 100).round(2)

    # Build output
    keep_cols = []
    if args.id_col in df.columns:
        keep_cols.append(args.id_col)

    output = pd.concat(
        [
            df[keep_cols] if keep_cols else pd.DataFrame(index=df.index),
            proba_df,
            pd.Series(all_pred, name="predicted_risk_level"),
            pd.Series(risk_percent_high, name="cancer_risk_percent_high"),
        ],
        axis=1,
    )

    output.to_csv(out_path, index=False)
    print(f"\nSaved predictions to: {out_path.resolve()}")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
