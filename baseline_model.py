"""
Fraud Detection Project: Baseline Model
Trains a plain Logistic Regression model (no imbalance handling yet) to
establish a reference point before adding SMOTE/class-weighting and a
tuned XGBoost/LightGBM model in the next step.

Usage:
    python baseline_model.py
"""

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from config import DB_CONFIG


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


def load_features():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM transaction_features;", engine)
    print(f"Loaded {len(df):,} rows from transaction_features")
    return df


def main():
    df = load_features()

    # Drop columns that aren't model inputs
    feature_cols = [c for c in df.columns if c not in ("id", "class")]
    X = df[feature_cols]
    y = df["class"]

    # Stratified split so both train and test keep the same ~0.17% fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"Train fraud rate: {y_train.mean():.5f} | Test fraud rate: {y_test.mean():.5f}")

    # Scale features — logistic regression is sensitive to feature scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Plain logistic regression — NO imbalance handling on purpose.
    # This is our reference point to compare against once we add
    # class_weight/SMOTE and a tuned gradient-boosted model in Step 4.
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # ---- Evaluation ----
    # Deliberately NOT reporting accuracy as a headline metric — with ~0.17%
    # fraud rate, a model that predicts "legit" for everything would score
    # ~99.8% accuracy while catching zero fraud. Precision/Recall/F1/PR-AUC
    # are the metrics that actually matter here.
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)

    print("\n===== Baseline Model Results (Logistic Regression, no imbalance handling) =====")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nConfusion Matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, y_pred))

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    print(
        "\nNote: Recall is likely low here — the model is missing a large share "
        "of fraud cases because it hasn't been told fraud is rare-but-important. "
        "This motivates the imbalance handling (SMOTE / class_weight) in the next step."
    )


if __name__ == "__main__":
    main()