"""
Fraud Detection Project: Imbalance Handling + Tuned Model
Trains and compares three approaches to handling class imbalance:
  1. XGBoost with scale_pos_weight (class weighting)
  2. XGBoost with SMOTE oversampling (applied to training data only)
  3. A tuned version of the better-performing approach (small hyperparameter search)

All three are evaluated with the same metrics as the baseline
(Precision, Recall, F1, PR-AUC) for a direct before/after comparison.

Usage:
    python imbalance_model.py
"""

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

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


def evaluate(name, y_test, y_pred, y_proba, results):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n===== {name} =====")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("Confusion Matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, y_pred))

    results.append(
        {"model": name, "precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}
    )


def main():
    df = load_features()
    feature_cols = [c for c in df.columns if c not in ("id", "class")]
    X = df[feature_cols]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    results = []

    # ------------------------------------------------------------
    # Approach 1: XGBoost with class weighting (scale_pos_weight)
    # scale_pos_weight = (# negative) / (# positive) tells XGBoost to
    # treat each fraud example as this many times more important.
    # ------------------------------------------------------------
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = neg / pos
    print(f"\nComputed scale_pos_weight = {scale_pos_weight:.2f}")

    model_weighted = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model_weighted.fit(X_train, y_train)
    y_pred_w = model_weighted.predict(X_test)
    y_proba_w = model_weighted.predict_proba(X_test)[:, 1]
    evaluate("XGBoost + class_weight", y_test, y_pred_w, y_proba_w, results)

    # ------------------------------------------------------------
    # Approach 2: XGBoost with SMOTE oversampling
    # IMPORTANT: SMOTE is applied ONLY to the training set. Applying it
    # before the train/test split (or to the test set) would leak
    # synthetic information into evaluation and inflate the results
    # dishonestly.
    # ------------------------------------------------------------
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"\nAfter SMOTE, training class balance: {y_train_sm.value_counts().to_dict()}")

    model_smote = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=42,
    )
    model_smote.fit(X_train_sm, y_train_sm)
    y_pred_sm = model_smote.predict(X_test)
    y_proba_sm = model_smote.predict_proba(X_test)[:, 1]
    evaluate("XGBoost + SMOTE", y_test, y_pred_sm, y_proba_sm, results)

    # ------------------------------------------------------------
    # Approach 3: Hyperparameter tuning on whichever approach wins
    # so far, scored on PR-AUC (not accuracy).
    # We tune the class-weighted approach here as a reasonable default;
    # swap in the SMOTE-resampled data if SMOTE wins in your results.
    # ------------------------------------------------------------
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="average_precision",  # PR-AUC — matches our evaluation metric
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    print("\nRunning hyperparameter search (this may take a few minutes)...")
    search.fit(X_train, y_train)

    print(f"\nBest params found: {search.best_params_}")
    best_model = search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    y_proba_tuned = best_model.predict_proba(X_test)[:, 1]
    evaluate("XGBoost + class_weight (tuned)", y_test, y_pred_tuned, y_proba_tuned, results)

    # ------------------------------------------------------------
    # Summary comparison table
    # ------------------------------------------------------------
    results_df = pd.DataFrame(results)
    print("\n===== Summary: All Models Compared =====")
    print(results_df.to_string(index=False))
    print(
        "\nCompare these rows against the baseline Logistic Regression "
        "(Precision 0.8429, Recall 0.6020, F1 0.7024, PR-AUC 0.7431) "
        "to quantify the improvement from imbalance handling and tuning."
    )


if __name__ == "__main__":
    main()