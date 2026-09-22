"""
Fraud Detection Project: Precision-Recall Curve & Threshold Analysis
Trains the final chosen model (XGBoost + class_weight, untuned - selected for
having the best PR-AUC among the approaches compared in Step 4) and analyzes
the precision/recall trade-off across different classification thresholds,
instead of blindly using the default 0.5 cutoff.

Usage:
    python threshold_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score
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


def main():
    df = load_features()
    feature_cols = [c for c in df.columns if c not in ("id", "class")]
    X = df[feature_cols]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------
    # Train the final chosen model: XGBoost + class_weight, untuned
    # (selected in Step 4 for having the best PR-AUC: 0.841)
    # ------------------------------------------------------------
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    print(f"Confirmed PR-AUC on test set: {pr_auc:.4f}")

    # ------------------------------------------------------------
    # Precision-Recall curve across all thresholds
    # ------------------------------------------------------------
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (PR-AUC = {pr_auc:.4f})")
    plt.grid(True)
    plt.legend()
    plt.savefig("precision_recall_curve.png", dpi=150, bbox_inches="tight")
    print("Saved chart to precision_recall_curve.png")
    plt.show()

    # ------------------------------------------------------------
    # Threshold analysis table
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1,
    # so we align them by dropping the last precision/recall point (which
    # corresponds to threshold = 1.0, recall = 0, not an actual threshold).
    # ------------------------------------------------------------
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

    threshold_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precisions[:-1],
        "recall": recalls[:-1],
        "f1": f1_scores,
    })

    # Show a spread of thresholds, not just every single one (there can be thousands)
    sampled = threshold_df.iloc[::max(1, len(threshold_df) // 20)]
    print("\n===== Threshold Analysis (sampled) =====")
    print(sampled.to_string(index=False))

    best_f1_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    print(f"\nThreshold that maximizes F1: {best_f1_row['threshold']:.4f}")
    print(f"  -> Precision: {best_f1_row['precision']:.4f}, Recall: {best_f1_row['recall']:.4f}, F1: {best_f1_row['f1']:.4f}")

    # Example: find the threshold needed to hit a specific recall target,
    # e.g., "we want to catch at least 90% of fraud" - a realistic business ask.
    target_recall = 0.90
    candidates = threshold_df[threshold_df["recall"] >= target_recall]
    if not candidates.empty:
        # Among thresholds hitting the recall target, pick the one with best precision
        best_for_target = candidates.loc[candidates["precision"].idxmax()]
        print(f"\nTo achieve >= {target_recall:.0%} recall:")
        print(f"  Threshold: {best_for_target['threshold']:.4f}")
        print(f"  -> Precision: {best_for_target['precision']:.4f}, Recall: {best_for_target['recall']:.4f}")
    else:
        print(f"\nNo threshold in this model achieves >= {target_recall:.0%} recall.")

    print(
        "\nNote: The default classification threshold is 0.5, which is what "
        "Step 4's evaluation used. This analysis shows how precision and recall "
        "trade off at other thresholds, so the final threshold can be chosen "
        "based on business priorities (e.g., minimizing missed fraud vs. "
        "minimizing false alarms for the review team) rather than an arbitrary default."
    )


if __name__ == "__main__":
    main()