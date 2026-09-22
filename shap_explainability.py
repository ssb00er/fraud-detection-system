"""
Fraud Detection Project: SHAP Explainability
Trains the finalized model (XGBoost + class_weight, threshold = 0.9866 chosen
in Step 4's threshold analysis), then:
  1. Saves the model + threshold to disk for reuse in the FastAPI service (Step 6).
  2. Computes global feature importance using SHAP (which features matter most overall).
  3. Computes local explanations for individual transactions (why THIS transaction
     was flagged) - this is what will feed into the LLM plain-English layer (Step 7).

Usage:
    python shap_explainability.py
"""

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import DB_CONFIG

FINAL_THRESHOLD = 0.9866  # chosen in Step 4's precision-recall threshold analysis
MODEL_PATH = "fraud_model.joblib"


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
    # Train and save the finalized model
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

    joblib.dump({"model": model, "threshold": FINAL_THRESHOLD, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"Saved model + threshold + feature list to {MODEL_PATH}")

    # ------------------------------------------------------------
    # SHAP explainer - TreeExplainer is fast and exact for tree models like XGBoost
    # ------------------------------------------------------------
    print("\nComputing SHAP values (this may take a minute on the full test set)...")
    explainer = shap.TreeExplainer(model)

    # Use a sample of the test set for the summary plot (faster, still representative)
    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # ------------------------------------------------------------
    # 1. Global feature importance (summary plot)
    # ------------------------------------------------------------
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches="tight")
    print("Saved shap_summary_plot.png")
    plt.close()

    # Bar version - easier to read the ranking at a glance
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Mean |SHAP value| by Feature")
    plt.tight_layout()
    plt.savefig("shap_importance_bar.png", dpi=150, bbox_inches="tight")
    print("Saved shap_importance_bar.png")
    plt.close()

    # ------------------------------------------------------------
    # 2. Local explanation - pick one actual fraud case flagged correctly
    # This is the kind of output that will later be handed to the LLM
    # to generate a plain-English explanation.
    # ------------------------------------------------------------
    y_proba_test = model.predict_proba(X_test)[:, 1]
    flagged_mask = (y_proba_test >= FINAL_THRESHOLD) & (y_test.values == 1)
    flagged_indices = np.where(flagged_mask)[0]

    if len(flagged_indices) > 0:
        idx = flagged_indices[0]
        instance = X_test.iloc[[idx]]
        instance_shap = explainer.shap_values(instance)

        print(f"\n===== Example Local Explanation (correctly flagged fraud case) =====")
        print(f"Model probability: {y_proba_test[idx]:.4f} (threshold: {FINAL_THRESHOLD})")

        # Build a sorted list of (feature, shap_value, feature_value) for this instance
        contributions = pd.DataFrame({
            "feature": feature_cols,
            "shap_value": instance_shap[0],
            "feature_value": instance.iloc[0].values,
        }).sort_values("shap_value", key=abs, ascending=False)

        print("\nTop contributing features (positive = pushed toward 'fraud'):")
        print(contributions.head(10).to_string(index=False))

        # Save this as a force plot too - useful visual for the README
        shap.force_plot(
            explainer.expected_value, instance_shap[0], instance,
            matplotlib=True, show=False
        )
        plt.savefig("shap_force_plot_example.png", dpi=150, bbox_inches="tight")
        print("Saved shap_force_plot_example.png")
        plt.close()
    else:
        print("\nNo transactions in the test set were flagged at this threshold - "
              "try lowering FINAL_THRESHOLD temporarily to find an example.")

    print(
        "\nNext step: the 'contributions' table above (feature, shap_value, feature_value) "
        "is exactly the structured data that will be passed to an LLM in Step 6/7 to "
        "generate a plain-English explanation for a fraud analyst."
    )


if __name__ == "__main__":
    main()