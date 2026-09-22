"""
Fraud Detection Project: Streamlit Demo UI
A simple web interface that lets you pick a real transaction from the database,
send it to the FastAPI service (api_v2.py), and view the fraud score, SHAP-based
contributing features, and the plain-English LLM explanation.

IMPORTANT: Run the FastAPI service first, in a separate terminal:
    uvicorn api_v2:app --reload

Then run this app:
    streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import requests
from sqlalchemy import create_engine

from config import DB_CONFIG

API_URL = "http://127.0.0.1:8000/predict_explained"

FEATURE_COLS = (
    ["tx_time", "amount", "hour_of_day", "amount_percentile", "avg_amount_last_50_tx"]
    + [f"v{i}" for i in range(1, 29)]
)


@st.cache_resource
def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


def fetch_random_transaction(fraud_only=False):
    engine = get_engine()
    where_clause = "WHERE class = 1" if fraud_only else ""
    query = f"SELECT * FROM transaction_features {where_clause} ORDER BY RANDOM() LIMIT 1;"
    df = pd.read_sql(query, engine)
    return df.iloc[0]


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Fraud Detection Demo", page_icon="🛡️", layout="wide")
st.title("🛡️ Explainable Fraud Detection System")
st.caption(
    "Loads a real transaction, scores it with a tuned XGBoost model, explains the "
    "decision with SHAP, and summarizes it in plain English via an LLM."
)

if "transaction" not in st.session_state:
    st.session_state.transaction = None
if "result" not in st.session_state:
    st.session_state.result = None

# ------------------------------------------------------------
# Sidebar: load a transaction
# ------------------------------------------------------------
st.sidebar.header("1. Load a Transaction")
col_a, col_b = st.sidebar.columns(2)
if col_a.button("Random transaction"):
    st.session_state.transaction = fetch_random_transaction(fraud_only=False)
    st.session_state.result = None
if col_b.button("Random FRAUD case"):
    st.session_state.transaction = fetch_random_transaction(fraud_only=True)
    st.session_state.result = None

st.sidebar.markdown("---")
st.sidebar.caption(
    "'Random FRAUD case' pulls from known fraud labels in the dataset - useful for "
    "demoing what a correctly-caught fraud looks like. In a real deployment, of "
    "course, the model wouldn't have access to the true label."
)

# ------------------------------------------------------------
# Main area: show editable transaction + submit
# ------------------------------------------------------------
if st.session_state.transaction is not None:
    tx = st.session_state.transaction

    st.subheader("2. Transaction Details")
    st.caption("Values loaded from the database. You can edit them before submitting.")

    edited_values = {}
    cols = st.columns(4)
    for i, feature in enumerate(FEATURE_COLS):
        with cols[i % 4]:
            edited_values[feature] = st.number_input(
                feature, value=float(tx[feature]), format="%.4f", key=f"input_{feature}"
            )

    true_label = tx.get("class")
    if true_label is not None:
        st.info(f"Ground-truth label for this transaction (from dataset): "
                f"{'FRAUD' if true_label == 1 else 'Legitimate'} "
                f"(hidden from the model - shown here only for demo purposes)")

    if st.button("🔍 Check Transaction", type="primary"):
        with st.spinner("Scoring transaction and generating explanation..."):
            try:
                response = requests.post(API_URL, json=edited_values, timeout=30)
                response.raise_for_status()
                st.session_state.result = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the API. Make sure it's running: "
                    "`uvicorn api_v2:app --reload`"
                )
            except Exception as e:
                st.error(f"Request failed: {e}")

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
if st.session_state.result is not None:
    result = st.session_state.result

    st.subheader("3. Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
    col2.metric("Decision Threshold", f"{result['threshold_used']:.4f}")
    col3.metric("Flagged as Fraud?", "🚨 YES" if result["is_flagged"] else "✅ No")

    st.markdown("### Why the model made this decision")
    contrib_df = pd.DataFrame(result["top_contributing_features"])
    contrib_df = contrib_df.sort_values("shap_value")
    st.bar_chart(contrib_df.set_index("feature")["shap_value"])
    st.caption("Positive values pushed the prediction toward 'fraud'; negative values pushed away from it.")

    st.markdown("### Plain-English Explanation")
    st.success(result["plain_english_explanation"])

else:
    st.info("👈 Load a transaction from the sidebar to get started.")