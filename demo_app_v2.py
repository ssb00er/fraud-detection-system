"""
Fraud Detection Project: Streamlit Demo UI (polished version)
A dashboard-style interface that loads a real transaction, sends it to the
FastAPI service (api_v2.py), and displays the fraud score, SHAP-based feature
contributions, and a plain-English LLM explanation.

IMPORTANT: Run the FastAPI service first, in a separate terminal:
    uvicorn api_v2:app --reload

Then run this app:
    streamlit run demo_app.py

Requires: pip install streamlit requests plotly
The .streamlit/config.toml file (same folder) sets the dark theme - keep it
alongside this script.
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from sqlalchemy import create_engine

from config import DB_CONFIG

API_URL = "http://127.0.0.1:8000/predict_explained"

FEATURE_COLS = (
    ["tx_time", "amount", "hour_of_day", "amount_percentile", "avg_amount_last_50_tx"]
    + [f"v{i}" for i in range(1, 29)]
)

ACCENT = "#7C3AED"       # purple - primary accent
DANGER = "#EF4444"       # red - fraud
SAFE = "#22C55E"         # green - legit
CARD_BG = "#1A1D29"
MUTED = "#9096A6"

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(circle at top left, #171A26 0%, #0F1117 60%);
    }}

    .hero {{
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(135deg, #241B3D 0%, #17192B 100%);
        border: 1px solid #33304a;
        margin-bottom: 28px;
    }}
    .hero h1 {{
        font-size: 30px;
        margin-bottom: 4px;
        background: linear-gradient(90deg, #C084FC, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero p {{
        color: {MUTED};
        font-size: 15px;
        margin: 0;
    }}

    .section-label {{
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: {ACCENT};
        margin-bottom: 10px;
        margin-top: 6px;
    }}

    div[data-testid="stButton"] > button {{
        border-radius: 10px;
        border: 1px solid #3a3660;
        background: linear-gradient(135deg, #2A2547 0%, #1E1B33 100%);
        color: #E4E6EB;
        font-weight: 600;
        padding: 0.6em 1em;
        transition: all 0.15s ease-in-out;
    }}
    div[data-testid="stButton"] > button:hover {{
        border-color: {ACCENT};
        box-shadow: 0 0 12px rgba(124, 58, 237, 0.5);
        transform: translateY(-1px);
    }}

    /* Primary "Check Transaction" button gets a distinct gradient */
    div[data-testid="stButton"] > button[kind="primary"] {{
        background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%);
        border: none;
        color: white;
        font-size: 16px;
    }}
    div[data-testid="stButton"] > button[kind="primary"]:hover {{
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.7);
        transform: translateY(-2px);
    }}

    .metric-card {{
        background: {CARD_BG};
        border: 1px solid #2A2D3E;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
    }}
    .metric-card .label {{
        color: {MUTED};
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }}
    .metric-card .value {{
        font-size: 26px;
        font-weight: 700;
    }}

    .explanation-box {{
        background: linear-gradient(135deg, #1B2E22 0%, #16251C 100%);
        border-left: 4px solid {SAFE};
        border-radius: 10px;
        padding: 18px 20px;
        font-size: 15px;
        line-height: 1.6;
    }}
    .explanation-box.flagged {{
        background: linear-gradient(135deg, #2E1B1B 0%, #251616 100%);
        border-left: 4px solid {DANGER};
    }}

    .ground-truth-pill {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🛡️ Explainable Fraud Detection System</h1>
    <p>Real transaction scored by a tuned XGBoost model · explained with SHAP · summarized by an LLM</p>
</div>
""", unsafe_allow_html=True)


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


def make_gauge(probability, threshold):
    color = DANGER if probability >= threshold else SAFE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 42, "color": "#E4E6EB"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": CARD_BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 100], "color": "#212433"},
                {"range": [threshold * 100, 100], "color": "#2E1B1B"},
            ],
            "threshold": {
                "line": {"color": "#E4E6EB", "width": 3},
                "thickness": 0.9,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        font={"color": "#E4E6EB"},
    )
    return fig


if "transaction" not in st.session_state:
    st.session_state.transaction = None
if "result" not in st.session_state:
    st.session_state.result = None

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="section-label">Load a Transaction</div>', unsafe_allow_html=True)
    if st.button("🎲  Random transaction", use_container_width=True):
        st.session_state.transaction = fetch_random_transaction(fraud_only=False)
        st.session_state.result = None
    if st.button("🚩  Random known-fraud case", use_container_width=True):
        st.session_state.transaction = fetch_random_transaction(fraud_only=True)
        st.session_state.result = None
    st.markdown("---")
    st.caption(
        "The 'known-fraud' option pulls from labeled fraud rows in the dataset, "
        "for demo purposes. In production, the model would never see the true label."
    )

# ------------------------------------------------------------
# Main content
# ------------------------------------------------------------
if st.session_state.transaction is not None:
    tx = st.session_state.transaction

    st.markdown('<div class="section-label">Transaction Details</div>', unsafe_allow_html=True)
    st.caption("Loaded from the database — editable before submitting.")

    edited_values = {}
    cols = st.columns(4)
    for i, feature in enumerate(FEATURE_COLS):
        with cols[i % 4]:
            edited_values[feature] = st.number_input(
                feature, value=float(tx[feature]), format="%.4f", key=f"input_{feature}"
            )

    true_label = tx.get("class")
    if true_label is not None:
        pill_color = DANGER if true_label == 1 else SAFE
        label_text = "FRAUD" if true_label == 1 else "LEGITIMATE"
        st.markdown(
            f'Ground truth (hidden from model): '
            f'<span class="ground-truth-pill" style="background:{pill_color}22;color:{pill_color};">{label_text}</span>',
            unsafe_allow_html=True,
        )

    st.write("")
    if st.button("🔍  Check Transaction", type="primary", use_container_width=True):
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
    is_flagged = result["is_flagged"]

    st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

    col_gauge, col_metrics = st.columns([1.1, 1])

    with col_gauge:
        st.plotly_chart(
            make_gauge(result["fraud_probability"], result["threshold_used"]),
            use_container_width=True,
        )

    with col_metrics:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:14px;">
            <div class="label">Decision Threshold</div>
            <div class="value">{result['threshold_used']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        verdict_color = DANGER if is_flagged else SAFE
        verdict_text = "🚨 FLAGGED AS FRAUD" if is_flagged else "✅ LEGITIMATE"
        st.markdown(f"""
        <div class="metric-card" style="border-color:{verdict_color}55;">
            <div class="label">Model Verdict</div>
            <div class="value" style="color:{verdict_color};">{verdict_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-label">Why the Model Made This Decision</div>', unsafe_allow_html=True)
    contrib_df = pd.DataFrame(result["top_contributing_features"]).sort_values("shap_value")
    colors = [DANGER if v > 0 else SAFE for v in contrib_df["shap_value"]]

    fig_bar = go.Figure(go.Bar(
        x=contrib_df["shap_value"],
        y=contrib_df["feature"],
        orientation="h",
        marker_color=colors,
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E4E6EB"},
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#2A2D3E", zerolinecolor="#3A3D4E"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Red bars pushed the prediction toward fraud; green bars pushed away from it.")

    st.write("")
    st.markdown('<div class="section-label">Plain-English Explanation</div>', unsafe_allow_html=True)
    box_class = "explanation-box flagged" if is_flagged else "explanation-box"
    st.markdown(
        f'<div class="{box_class}">{result["plain_english_explanation"]}</div>',
        unsafe_allow_html=True,
    )

elif st.session_state.transaction is None:
    st.info("👈 Load a transaction from the sidebar to get started.")