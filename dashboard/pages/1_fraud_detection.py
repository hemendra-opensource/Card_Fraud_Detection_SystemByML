"""
Page 1: Real-Time Fraud Detection
Allows users to input a single transaction and get an instant fraud probability score.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.utils import load_model
from src.feature_engineering import create_features

st.set_page_config(page_title="Fraud Detection · FraudShield AI", page_icon="🔍", layout="wide")

from dashboard.sidebar import render_sidebar
render_sidebar("pages/1_fraud_detection.py")

# ─── CSS (shared) ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg-primary:#0a0e1a;--bg-card:#1a2235;--bg-card-hover:#1f2a40;
    --accent-blue:#3b82f6;--accent-cyan:#06b6d4;--accent-purple:#8b5cf6;
    --accent-green:#10b981;--accent-red:#ef4444;--accent-orange:#f59e0b;
    --text-primary:#f1f5f9;--text-secondary:#94a3b8;--text-muted:#64748b;
    --border:rgba(148,163,184,0.12);
}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background-color:var(--bg-primary)!important;color:var(--text-primary)!important;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:linear-gradient(135deg,#0a0e1a 0%,#0d1b2a 50%,#0a0e1a 100%)!important;}
.main .block-container{padding:1.5rem 2.5rem!important;max-width:1600px!important;}
[data-testid="metric-container"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:16px!important;padding:1.2rem!important;transition:all 0.3s ease!important;}
[data-testid="metric-container"]:hover{border-color:var(--accent-blue)!important;transform:translateY(-2px)!important;box-shadow:0 8px 32px rgba(59,130,246,0.15)!important;}
[data-testid="stMetricLabel"]{color:var(--text-secondary)!important;font-size:0.8rem!important;font-weight:500!important;text-transform:uppercase!important;letter-spacing:0.05em!important;}
[data-testid="stMetricValue"]{color:var(--text-primary)!important;font-size:1.7rem!important;font-weight:700!important;}
.stButton>button{background:linear-gradient(135deg,var(--accent-blue),var(--accent-purple))!important;color:white!important;border:none!important;border-radius:12px!important;padding:0.7rem 2rem!important;font-weight:600!important;font-size:0.95rem!important;transition:all 0.3s ease!important;width:100%!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(59,130,246,0.4)!important;}
.stNumberInput>div>div>input,.stSelectbox>div>div{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text-primary)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117 0%,#111827 100%)!important;border-right:1px solid var(--border)!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-thumb{background:#475569;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_fraud_models():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model = load_model(os.path.join(base, "models/random_forest.pkl"))
    scaler = load_model(os.path.join(base, "models/scaler.pkl"))
    return model, scaler

with st.spinner("Loading fraud detection model..."):
    try:
        model, scaler = load_fraud_models()
        model_loaded = True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model_loaded = False

# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
    <div style="display:inline-flex;align-items:center;gap:0.5rem;padding:0.3rem 0.9rem;
                background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);
                border-radius:50px;margin-bottom:0.75rem;">
        <span style="font-size:0.78rem;color:#3b82f6;font-weight:600;letter-spacing:0.05em;">🔍 FRAUD DETECTION ENGINE</span>
    </div>
    <h1 style="margin:0;font-size:2rem;font-weight:800;
               background:linear-gradient(135deg,#f1f5f9,#3b82f6);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Real-Time Transaction Scoring
    </h1>
    <p style="margin:0.5rem 0 0;color:#94a3b8;font-size:0.95rem;">
        Enter transaction details to get an instant fraud probability score from the Random Forest model.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Layout ───────────────────────────────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown("""
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:20px;padding:1.75rem;">
        <h3 style="margin:0 0 1.25rem;font-size:1rem;font-weight:700;color:#3b82f6;">
            📝 Transaction Details
        </h3>
    """, unsafe_allow_html=True)

    tx_type = st.selectbox(
        "Transaction Type",
        ["TRANSFER", "CASH_OUT"],
        help="Only TRANSFER and CASH_OUT transactions can be fraudulent in PaySim."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        step = st.number_input("Step (Hour)", min_value=1, max_value=743, value=1,
                                help="Time step — 1 hour per step (max 743 hours = ~31 days)")
        old_balance_orig = st.number_input("Sender's Old Balance ($)", min_value=0.0, value=10000.0,
                                            step=100.0, format="%.2f")
        new_balance_orig = st.number_input("Sender's New Balance ($)", min_value=0.0, value=0.0,
                                            step=100.0, format="%.2f")

    with col_b:
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=10000.0,
                                  step=100.0, format="%.2f")
        old_balance_dest = st.number_input("Receiver's Old Balance ($)", min_value=0.0, value=0.0,
                                            step=100.0, format="%.2f")
        new_balance_dest = st.number_input("Receiver's New Balance ($)", min_value=0.0, value=0.0,
                                            step=100.0, format="%.2f")

    name_orig = st.text_input("Sender ID (nameOrig)", value="C1234567890",
                               help="Customer ID starting with C")
    name_dest = st.text_input("Receiver ID (nameDest)", value="C9876543210",
                               help="Receiver ID starting with C (customer) or M (merchant)")

    st.markdown("</div>", unsafe_allow_html=True)

    run_btn = st.button("🚀 Analyze Transaction", use_container_width=True)

# ─── Sample Presets ───────────────────────────────────────────────────────────
with col_form:
    st.markdown("""
    <div style="background:rgba(59,130,246,0.06);border:1px dashed rgba(59,130,246,0.3);
                border-radius:14px;padding:1rem;margin-top:1rem;">
        <p style="margin:0 0 0.5rem;font-size:0.8rem;font-weight:600;color:#3b82f6;">
            💡 Quick Test Scenarios
        </p>
        <p style="margin:0;font-size:0.75rem;color:#64748b;line-height:1.6;">
            <strong style="color:#ef4444;">Fraud Pattern:</strong> TRANSFER, amount = old balance, new balance = 0, receiver balance unchanged<br>
            <strong style="color:#10b981;">Legit Pattern:</strong> CASH_OUT, small amount, balances update correctly
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Results Panel ────────────────────────────────────────────────────────────
with col_result:
    if run_btn and model_loaded:
        # Build input DataFrame
        row = pd.DataFrame([{
            'step': step,
            'type': tx_type,
            'amount': amount,
            'nameOrig': name_orig,
            'nameDest': name_dest,
            'oldbalanceOrg': old_balance_orig,
            'newbalanceOrig': new_balance_orig,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'isFraud': 0,
            'isFlaggedFraud': 0
        }])

        # Feature engineering
        row_feat = create_features(row)
        row_feat['type_TRANSFER'] = (row_feat['type'] == 'TRANSFER').astype(int)
        row_feat['type_CASH_OUT'] = (row_feat['type'] == 'CASH_OUT').astype(int)

        feature_cols = [
            'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'balance_difference_orig',
            'balance_difference_dest', 'transaction_amount_ratio',
            'dest_is_merchant', 'transaction_velocity',
            'type_TRANSFER', 'type_CASH_OUT'
        ]

        X_input = row_feat[feature_cols].copy().fillna(0)
        numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                        'oldbalanceDest', 'newbalanceDest', 'balance_difference_orig',
                        'balance_difference_dest', 'transaction_amount_ratio', 'transaction_velocity']
        X_input[numeric_cols] = scaler.transform(X_input[numeric_cols])

        # Predict
        fraud_prob = model.predict_proba(X_input)[0][1]
        fraud_score = round(fraud_prob * 100, 2)
        prediction = model.predict(X_input)[0]

        # Risk Category
        if fraud_score < 30:
            risk_cat = "Low Risk"
            risk_color = "#10b981"
            risk_bg = "rgba(16,185,129,0.1)"
            risk_border = "rgba(16,185,129,0.4)"
            risk_icon = "✅"
            action = "Transaction appears legitimate. No action required."
        elif fraud_score < 70:
            risk_cat = "Medium Risk"
            risk_color = "#f59e0b"
            risk_bg = "rgba(245,158,11,0.1)"
            risk_border = "rgba(245,158,11,0.4)"
            risk_icon = "⚠️"
            action = "Suspicious activity detected. Flag for manual review."
        else:
            risk_cat = "High Risk — FRAUD DETECTED"
            risk_color = "#ef4444"
            risk_bg = "rgba(239,68,68,0.1)"
            risk_border = "rgba(239,68,68,0.4)"
            risk_icon = "🚨"
            action = "HIGH FRAUD PROBABILITY. Block transaction immediately."

        # ── Score Gauge ──
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fraud_score,
            title={"text": "Fraud Probability Score", "font": {"size": 16, "color": "#94a3b8", "family": "Inter"}},
            number={"suffix": "%", "font": {"size": 42, "color": risk_color, "family": "Inter"}},
            delta={"reference": 50, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569",
                         "tickfont": {"color": "#64748b", "size": 10}},
                "bar": {"color": risk_color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(16,185,129,0.15)"},
                    {"range": [30, 70], "color": "rgba(245,158,11,0.15)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.15)"}
                ],
                "threshold": {"line": {"color": risk_color, "width": 3}, "thickness": 0.8, "value": fraud_score}
            }
        ))
        fig_gauge.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=30, t=40, b=20),
            font=dict(family="Inter")
        )

        # Display Results
        st.markdown(f"""
        <div style="background:var(--bg-card);border:2px solid {risk_border};
                    border-radius:20px;padding:1.75rem;margin-bottom:1rem;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
                <h3 style="margin:0;font-size:1rem;font-weight:700;color:#f1f5f9;">🎯 Analysis Result</h3>
                <span style="padding:0.4rem 1rem;border-radius:20px;font-size:0.82rem;font-weight:700;
                             background:{risk_bg};color:{risk_color};border:1px solid {risk_border};">
                    {risk_icon} {risk_cat}
                </span>
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""
            <div style="padding:1rem;background:{risk_bg};border-radius:12px;
                        border-left:4px solid {risk_color};margin-top:0.5rem;">
                <p style="margin:0;font-size:0.85rem;font-weight:600;color:{risk_color};">{risk_icon} Recommended Action</p>
                <p style="margin:0.3rem 0 0;font-size:0.82rem;color:#94a3b8;">{action}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Feature Breakdown ──
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1.5rem;margin-top:0.75rem;">
            <h4 style="margin:0 0 1rem;font-size:0.9rem;font-weight:700;color:#3b82f6;">
                📊 Engineered Feature Values
            </h4>
        """, unsafe_allow_html=True)

        bal_diff_orig = old_balance_orig - new_balance_orig - amount
        bal_diff_dest = new_balance_dest - old_balance_dest - amount
        tx_ratio = amount / old_balance_orig if old_balance_orig > 0 else amount
        dest_merchant = 1 if name_dest.startswith('M') else 0

        feature_display = [
            ("balance_difference_orig", f"{bal_diff_orig:,.2f}", "Sender balance anomaly"),
            ("balance_difference_dest", f"{bal_diff_dest:,.2f}", "Receiver update anomaly"),
            ("transaction_amount_ratio", f"{tx_ratio:.4f}", "Amount as % of balance"),
            ("dest_is_merchant", str(dest_merchant), "1=Merchant, 0=Customer"),
        ]

        for feat, val, desc in feature_display:
            color = "#ef4444" if (feat == "balance_difference_orig" and bal_diff_orig == 0) or \
                                  (feat == "transaction_amount_ratio" and tx_ratio >= 0.99) \
                    else "#f1f5f9"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:0.6rem 0.8rem;background:rgba(148,163,184,0.05);
                        border-radius:8px;margin-bottom:0.4rem;">
                <div>
                    <span style="font-size:0.8rem;font-weight:600;color:#94a3b8;font-family:'JetBrains Mono',monospace;">{feat}</span>
                    <span style="margin-left:0.5rem;font-size:0.72rem;color:#475569;">({desc})</span>
                </div>
                <span style="font-size:0.85rem;font-weight:700;color:{color};font-family:'JetBrains Mono',monospace;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    elif not model_loaded:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
                    border-radius:16px;padding:2rem;text-align:center;">
            <p style="font-size:2rem;margin:0;">⚠️</p>
            <p style="color:#ef4444;font-weight:600;margin:0.5rem 0 0;">Model not loaded</p>
            <p style="color:#94a3b8;font-size:0.85rem;margin:0.3rem 0 0;">
                Run the training pipeline first: <code>python -m src.train_pipeline</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:var(--bg-card);border:1px dashed rgba(148,163,184,0.2);
                    border-radius:20px;padding:3rem;text-align:center;height:100%;">
            <div style="font-size:3.5rem;margin-bottom:1rem;">🔍</div>
            <h3 style="margin:0;font-size:1.1rem;font-weight:700;color:#94a3b8;">
                Awaiting Transaction Input
            </h3>
            <p style="margin:0.5rem 0 0;color:#475569;font-size:0.85rem;">
                Fill in the transaction details on the left<br>and click <strong>Analyze Transaction</strong>.
            </p>
            <div style="margin-top:2rem;padding:1rem;background:rgba(59,130,246,0.06);
                        border-radius:12px;border:1px dashed rgba(59,130,246,0.2);">
                <p style="margin:0;font-size:0.8rem;color:#64748b;">
                    Risk thresholds:<br>
                    <span style="color:#10b981;">● Low Risk: 0–30</span> &nbsp;
                    <span style="color:#f59e0b;">● Medium Risk: 30–70</span> &nbsp;
                    <span style="color:#ef4444;">● High Risk: 70–100</span>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Batch Scoring Section ─────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<h3 style="font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:0.75rem;">
    📤 Batch Transaction Scoring
</h3>
""", unsafe_allow_html=True)

with st.expander("Upload a CSV file to score multiple transactions at once"):
    uploaded = st.file_uploader(
        "Upload CSV with columns: step, type, amount, nameOrig, nameDest, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest",
        type=["csv"]
    )
    if uploaded is not None and model_loaded:
        batch_df = pd.read_csv(uploaded)
        required_cols = ['step','type','amount','nameOrig','nameDest',
                         'oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
        missing_cols = [c for c in required_cols if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            with st.spinner("Scoring transactions..."):
                # Filter to TRANSFER/CASH_OUT only
                batch_df = batch_df[batch_df['type'].isin(['TRANSFER','CASH_OUT'])].copy()
                if 'isFraud' not in batch_df.columns:
                    batch_df['isFraud'] = 0
                if 'isFlaggedFraud' not in batch_df.columns:
                    batch_df['isFlaggedFraud'] = 0

                batch_feat = create_features(batch_df)
                batch_feat['type_TRANSFER'] = (batch_feat['type'] == 'TRANSFER').astype(int)
                batch_feat['type_CASH_OUT'] = (batch_feat['type'] == 'CASH_OUT').astype(int)

                feature_cols = ['step','amount','oldbalanceOrg','newbalanceOrig',
                                'oldbalanceDest','newbalanceDest','balance_difference_orig',
                                'balance_difference_dest','transaction_amount_ratio',
                                'dest_is_merchant','transaction_velocity',
                                'type_TRANSFER','type_CASH_OUT']
                numeric_cols = ['step','amount','oldbalanceOrg','newbalanceOrig',
                                'oldbalanceDest','newbalanceDest','balance_difference_orig',
                                'balance_difference_dest','transaction_amount_ratio','transaction_velocity']

                X_batch = batch_feat[feature_cols].fillna(0)
                X_batch[numeric_cols] = scaler.transform(X_batch[numeric_cols])

                probs = model.predict_proba(X_batch)[:, 1]
                batch_df['fraud_score'] = np.round(probs * 100, 2)
                batch_df['risk_category'] = pd.cut(
                    batch_df['fraud_score'],
                    bins=[-1, 30, 70, 101],
                    labels=['Low Risk', 'Medium Risk', 'High Risk']
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Scored", f"{len(batch_df):,}")
            c2.metric("High Risk", f"{(batch_df['risk_category']=='High Risk').sum():,}",
                      f"{(batch_df['risk_category']=='High Risk').mean()*100:.1f}%")
            c3.metric("Avg Fraud Score", f"{batch_df['fraud_score'].mean():.1f}%")

            st.dataframe(
                batch_df[['nameOrig','nameDest','type','amount','fraud_score','risk_category']].head(100),
                use_container_width=True
            )

            csv_out = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Scored CSV", csv_out, "scored_batch.csv", "text/csv")
