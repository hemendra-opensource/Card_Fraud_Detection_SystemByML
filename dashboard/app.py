"""
FraudShield AI — Dashboard Entry Point
=======================================
Run with:  python -m streamlit run dashboard/app.py
           OR from project root: streamlit run dashboard/app.py

Streamlit multipage architecture:
  dashboard/app.py         ← this file (homepage)
  dashboard/pages/
    1_fraud_detection.py
    2_transaction_explorer.py
    3_customer_segmentation.py
    4_model_performance.py
"""

import sys
import os

# ── Ensure project root is on sys.path so 'src.*' imports work from any page ──
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI · Overview",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "**FraudShield AI** — End-to-End ML Fraud Detection & "
            "Customer Segmentation System\n\n"
            "Model: Random Forest · Dataset: PaySim 6.35M rows · "
            "ROC-AUC: 0.9991"
        ),
    },
)

# ─── Model Loading (cached — shared across pages via st.session_state) ────────
@st.cache_resource(show_spinner=False)
def _load_models():
    """Load fraud detection and segmentation models from models/."""
    from src.utils import load_model

    base = _PROJECT_ROOT
    paths = {
        "random_forest":  os.path.join(base, "models", "random_forest.pkl"),
        "scaler":         os.path.join(base, "models", "scaler.pkl"),
        "kmeans":         os.path.join(base, "models", "kmeans.pkl"),
        "scaler_kmeans":  os.path.join(base, "models", "scaler_kmeans.pkl"),
    }

    loaded, missing = {}, []
    for name, path in paths.items():
        if os.path.exists(path):
            loaded[name] = load_model(path)
        else:
            missing.append(name)

    return loaded, missing


# Try to load models silently; surface status in the sidebar
with st.spinner("Initialising FraudShield AI…"):
    try:
        _models, _missing = _load_models()
        _model_ok = len(_missing) == 0
    except Exception as _e:
        _models, _missing, _model_ok = {}, ["all"], False
        _load_error = str(_e)
    else:
        _load_error = None

# Stash in session_state so individual pages can reuse without reloading
if "models" not in st.session_state:
    st.session_state["models"] = _models
if "model_ok" not in st.session_state:
    st.session_state["model_ok"] = _model_ok

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ── */
:root {
    --bg-primary:   #0a0e1a;
    --bg-secondary: #111827;
    --bg-card:      #1a2235;
    --bg-hover:     #1f2a40;
    --blue:         #3b82f6;
    --cyan:         #06b6d4;
    --purple:       #8b5cf6;
    --green:        #10b981;
    --red:          #ef4444;
    --amber:        #f59e0b;
    --text:         #f1f5f9;
    --muted:        #94a3b8;
    --faint:        #64748b;
    --border:       rgba(148,163,184,.12);
    --glow-blue:    rgba(59,130,246,.15);
    --glow-purple:  rgba(139,92,246,.15);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%) !important;
}
.main .block-container {
    padding: 1.5rem 2.5rem !important;
    max-width: 1600px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

/* ── Sidebar nav links ── */
[data-testid="stSidebarNav"] a {
    color: var(--muted) !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s ease !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: var(--glow-blue) !important;
    color: var(--blue) !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: linear-gradient(135deg, var(--glow-blue), var(--glow-purple)) !important;
    color: var(--cyan) !important;
    border-left: 3px solid var(--blue) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="metric-container"]:hover {
    border-color: var(--blue) !important;
    background: var(--bg-hover) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px var(--glow-blue) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: .06em !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--blue), var(--purple)) !important;
    color: white !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--blue), var(--purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all .3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(59,130,246,.4) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stMultiselect > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--faint); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue); }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-weight: 500 !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: var(--blue) transparent transparent !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--blue), var(--cyan)) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
from dashboard.sidebar import render_sidebar
render_sidebar("app.py")

if _missing:
    with st.sidebar:
        with st.expander("⚠️ Missing models", expanded=False):
            st.markdown(
                "<p style='font-size:.78rem;color:#f59e0b;'>Run the training pipeline:</p>",
                unsafe_allow_html=True,
            )
            st.code("python -m src.train_pipeline", language="bash")

if _load_error:
    with st.sidebar:
        with st.expander("❌ Load error"):
            st.error(_load_error)

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center;padding:2.5rem 0 1rem;">
    <div style="display:inline-flex;align-items:center;gap:.5rem;padding:.35rem 1rem;
                background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);
                border-radius:50px;margin-bottom:1.1rem;">
        <div style="width:7px;height:7px;border-radius:50%;background:#10b981;
                    box-shadow:0 0 8px #10b981;animation:pulse 2s infinite;"></div>
        <span style="font-size:.78rem;color:#3b82f6;font-weight:600;letter-spacing:.06em;">
            SYSTEM OPERATIONAL
        </span>
    </div>
    <h1 style="font-size:2.9rem;font-weight:800;margin:0;line-height:1.1;
               background:linear-gradient(135deg,#f1f5f9 0%,#3b82f6 50%,#06b6d4 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        AI Fraud Detection &<br>Segmentation System
    </h1>
    <p style="margin:1rem auto 0;font-size:1rem;color:#94a3b8;max-width:580px;line-height:1.65;">
        Production-grade ML platform for real-time fraud detection and customer
        behaviour analysis on 6.35&nbsp;million mobile money transactions.
    </p>
</div>
<style>
@keyframes pulse {
    0%,100%{opacity:1;} 50%{opacity:.4;}
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Top-Level KPIs ───────────────────────────────────────────────────────────
st.markdown("<div style='height:.75rem;'></div>", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
kpi_rows = [
    (c1, "🎯 ROC-AUC",    "0.9991",   "+0.12%"),
    (c2, "✅ F1 Score",   "99.82%",   "+0.08%"),
    (c3, "🔒 Precision",  "99.94%",   "+0.05%"),
    (c4, "📡 Recall",     "99.70%",   "+0.15%"),
    (c5, "📁 Transactions","6.35M",   "PaySim"),
    (c6, "👥 Customers",  "2.77M",    "Segmented"),
]
for col, label, value, delta in kpi_rows:
    col.metric(label, value, delta)

st.markdown("<div style='height:.5rem;'></div>", unsafe_allow_html=True)

# ─── Navigation Feature Cards ─────────────────────────────────────────────────
st.markdown(
    """
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.5rem 0;">
""",
    unsafe_allow_html=True,
)

_pages = [
    ("#3b82f6",  "🔍", "Fraud Detection",
     "Score any transaction in real-time using the tuned Random Forest model. "
     "Supports batch CSV upload with risk-category labelling.",
     "dashboard/pages/1_fraud_detection.py"),
    ("#8b5cf6",  "📊", "Transaction Explorer",
     "Filter and visualise 500 K transactions. Interactive histograms, time-series "
     "fraud rates, and risk-category breakdowns.",
     "dashboard/pages/2_transaction_explorer.py"),
    ("#06b6d4",  "👥", "Customer Segmentation",
     "K-Means (K=4) clusters 2.77 M customers into Low-Risk, High-Value, and "
     "Fraud-Prone groups. PCA scatter + radar profiles.",
     "dashboard/pages/3_customer_segmentation.py"),
    ("#10b981",  "📈", "Model Performance",
     "ROC curve, confusion matrix, 5-fold CV stability, feature importances, "
     "imbalance-strategy comparison, and hyperparameter audit.",
     "dashboard/pages/4_model_performance.py"),
]

for color, icon, title, desc, _link in _pages:
    st.markdown(
        f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);
                    border-top:3px solid {color};border-radius:18px;padding:1.6rem;
                    transition:all .3s ease;cursor:pointer;"
             onmouseover="this.style.transform='translateY(-4px)';
                          this.style.boxShadow='0 12px 40px {color}28';"
             onmouseout="this.style.transform='';this.style.boxShadow='';">
            <div style="font-size:1.9rem;margin-bottom:.7rem;">{icon}</div>
            <h3 style="margin:0;font-size:.95rem;font-weight:700;color:#f1f5f9;">{title}</h3>
            <p style="margin:.45rem 0 0;font-size:.8rem;color:#94a3b8;line-height:1.55;">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ─── Two-column deep-dive ─────────────────────────────────────────────────────
col_arch, col_pipe = st.columns(2, gap="large")

# ── Left: Model Architecture ──
with col_arch:
    st.markdown(
        """
    <div style="background:var(--bg-card);border:1px solid var(--border);
                border-radius:16px;padding:1.5rem;height:100%;">
        <h3 style="margin:0 0 1rem;font-size:.95rem;font-weight:700;color:#3b82f6;">
            🧠 Model Architecture
        </h3>
    """,
        unsafe_allow_html=True,
    )
    _arch_rows = [
        ("#3b82f6", "Algorithm",       "RandomForestClassifier"),
        ("#8b5cf6", "N Estimators",    "50 trees"),
        ("#06b6d4", "Max Depth",        "10"),
        ("#10b981", "Class Imbalance",  "class_weight = 'balanced'"),
        ("#f59e0b", "Tuning",           "GridSearchCV · 3-fold · 48 fits"),
        ("#ef4444", "Features",         "13 engineered (domain + ratio)"),
        ("#94a3b8", "Scaler",           "StandardScaler (fit on train)"),
        ("#94a3b8", "Random State",     "42 (reproducible)"),
    ]
    for color, key, val in _arch_rows:
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:.6rem .9rem;background:rgba(148,163,184,.04);
                        border-radius:9px;margin-bottom:.35rem;border-left:3px solid {color}30;">
                <span style="font-size:.82rem;color:#94a3b8;font-weight:500;">{key}</span>
                <span style="font-size:.82rem;color:#f1f5f9;font-weight:700;">{val}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Right: Pipeline stages ──
with col_pipe:
    st.markdown(
        """
    <div style="background:var(--bg-card);border:1px solid var(--border);
                border-radius:16px;padding:1.5rem;height:100%;">
        <h3 style="margin:0 0 1rem;font-size:.95rem;font-weight:700;color:#8b5cf6;">
            🔄 ML Pipeline Stages
        </h3>
    """,
        unsafe_allow_html=True,
    )
    _stages = [
        ("#3b82f6,#1d4ed8",  "Data Ingestion",       "PaySim · 6.35M rows"),
        ("#8b5cf6,#6d28d9",  "Preprocessing",         "Clean · filter TRANSFER/CASH_OUT"),
        ("#06b6d4,#0891b2",  "Feature Engineering",   "13 domain features + scaling"),
        ("#10b981,#059669",  "Model Training",         "Baseline → Weighted → SMOTE → Tuned"),
        ("#f59e0b,#d97706",  "Evaluation",             "ROC · CM · 5-Fold CV"),
        ("#ef4444,#dc2626",  "Fraud Scoring",          "Score all 6.35M transactions"),
        ("#ec4899,#db2777",  "Customer Segmentation",  "K-Means K=4 · PCA 2D · 2.77M"),
    ]
    for i, (grad, title, desc) in enumerate(_stages, 1):
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:.7rem;padding:.55rem 0;
                        border-bottom:1px solid rgba(148,163,184,.07);">
                <div style="width:26px;height:26px;border-radius:50%;flex-shrink:0;
                            background:linear-gradient(135deg,{grad});
                            display:flex;align-items:center;justify-content:center;
                            font-size:.68rem;font-weight:800;color:white;">{i}</div>
                <div style="flex:1;">
                    <p style="margin:0;font-size:.83rem;font-weight:600;color:#f1f5f9;">{title}</p>
                    <p style="margin:0;font-size:.73rem;color:#64748b;">{desc}</p>
                </div>
                <span style="font-size:.68rem;padding:.18rem .55rem;border-radius:20px;
                             background:rgba(16,185,129,.15);color:#10b981;font-weight:700;
                             flex-shrink:0;">✓</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Feature Importance Chart + Insights ─────────────────────────────────────
st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='font-size:.95rem;font-weight:700;color:#f1f5f9;margin-bottom:.75rem;'>"
    "🏆 Top Predictive Features</h3>",
    unsafe_allow_html=True,
)

_FEAT_DATA = {
    "Feature": [
        "balance_difference_orig", "oldbalanceOrg",  "newbalanceOrig",
        "transaction_amount_ratio","newbalanceDest",  "balance_difference_dest",
        "oldbalanceDest",           "amount",          "step", "type_TRANSFER",
    ],
    "Importance": [
        0.4834, 0.1316, 0.0967,
        0.0726, 0.0630, 0.0437,
        0.0369, 0.0317, 0.0252, 0.0084,
    ],
}
_feat_df = pd.DataFrame(_FEAT_DATA)

_fig_feat = go.Figure(
    go.Bar(
        x=_feat_df["Importance"],
        y=_feat_df["Feature"],
        orientation="h",
        marker=dict(
            color=_feat_df["Importance"],
            colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#06b6d4"]],
            showscale=False,
        ),
        text=[f"{v:.1%}" for v in _feat_df["Importance"]],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    )
)
_fig_feat.update_layout(
    height=340,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=70, t=10, b=10),
    yaxis=dict(
        autorange="reversed",
        tickfont=dict(color="#94a3b8", size=11, family="JetBrains Mono"),
        gridcolor="rgba(148,163,184,.06)",
    ),
    xaxis=dict(
        tickfont=dict(color="#64748b", size=10),
        gridcolor="rgba(148,163,184,.06)",
    ),
    font=dict(family="Inter", color="#94a3b8"),
    showlegend=False,
)

col_feat, col_ins = st.columns([3, 2], gap="large")

with col_feat:
   st.markdown(
    """
    <div style="
        background:var(--bg-card);
        border:1px solid var(--border);
        border-radius:16px;
        padding:1rem;
    ">
    """,
    unsafe_allow_html=True,
)

st.plotly_chart(
    _fig_feat,
    use_container_width=True,
    config={"displayModeBar": False}
)

st.markdown("</div>", unsafe_allow_html=True)

with col_ins:
    _insights = [
        ("#ef4444", "Balance Depletion · #1 Signal",
         "48.3% importance — fraudulent TRANSFERs drain the sender's account to zero, "
         "producing a unique balance-difference signature."),
        ("#3b82f6", "Opening Balance",
         "13.2% importance — fraud exclusively targets accounts with a non-zero starting balance."),
        ("#f59e0b", "Amount Ratio ≈ 1.0",
         "7.3% importance — fraudsters transfer 100% of the available balance."),
        ("#8b5cf6", "Receiver Not Updated",
         "PaySim anomaly: destination balance stays at 0 post-transfer in fraudulent transactions."),
    ]
    st.markdown(
    """
    <div style="
        background:var(--bg-card);
        border:1px solid var(--border);
        border-radius:16px;
        padding:1.5rem;
        height:100%;
    ">
        <h4 style="
            margin:0 0 .9rem;
            font-size:.88rem;
            font-weight:700;
            color:#3b82f6;
        ">
            💡 Key Insights
        </h4>
    """,
    unsafe_allow_html=True,
)
    
    for color, title, body in _insights:
        st.markdown(
            f"""
            <div style="padding:.7rem .9rem;background:{color}14;border-radius:10px;
                        border-left:3px solid {color};margin-bottom:.6rem;">
                <p style="margin:0;font-size:.8rem;font-weight:700;color:#f1f5f9;">{title}</p>
                <p style="margin:.25rem 0 0;font-size:.74rem;color:#94a3b8;line-height:1.45;">{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Artifact Inventory ───────────────────────────────────────────────────────
st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

with st.expander("📁 Generated Artifact Inventory", expanded=False):
    _base = _PROJECT_ROOT
    _artifacts = [
        ("models/random_forest.pkl",            "Tuned RF classifier · main fraud model"),
        ("models/scaler.pkl",                    "StandardScaler for transaction features"),
        ("models/kmeans.pkl",                    "K-Means model (K=4) for segmentation"),
        ("models/scaler_kmeans.pkl",             "StandardScaler for customer features"),
        ("data/processed/scored_transactions.csv","All 6.35M transactions + fraud scores"),
        ("data/processed/customer_segments.csv", "2.77M customer profiles + cluster labels"),
        ("reports/figures/confusion_matrix.png", "Test-set confusion matrix"),
        ("reports/figures/roc_curve.png",        "ROC curve · AUC = 0.9991"),
        ("reports/figures/feature_importance.png","Top-13 Gini feature importances"),
        ("reports/figures/elbow_curve.png",       "K-Means elbow (K=1–8)"),
        ("reports/figures/pca_clusters.png",      "2-D PCA cluster scatter (50K sample)"),
    ]
    rows = []
    for rel, desc in _artifacts:
        full = os.path.join(_base, rel.replace("/", os.sep))
        exists = os.path.exists(full)
        size_str = ""
        if exists:
            sz = os.path.getsize(full)
            size_str = f"{sz/1e6:.1f} MB" if sz >= 1e6 else f"{sz/1e3:.0f} KB"
        rows.append(
            {"File": rel, "Description": desc,
             "Status": "✅ Present" if exists else "❌ Missing",
             "Size": size_str if exists else "—"}
        )
    _inv_df = pd.DataFrame(rows)

    def _style_status(v):
        if "✅" in str(v):
            return "color:#10b981;font-weight:600"
        return "color:#ef4444;font-weight:600"


    st.dataframe(
    _inv_df.style.map(_style_status, subset=["Status"]),
    use_container_width=True
)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center;padding:2.5rem 0 1rem;margin-top:1.5rem;
            border-top:1px solid rgba(148,163,184,.1);">
    <p style="margin:0;font-size:.8rem;color:#475569;">
        🛡️ <strong style="color:#3b82f6;">FraudShield AI</strong>
        &nbsp;·&nbsp; Random Forest + K-Means on PaySim
        &nbsp;·&nbsp; <span style="color:#10b981;">ROC-AUC: 0.9991</span>
    </p>
    <p style="margin:.3rem 0 0;font-size:.72rem;color:#334155;">
        End-to-End ML Portfolio Project &nbsp;·&nbsp; 6,354,407 transactions analysed
    </p>
</div>
""",
    unsafe_allow_html=True,
)
