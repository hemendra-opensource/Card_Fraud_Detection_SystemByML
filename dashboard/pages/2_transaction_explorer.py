"""
Page 2: Transaction Explorer
Interactive analytics dashboard for the scored transactions dataset.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Transaction Explorer · FraudShield AI", page_icon="📊", layout="wide")

from dashboard.sidebar import render_sidebar
render_sidebar("pages/2_transaction_explorer.py")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root{--bg-primary:#0a0e1a;--bg-card:#1a2235;--bg-card-hover:#1f2a40;--accent-blue:#3b82f6;--accent-cyan:#06b6d4;--accent-purple:#8b5cf6;--accent-green:#10b981;--accent-red:#ef4444;--accent-orange:#f59e0b;--text-primary:#f1f5f9;--text-secondary:#94a3b8;--text-muted:#64748b;--border:rgba(148,163,184,0.12);}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background-color:var(--bg-primary)!important;color:var(--text-primary)!important;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:linear-gradient(135deg,#0a0e1a 0%,#0d1b2a 50%,#0a0e1a 100%)!important;}
.main .block-container{padding:1.5rem 2.5rem!important;max-width:1600px!important;}
[data-testid="metric-container"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:16px!important;padding:1.2rem!important;transition:all 0.3s ease!important;}
[data-testid="metric-container"]:hover{border-color:var(--accent-blue)!important;transform:translateY(-2px)!important;}
[data-testid="stMetricLabel"]{color:var(--text-secondary)!important;font-size:0.8rem!important;text-transform:uppercase!important;letter-spacing:0.05em!important;}
[data-testid="stMetricValue"]{color:var(--text-primary)!important;font-size:1.7rem!important;font-weight:700!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117 0%,#111827 100%)!important;border-right:1px solid var(--border)!important;}
.stSelectbox>div>div,.stMultiselect>div>div{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text-primary)!important;}
[data-testid="stDataFrame"]{border-radius:12px!important;border:1px solid var(--border)!important;}
::-webkit-scrollbar{width:6px;height:6px;}::-webkit-scrollbar-thumb{background:#475569;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(148,163,184,0.07)"

@st.cache_data(show_spinner=False, max_entries=1)
def load_transactions():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    path = os.path.join(base, "data/processed/scored_transactions.csv")
    cols = ['step','type','amount','nameOrig','nameDest',
            'oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest',
            'isFraud','fraud_score','risk_category']
    df = pd.read_csv(path, usecols=cols, nrows=500_000)
    return df

with st.spinner("Loading transaction data..."):
    try:
        df = load_transactions()
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data_loaded = False

# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
    <div style="display:inline-flex;align-items:center;gap:0.5rem;padding:0.3rem 0.9rem;
                background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.3);
                border-radius:50px;margin-bottom:0.75rem;">
        <span style="font-size:0.78rem;color:#8b5cf6;font-weight:600;letter-spacing:0.05em;">📊 TRANSACTION ANALYTICS</span>
    </div>
    <h1 style="margin:0;font-size:2rem;font-weight:800;
               background:linear-gradient(135deg,#f1f5f9,#8b5cf6);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Transaction Explorer
    </h1>
    <p style="margin:0.5rem 0 0;color:#94a3b8;font-size:0.95rem;">
        Interactive analytics across 500K sampled transactions with fraud scores and risk classifications.
    </p>
</div>
""", unsafe_allow_html=True)

if not data_loaded:
    st.stop()

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <h3 style="font-size:0.9rem;font-weight:700;color:#3b82f6;margin:1rem 0 0.75rem;">
        🎛️ Filters
    </h3>
    """, unsafe_allow_html=True)

    tx_types = st.multiselect("Transaction Type", ["TRANSFER", "CASH_OUT"],
                               default=["TRANSFER", "CASH_OUT"])
    risk_cats = st.multiselect("Risk Category",
                                ["Low Risk", "Medium Risk", "High Risk"],
                                default=["Low Risk", "Medium Risk", "High Risk"])
    fraud_only = st.toggle("Show Confirmed Fraud Only", value=False)
    amount_range = st.slider("Amount Range ($)",
                              0, int(df['amount'].quantile(0.99)),
                              (0, int(df['amount'].quantile(0.99))))
    score_range = st.slider("Fraud Score Range", 0.0, 100.0, (0.0, 100.0))
    step_range = st.slider("Time Step Range", int(df['step'].min()), int(df['step'].max()),
                            (int(df['step'].min()), int(df['step'].max())))

# ─── Apply Filters ────────────────────────────────────────────────────────────
filtered = df.copy()
if tx_types:
    filtered = filtered[filtered['type'].isin(tx_types)]
if risk_cats:
    filtered = filtered[filtered['risk_category'].isin(risk_cats)]
if fraud_only:
    filtered = filtered[filtered['isFraud'] == 1]
filtered = filtered[
    (filtered['amount'] >= amount_range[0]) & (filtered['amount'] <= amount_range[1]) &
    (filtered['fraud_score'] >= score_range[0]) & (filtered['fraud_score'] <= score_range[1]) &
    (filtered['step'] >= step_range[0]) & (filtered['step'] <= step_range[1])
]

# ─── KPIs ─────────────────────────────────────────────────────────────────────
total = len(filtered)
fraud_count = filtered['isFraud'].sum()
fraud_rate = fraud_count / total * 100 if total > 0 else 0
avg_score = filtered['fraud_score'].mean()
total_amount = filtered['amount'].sum()
high_risk_count = (filtered['risk_category'] == 'High Risk').sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("Confirmed Fraud", f"{fraud_count:,}", f"{fraud_rate:.2f}%")
c3.metric("Avg Fraud Score", f"{avg_score:.1f}%")
c4.metric("Total Volume ($)", f"${total_amount/1e6:.2f}M")
c5.metric("High Risk Flagged", f"{high_risk_count:,}", f"{high_risk_count/total*100:.1f}%")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ─── Charts Row 1 ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1.5, 1.5], gap="medium")

with col1:
    # Fraud score distribution
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=filtered[filtered['isFraud']==0]['fraud_score'],
        name='Legitimate', nbinsx=50,
        marker_color='rgba(59,130,246,0.6)',
        opacity=0.75
    ))
    fig_hist.add_trace(go.Histogram(
        x=filtered[filtered['isFraud']==1]['fraud_score'],
        name='Fraud', nbinsx=50,
        marker_color='rgba(239,68,68,0.8)',
        opacity=0.75
    ))
    fig_hist.update_layout(
        title=dict(text="Fraud Score Distribution", font=dict(color="#f1f5f9", size=13)),
        barmode='overlay',
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=320,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title=dict(text="Fraud Score (%)", font=dict(color="#64748b")),
                   tickfont=dict(color="#64748b"), gridcolor=GRID_COLOR),
        yaxis=dict(title=dict(text="Count", font=dict(color="#64748b")),
                   tickfont=dict(color="#64748b"), gridcolor=GRID_COLOR),
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter")
    )
    fig_hist.add_vline(x=30, line_dash="dash", line_color="#f59e0b", opacity=0.7,
                        annotation_text="30%", annotation_font_color="#f59e0b")
    fig_hist.add_vline(x=70, line_dash="dash", line_color="#ef4444", opacity=0.7,
                        annotation_text="70%", annotation_font_color="#ef4444")
    st.markdown('<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1rem;">', unsafe_allow_html=True)
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Risk category pie
    risk_counts = filtered['risk_category'].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.6,
        marker=dict(colors=["#10b981", "#f59e0b", "#ef4444"],
                    line=dict(color="#0a0e1a", width=2)),
        textfont=dict(color="white", family="Inter"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>"
    ))
    fig_pie.update_layout(
        title=dict(text="Risk Distribution", font=dict(color="#f1f5f9", size=13)),
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=320,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=-0.15),
        margin=dict(l=0, r=0, t=40, b=30),
        annotations=[dict(text=f"{fraud_rate:.1f}%<br><span style='font-size:10px'>Fraud Rate</span>",
                          x=0.5, y=0.5, font=dict(size=14, color="#f1f5f9", family="Inter"),
                          showarrow=False)],
        font=dict(family="Inter")
    )
    st.markdown('<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1rem;">', unsafe_allow_html=True)
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    # Transaction type breakdown
    type_fraud = filtered.groupby('type').agg(
        total=('isFraud', 'count'),
        fraud=('isFraud', 'sum')
    ).reset_index()
    type_fraud['legit'] = type_fraud['total'] - type_fraud['fraud']
    type_fraud['fraud_rate'] = (type_fraud['fraud'] / type_fraud['total'] * 100).round(2)

    fig_bar_type = go.Figure()
    fig_bar_type.add_trace(go.Bar(name='Legitimate', x=type_fraud['type'], y=type_fraud['legit'],
                                   marker_color='rgba(59,130,246,0.7)'))
    fig_bar_type.add_trace(go.Bar(name='Fraud', x=type_fraud['type'], y=type_fraud['fraud'],
                                   marker_color='rgba(239,68,68,0.9)'))
    fig_bar_type.update_layout(
        title=dict(text="Fraud by Type", font=dict(color="#f1f5f9", size=13)),
        barmode='stack',
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=320,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(tickfont=dict(color="#94a3b8"), gridcolor=GRID_COLOR),
        yaxis=dict(tickfont=dict(color="#64748b"), gridcolor=GRID_COLOR),
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter")
    )
    st.markdown('<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1rem;">', unsafe_allow_html=True)
    st.plotly_chart(fig_bar_type, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Charts Row 2 ─────────────────────────────────────────────────────────────
col4, col5 = st.columns([2, 1.5], gap="medium")

with col4:
    # Fraud rate over time (step bins)
    step_agg = filtered.copy()
    step_agg['step_bin'] = (step_agg['step'] // 24).astype(int)
    step_time = step_agg.groupby('step_bin').agg(
        total=('isFraud', 'count'),
        fraud=('isFraud', 'sum'),
        avg_score=('fraud_score', 'mean')
    ).reset_index()
    step_time['fraud_rate'] = step_time['fraud'] / step_time['total'] * 100
    step_time['day'] = step_time['step_bin']

    fig_time = make_subplots(specs=[[{"secondary_y": True}]])
    fig_time.add_trace(go.Scatter(
        x=step_time['day'], y=step_time['total'],
        name='Total Transactions', fill='tozeroy',
        fillcolor='rgba(59,130,246,0.1)',
        line=dict(color='#3b82f6', width=1.5)
    ), secondary_y=False)
    fig_time.add_trace(go.Scatter(
        x=step_time['day'], y=step_time['fraud_rate'],
        name='Fraud Rate (%)', mode='lines',
        line=dict(color='#ef4444', width=2, dash='dot')
    ), secondary_y=True)
    fig_time.update_layout(
        title=dict(text="Transaction Volume & Fraud Rate Over Time (Days)", font=dict(color="#f1f5f9", size=13)),
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=320,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=-0.2),
        xaxis=dict(title=dict(text="Day", font=dict(color="#64748b")),
                   tickfont=dict(color="#64748b"), gridcolor=GRID_COLOR),
        margin=dict(l=10, r=10, t=40, b=40),
        font=dict(family="Inter")
    )
    fig_time.update_yaxes(title_text="Transactions", title_font=dict(color="#64748b"),
                           tickfont=dict(color="#64748b"), gridcolor=GRID_COLOR, secondary_y=False)
    fig_time.update_yaxes(title_text="Fraud Rate (%)", title_font=dict(color="#ef4444"),
                           tickfont=dict(color="#ef4444"), secondary_y=True)
    st.markdown('<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1rem;">', unsafe_allow_html=True)
    st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col5:
    # Amount box plot by risk
    fig_box = go.Figure()
    colors_box = {"Low Risk": "#10b981", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"}
    for cat in ["Low Risk", "Medium Risk", "High Risk"]:
        sub = filtered[filtered['risk_category'] == cat]['amount']
        if len(sub) > 0:
            fig_box.add_trace(go.Box(
                y=sub.clip(upper=sub.quantile(0.99)),
                name=cat,
                marker_color=colors_box[cat],
                fillcolor=colors_box[cat].replace(")", ",0.15)").replace("rgb", "rgba"),
                line=dict(color=colors_box[cat]),
                boxmean='sd'
            ))
    fig_box.update_layout(
        title=dict(text="Transaction Amount by Risk Category", font=dict(color="#f1f5f9", size=13)),
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=320,
        showlegend=False,
        yaxis=dict(title=dict(text="Amount ($)", font=dict(color="#64748b")),
                   tickfont=dict(color="#64748b"), gridcolor=GRID_COLOR),
        xaxis=dict(title=dict(text="Risk Category", font=dict(color="#94a3b8")),
                   tickfont=dict(color="#94a3b8")),
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter")
    )
    st.markdown('<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1rem;">', unsafe_allow_html=True)
    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Data Table ───────────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("""
<h3 style="font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:0.75rem;">
    🗃️ Transaction Records
</h3>
""", unsafe_allow_html=True)

display_cols = ['nameOrig','nameDest','type','amount','isFraud','fraud_score','risk_category','step']
sort_col = st.selectbox("Sort by", ['fraud_score','amount','step'], index=0, label_visibility='collapsed')
show_n = st.select_slider("Rows to display", options=[25, 50, 100, 250, 500], value=50)

display_df = filtered[display_cols].sort_values(sort_col, ascending=False).head(show_n)

# Color-code risk_category
def highlight_risk(val):
    colors = {"High Risk": "color: #ef4444; font-weight: 700",
              "Medium Risk": "color: #f59e0b; font-weight: 600",
              "Low Risk": "color: #10b981"}
    return colors.get(val, "")

styled = display_df.style.map(highlight_risk, subset=['risk_category']) \
                         .format({'amount': '${:,.2f}', 'fraud_score': '{:.1f}%'})
st.dataframe(styled, use_container_width=True, height=420)

csv_export = filtered.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Export Filtered Data (CSV)", csv_export, "filtered_transactions.csv", "text/csv")
