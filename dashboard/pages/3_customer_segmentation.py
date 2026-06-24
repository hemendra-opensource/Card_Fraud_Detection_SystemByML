"""
Page 3: Customer Segmentation
Visualizes K-Means customer segments with PCA, profiles, and business insights.
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

st.set_page_config(page_title="Customer Segmentation · FraudShield AI", page_icon="👥", layout="wide")

from dashboard.sidebar import render_sidebar
render_sidebar("pages/3_customer_segmentation.py")

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
.stSelectbox>div>div{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text-primary)!important;}
[data-testid="stDataFrame"]{border-radius:12px!important;border:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg-card)!important;border-radius:12px!important;padding:4px!important;border:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab"]{border-radius:9px!important;color:var(--text-secondary)!important;font-weight:500!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--accent-blue),var(--accent-purple))!important;color:white!important;}
::-webkit-scrollbar{width:6px;height:6px;}::-webkit-scrollbar-thumb{background:#475569;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(148,163,184,0.07)"
SEGMENT_COLORS = {
    "Low-Risk Regular Customers": "#10b981",
    "High-Value Customers": "#3b82f6",
    "High-Value Customers (Group 2)": "#06b6d4",
    "Fraud-Prone Customers": "#ef4444",
    "High-Risk Customers": "#f59e0b",
    "Dormant Customers": "#8b5cf6"
}

@st.cache_data(show_spinner=False, max_entries=1)
def load_segments():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    path = os.path.join(base, "data/processed/customer_segments.csv")
    df = pd.read_csv(path, nrows=300_000)
    return df

with st.spinner("Loading customer segmentation data..."):
    try:
        seg_df = load_segments()
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data_loaded = False

# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
    <div style="display:inline-flex;align-items:center;gap:0.5rem;padding:0.3rem 0.9rem;
                background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.3);
                border-radius:50px;margin-bottom:0.75rem;">
        <span style="font-size:0.78rem;color:#06b6d4;font-weight:600;letter-spacing:0.05em;">👥 CUSTOMER INTELLIGENCE</span>
    </div>
    <h1 style="margin:0;font-size:2rem;font-weight:800;
               background:linear-gradient(135deg,#f1f5f9,#06b6d4);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Customer Segmentation
    </h1>
    <p style="margin:0.5rem 0 0;color:#94a3b8;font-size:0.95rem;">
        K-Means (K=4) clustering reveals behavioral patterns across 2.77M customers.
        PCA reduces 7 features to 2D for visualization (71.5% variance explained).
    </p>
</div>
""", unsafe_allow_html=True)

if not data_loaded:
    st.stop()

# ─── Segment KPIs ─────────────────────────────────────────────────────────────
seg_counts = seg_df['segment_label'].value_counts()
total_customers = len(seg_df)

segments = seg_df['segment_label'].unique().tolist()
cols = st.columns(min(len(segments), 4))

for i, seg in enumerate(sorted(segments)):
    count = seg_counts.get(seg, 0)
    pct = count / total_customers * 100
    color = SEGMENT_COLORS.get(seg, "#94a3b8")
    with cols[i % 4]:
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid {color}40;border-top:3px solid {color};
                    border-radius:16px;padding:1.2rem;margin-bottom:0.5rem;transition:all 0.3s;">
            <p style="margin:0;font-size:0.75rem;color:{color};font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">
                {seg}
            </p>
            <p style="margin:0.4rem 0 0;font-size:1.8rem;font-weight:800;color:#f1f5f9;">{count:,}</p>
            <p style="margin:0.2rem 0 0;font-size:0.82rem;color:#64748b;">{pct:.1f}% of customers</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ─── Main Charts ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🗺️ PCA Cluster Map", "📊 Segment Profiles", "📋 Customer Data"])

with tab1:
    col_pca, col_info = st.columns([3, 1.5], gap="large")

    with col_pca:
        # PCA scatter plot
        plot_sample = seg_df.sample(n=min(30000, len(seg_df)), random_state=42) if len(seg_df) > 30000 else seg_df

        # Assign colors
        color_sequence = [SEGMENT_COLORS.get(s, "#94a3b8") for s in sorted(plot_sample['segment_label'].unique())]

        fig_pca = px.scatter(
            plot_sample, x='pca1', y='pca2',
            color='segment_label',
            opacity=0.65,
            color_discrete_map=SEGMENT_COLORS,
            labels={'pca1': 'Principal Component 1 (43.1% variance)',
                    'pca2': 'Principal Component 2 (28.4% variance)',
                    'segment_label': 'Customer Segment'},
            title=f"Customer Segments in PCA 2D Space (n={len(plot_sample):,} sampled)",
            hover_data={'pca1': ':.2f', 'pca2': ':.2f'}
        )
        fig_pca.update_traces(marker=dict(size=3))
        fig_pca.update_layout(
            height=520,
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            title=dict(font=dict(color="#f1f5f9", size=13)),
            legend=dict(
                font=dict(color="#94a3b8", size=11, family="Inter"),
                bgcolor="rgba(26,34,53,0.9)",
                bordercolor="rgba(148,163,184,0.15)",
                borderwidth=1,
                title=dict(text="Segment", font=dict(color="#94a3b8")),
                orientation="v"
            ),
            xaxis=dict(title_font=dict(color="#64748b"), tickfont=dict(color="#64748b"),
                       gridcolor=GRID_COLOR, zeroline=False),
            yaxis=dict(title_font=dict(color="#64748b"), tickfont=dict(color="#64748b"),
                       gridcolor=GRID_COLOR, zeroline=False),
            margin=dict(l=10, r=10, t=50, b=10),
            font=dict(family="Inter")
        )
        st.markdown('<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1rem;">', unsafe_allow_html=True)
        st.plotly_chart(fig_pca, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1.5rem;">
            <h4 style="margin:0 0 1rem;font-size:0.9rem;font-weight:700;color:#06b6d4;">
                🧪 PCA Details
            </h4>
            <div style="padding:0.75rem;background:rgba(6,182,212,0.08);border-radius:10px;margin-bottom:0.75rem;">
                <p style="margin:0;font-size:0.75rem;color:#64748b;">PC1 Variance</p>
                <p style="margin:0.2rem 0 0;font-size:1.2rem;font-weight:700;color:#06b6d4;">43.1%</p>
            </div>
            <div style="padding:0.75rem;background:rgba(59,130,246,0.08);border-radius:10px;margin-bottom:0.75rem;">
                <p style="margin:0;font-size:0.75rem;color:#64748b;">PC2 Variance</p>
                <p style="margin:0.2rem 0 0;font-size:1.2rem;font-weight:700;color:#3b82f6;">28.4%</p>
            </div>
            <div style="padding:0.75rem;background:rgba(139,92,246,0.08);border-radius:10px;margin-bottom:0.75rem;">
                <p style="margin:0;font-size:0.75rem;color:#64748b;">Total Explained</p>
                <p style="margin:0.2rem 0 0;font-size:1.2rem;font-weight:700;color:#8b5cf6;">71.5%</p>
            </div>
            <div style="padding:0.75rem;background:rgba(16,185,129,0.08);border-radius:10px;margin-bottom:1rem;">
                <p style="margin:0;font-size:0.75rem;color:#64748b;">Input Features</p>
                <p style="margin:0.2rem 0 0;font-size:1.2rem;font-weight:700;color:#10b981;">7</p>
            </div>
            <h4 style="margin:0 0 0.75rem;font-size:0.9rem;font-weight:700;color:#f1f5f9;">
                📌 Segment Legends
            </h4>
        """, unsafe_allow_html=True)

        seg_descs = {
            "Low-Risk Regular Customers": ("🟢", "Infrequent, low-value transactions. No fraud history."),
            "High-Value Customers": ("🔵", "Large average balances and high total spending."),
            "High-Value Customers (Group 2)": ("🔷", "Multi-transaction, high-volume second tier."),
            "Fraud-Prone Customers": ("🔴", "High fraud scores and confirmed fraud history."),
        }
        for seg in sorted(seg_df['segment_label'].unique()):
            icon, desc = seg_descs.get(seg, ("⚪", "Customer segment"))
            color = SEGMENT_COLORS.get(seg, "#94a3b8")
            st.markdown(f"""
            <div style="display:flex;gap:0.6rem;padding:0.6rem 0;border-bottom:1px solid rgba(148,163,184,0.08);">
                <span style="font-size:1rem;flex-shrink:0;">{icon}</span>
                <div>
                    <p style="margin:0;font-size:0.78rem;font-weight:600;color:{color};">{seg}</p>
                    <p style="margin:0.15rem 0 0;font-size:0.72rem;color:#64748b;">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    # Segment profile comparison
    profile_cols = ['transaction_frequency', 'average_transaction_amount',
                    'total_spending', 'fraud_probability_score',
                    'number_of_suspicious_transactions', 'fraud_count', 'average_balance']

    profile = seg_df.groupby('segment_label')[profile_cols].mean().reset_index()

    col_radar, col_bars = st.columns([1, 1.5], gap="large")

    with col_radar:
        # Radar chart
        radar_features = ['transaction_frequency', 'average_transaction_amount',
                          'fraud_probability_score', 'number_of_suspicious_transactions', 'average_balance']
        radar_labels = ['Tx Frequency', 'Avg Amount', 'Fraud Score', 'Suspicious Txs', 'Avg Balance']

        fig_radar = go.Figure()
        for _, row in profile.iterrows():
            vals = row[radar_features].values
            # Normalize to 0-1 for radar
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
            norm_vals = np.concatenate([norm_vals, [norm_vals[0]]])
            label_loop = radar_labels + [radar_labels[0]]
            color = SEGMENT_COLORS.get(row['segment_label'], "#94a3b8")

            fig_radar.add_trace(go.Scatterpolar(
                r=norm_vals, theta=label_loop,
                fill='toself', name=row['segment_label'],
                fillcolor=color.replace("#", "rgba(").replace("10b981", "16,185,129") + ",0.1)",
                line=dict(color=color, width=2),
                opacity=0.8
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#64748b", size=9),
                                gridcolor=GRID_COLOR),
                angularaxis=dict(tickfont=dict(color="#94a3b8", size=10), gridcolor=GRID_COLOR),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8", size=10), bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor=CHART_BG,
            height=420,
            title=dict(text="Segment Profile Radar (Normalized)", font=dict(color="#f1f5f9", size=13)),
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(family="Inter")
        )
        st.markdown('<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1rem;">', unsafe_allow_html=True)
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_bars:
        # Bar chart comparison
        metric_select = st.selectbox(
            "Compare segments by:",
            ["average_balance", "fraud_probability_score", "total_spending",
             "transaction_frequency", "fraud_count", "number_of_suspicious_transactions"]
        )
        metric_labels = {
            "average_balance": "Avg Balance ($)",
            "fraud_probability_score": "Avg Fraud Score (%)",
            "total_spending": "Total Spending ($)",
            "transaction_frequency": "Avg Tx Frequency",
            "fraud_count": "Avg Fraud Count",
            "number_of_suspicious_transactions": "Avg Suspicious Txs"
        }

        profile_sorted = profile.sort_values(metric_select, ascending=False)
        bar_colors = [SEGMENT_COLORS.get(s, "#94a3b8") for s in profile_sorted['segment_label']]

        fig_bar = go.Figure(go.Bar(
            x=profile_sorted['segment_label'],
            y=profile_sorted[metric_select],
            marker=dict(color=bar_colors, opacity=0.85),
            text=[f"{v:,.1f}" for v in profile_sorted[metric_select]],
            textposition='outside',
            textfont=dict(color="#94a3b8", size=10),
            hovertemplate="<b>%{x}</b><br>" + metric_labels[metric_select] + ": %{y:,.2f}<extra></extra>"
        ))
        fig_bar.update_layout(
            title=dict(text=f"{metric_labels[metric_select]} by Segment", font=dict(color="#f1f5f9", size=13)),
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            height=420,
            xaxis=dict(tickfont=dict(color="#94a3b8", size=9), gridcolor=GRID_COLOR,
                       tickangle=-15),
            yaxis=dict(title=metric_labels[metric_select], titlefont=dict(color="#64748b"),
                       tickfont=dict(color="#64748b"), gridcolor=GRID_COLOR),
            margin=dict(l=10, r=10, t=50, b=80),
            font=dict(family="Inter")
        )
        st.markdown('<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1rem;">', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Segment summary table
    st.markdown("""
    <h4 style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin:1.5rem 0 0.75rem;">
        📋 Segment Profile Summary
    </h4>
    """, unsafe_allow_html=True)

    profile_display = profile.copy()
    profile_display['Count'] = [seg_counts.get(s, 0) for s in profile_display['segment_label']]
    profile_display['Share (%)'] = (profile_display['Count'] / total_customers * 100).round(2)
    profile_display = profile_display.rename(columns={
        'segment_label': 'Segment',
        'transaction_frequency': 'Avg Tx Freq',
        'average_transaction_amount': 'Avg Amount ($)',
        'total_spending': 'Total Spend ($)',
        'fraud_probability_score': 'Avg Fraud Score',
        'number_of_suspicious_transactions': 'Suspicious Txs',
        'fraud_count': 'Fraud Count',
        'average_balance': 'Avg Balance ($)'
    })
    cols_order = ['Segment', 'Count', 'Share (%)', 'Avg Tx Freq', 'Avg Amount ($)',
                  'Total Spend ($)', 'Avg Fraud Score', 'Fraud Count', 'Avg Balance ($)']
    st.dataframe(
        profile_display[cols_order].style.format({
            'Count': '{:,.0f}',
            'Share (%)': '{:.2f}%',
            'Avg Tx Freq': '{:.2f}',
            'Avg Amount ($)': '${:,.0f}',
            'Total Spend ($)': '${:,.0f}',
            'Avg Fraud Score': '{:.2f}%',
            'Fraud Count': '{:.3f}',
            'Avg Balance ($)': '${:,.0f}'
        }),
        use_container_width=True,
        height=200
    )

with tab3:
    # Customer-level data table
    st.markdown("""
    <h4 style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin:0.5rem 0 0.75rem;">
        👤 Individual Customer Records
    </h4>
    """, unsafe_allow_html=True)

    seg_filter = st.selectbox("Filter by Segment", ["All"] + sorted(seg_df['segment_label'].unique().tolist()))
    fraud_filter = st.toggle("Fraud customers only", value=False, key="seg_fraud_toggle")

    display_segs = seg_df.copy()
    if seg_filter != "All":
        display_segs = display_segs[display_segs['segment_label'] == seg_filter]
    if fraud_filter:
        display_segs = display_segs[display_segs['fraud_count'] > 0]

    disp_cols = ['nameOrig', 'segment_label', 'transaction_frequency',
                 'average_transaction_amount', 'total_spending',
                 'fraud_probability_score', 'fraud_count', 'average_balance']

    show_cust = st.select_slider("Customers to display", options=[50, 100, 200, 500], value=100)
    st.dataframe(
        display_segs[disp_cols].sort_values('fraud_probability_score', ascending=False)
                                .head(show_cust)
                                .style.format({
                                    'transaction_frequency': '{:.0f}',
                                    'average_transaction_amount': '${:,.2f}',
                                    'total_spending': '${:,.0f}',
                                    'fraud_probability_score': '{:.2f}%',
                                    'fraud_count': '{:.0f}',
                                    'average_balance': '${:,.0f}'
                                }),
        use_container_width=True,
        height=480
    )

    csv_seg = display_segs.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Export Segment Data (CSV)", csv_seg, "customer_segments_filtered.csv", "text/csv")
