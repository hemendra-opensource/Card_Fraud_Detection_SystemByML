import streamlit as st
import os

def render_sidebar(current_page: str):
    """
    Renders a unified, premium navigation sidebar with branding,
    custom page links, model status indicators, and footer info.
    """
    # Hide default sidebar navigation
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        # ── Branding ──
        st.markdown(
            """
            <div style="text-align:center;padding:1rem 0 1.5rem;">
                <div style="font-size:3rem;margin-bottom:.4rem;">🛡️</div>
                <h2 style="margin:0;font-size:1.3rem;font-weight:800;
                           background:linear-gradient(135deg,#3b82f6,#06b6d4);
                           -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    FraudShield AI
                </h2>
                <p style="margin:.25rem 0 0;font-size:.72rem;color:#64748b;
                          letter-spacing:.1em;text-transform:uppercase;">
                    ML Detection System
                </p>
            </div>
            <hr style="border:none;border-top:1px solid rgba(148,163,184,.12);margin:0 0 1rem;">
            """,
            unsafe_allow_html=True,
        )

        # ── Nav header ──
        st.markdown(
            "<p style='font-size:.72rem;color:#64748b;text-transform:uppercase;"
            "letter-spacing:.08em;font-weight:700;margin-bottom:.4rem;padding:.2rem .5rem;'>Navigate</p>",
            unsafe_allow_html=True,
        )

        # ── Page links (Streamlit multipage — relative to entrypoint directory) ──
        st.page_link("app.py",
                     label="🏠  Overview",
                     use_container_width=True,
                     disabled=(current_page == "app.py"))
        st.page_link("pages/1_fraud_detection.py",
                     label="🔍  Fraud Detection",
                     use_container_width=True,
                     disabled=(current_page == "pages/1_fraud_detection.py"))
        st.page_link("pages/2_transaction_explorer.py",
                     label="📊  Transaction Explorer",
                     use_container_width=True,
                     disabled=(current_page == "pages/2_transaction_explorer.py"))
        st.page_link("pages/3_customer_segmentation.py",
                     label="👥  Customer Segmentation",
                     use_container_width=True,
                     disabled=(current_page == "pages/3_customer_segmentation.py"))
        st.page_link("pages/4_model_performance.py",
                     label="📈  Model Performance",
                     use_container_width=True,
                     disabled=(current_page == "pages/4_model_performance.py"))

        st.markdown(
            "<hr style='border:none;border-top:1px solid rgba(148,163,184,.12);margin:1.2rem 0;'>",
            unsafe_allow_html=True,
        )

        # ── Model status card ──
        models = st.session_state.get("models", {})
        rf_ok    = "random_forest"  in models
        km_ok    = "kmeans"         in models
        sc_ok    = "scaler"         in models
        sck_ok   = "scaler_kmeans"  in models

        def _status_dot(ok: bool) -> str:
            color = "#10b981" if ok else "#ef4444"
            label = "Active" if ok else "Missing"
            return (
                f"<div style='display:flex;align-items:center;gap:.4rem;margin-bottom:.3rem;'>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:{color};"
                f"box-shadow:0 0 6px {color};flex-shrink:0;'></div>"
                f"<span style='font-size:.74rem;color:{color};font-weight:500;'>{label}</span>"
                f"</div>"
            )

        st.markdown(
            f"""
            <div style="padding:1rem;background:rgba(59,130,246,.08);border-radius:12px;
                        border:1px solid rgba(59,130,246,.2);">
                <p style="margin:0 0 .6rem;font-size:.78rem;font-weight:700;color:#3b82f6;">
                    📡 Model Status
                </p>
                <p style="margin:0 0 .2rem;font-size:.72rem;color:#94a3b8;">Random Forest</p>
                {_status_dot(rf_ok)}
                <p style="margin:.4rem 0 .2rem;font-size:.72rem;color:#94a3b8;">K-Means Segmentation</p>
                {_status_dot(km_ok)}
                <p style="margin:.4rem 0 .2rem;font-size:.72rem;color:#94a3b8;">Feature Scaler</p>
                {_status_dot(sc_ok)}
                <p style="margin:.4rem 0 .2rem;font-size:.72rem;color:#94a3b8;">Cluster Scaler</p>
                {_status_dot(sck_ok)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Footer ──
        st.markdown(
            """
            <div style="text-align:center;margin-top:1.5rem;">
                <p style="font-size:.7rem;color:#475569;margin:0;">PaySim · 6.35M rows</p>
                <p style="font-size:.7rem;color:#475569;margin:.1rem 0;">ROC-AUC: 0.9991 · F1: 0.9982</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
