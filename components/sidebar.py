"""
Sidebar component with model selection and platform configuration
"""

import streamlit as st
from .constants import MODEL_OPTIONS, PLATFORM_OPTIONS
from .model_loader import load_all_predictors


def render_sidebar():
    """Render the sidebar with branding, model selection, and platform toggles"""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <span class="sidebar-logo-mark">V</span>
          <div>
            <div class="sidebar-logo-title">Verify.AI</div>
            <div class="sidebar-logo-sub">Fake News Scanner</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Show model status
    baseline, deep_learning, bert = load_all_predictors()
    
    if baseline and baseline.models:
        st.sidebar.markdown(
            '<div style="font-size:10px;color:#059669;margin-bottom:0.8rem;">🟢 Models ready</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div style="font-size:10px;color:#DC2626;margin-bottom:0.8rem;">🔴 Models not loaded</div>',
            unsafe_allow_html=True
        )

    st.sidebar.markdown('<div class="section-label">Mô hình</div>', unsafe_allow_html=True)
    st.sidebar.radio(
        "Model",
        MODEL_OPTIONS,
        key="selected_model",
        format_func=lambda name: f"{name} {'⭐' if name == 'BERT' else ''}",
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div class="section-label">Nền tảng quét</div>', unsafe_allow_html=True)
    for key, label, _ in PLATFORM_OPTIONS:
        # Use the checkbox value directly with a unique key
        current_value = st.sidebar.checkbox(
            label,
            value=st.session_state["platforms"].get(key, False),
            key=f"platform_{key}",
        )
        # Update the platforms dictionary
        st.session_state["platforms"][key] = current_value
