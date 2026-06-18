"""
Top navigation bar component
"""

import streamlit as st
from .constants import TAB_OPTIONS


def render_topbar():
    """Render the top navigation bar with logo and tabs"""
    st.markdown(
        """
        <div class="topbar">
          <div class="logo-wrap">
            <span class="logo-mark">V</span>
            <div class="logo">Verify.AI</div>
          </div>
          <div class="live-wrap"><span class="live-dot"></span>Live · v2.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.radio(
        "Điều hướng",
        TAB_OPTIONS,
        key="active_tab",
        horizontal=True,
        label_visibility="collapsed",
    )
