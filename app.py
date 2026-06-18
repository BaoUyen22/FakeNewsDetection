"""
Verify.AI - Fake News Detection Application
Main entry point for the Streamlit app
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root and test dir to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "test"))

# Import components
from components import (
    inject_styles,
    render_topbar,
    render_sidebar,
    render_manual_analysis,
    render_auto_scanner,
)
from components.constants import TAB_OPTIONS
from components.utils import init_session_state
from components.model_loader import load_all_predictors


def get_default_session_state() -> dict:
    """Get default session state configuration"""
    return {
        "active_tab": TAB_OPTIONS[0],
        "selected_model": "Linear SVC", 
        "manual_text": "",
        "manual_result": None,
        "scan_running": False,
        "platforms": {
            "x": True,
            "rss": True,
            "facebook": False,
            "web": False,
        },
        "topics": ["Chính trị", "Y tế / COVID", "Khoa học", "Môi trường"],
        "scan_interval": "15 phút",
        "feed_items": [],
        "stats": {
            "total": 147,
            "fake": 53,
            "real": 82,
            "unclear": 12,
        },
        "last_scan_ts": 0.0,
        "scan_counter": 0,
    }


def main():
    """Main application entry point"""
    # Configure page
    st.set_page_config(
        page_title="Verify.AI",
        page_icon="📰",
        layout="wide"
    )
    
    # Pre-load models (cached)
    load_all_predictors()
    
    # Initialize session state
    init_session_state(st, get_default_session_state())
    
    # Inject custom CSS
    inject_styles()
    
    # Render UI components
    render_topbar()
    render_sidebar()

    # Render active tab
    if st.session_state["active_tab"] == TAB_OPTIONS[0]:
        render_manual_analysis()
    else:
        render_auto_scanner()


if __name__ == "__main__":
    main()
