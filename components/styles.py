"""
CSS styles for Verify.AI Streamlit app
"""

import streamlit as st


def inject_styles():
    """Inject custom CSS styles into the Streamlit app"""
    st.markdown(
        """
        <style>
          :root {
            --bg-page: #F8F9FA;
            --bg-primary: #FFFFFF;
            --bg-secondary: #F3F4F6;
            --accent-teal: #1D9E75;
            --accent-teal-light: #E1F5EE;
            --accent-teal-border: #5DCAA5;
            --text-primary: #111827;
            --text-secondary: #6B7280;
            --text-tertiary: #9CA3AF;
            --border: rgba(0, 0, 0, 0.08);
            --border-medium: rgba(0, 0, 0, 0.15);
          }

          .stApp {
            background: var(--bg-page);
            color: var(--text-primary);
          }

          header[data-testid="stHeader"] {
            background: transparent;
          }

          div[data-testid="stToolbar"],
          div[data-testid="stDecoration"],
          div[data-testid="stAppDeployButton"] {
            display: none !important;
          }

          .block-container {
            padding-top: 1rem;
            padding-bottom: 1.2rem;
            max-width: 1120px;
          }

          div[data-testid="stSidebar"] {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
          }

          div[data-testid="stSidebarUserContent"],
          div[data-testid="stSidebarContent"] {
            padding-top: 0.45rem;
          }

          div[data-testid="stSidebar"] > div:first-child {
            width: 156px;
            min-width: 156px;
            max-width: 156px;
          }

          .topbar {
            height: 48px;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--bg-primary);
            padding: 0.5rem 0.9rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.7rem;
          }

          .logo-wrap {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
          }

          .logo-mark {
            width: 20px;
            height: 20px;
            border-radius: 6px;
            background: linear-gradient(135deg, #1D9E75 0%, #5DCAA5 100%);
            color: #E1F5EE;
            font-size: 11px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
          }

          .logo {
            font-size: 15px;
            font-weight: 500;
          }

          .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.25rem 0 0.5rem 0;
            margin-bottom: 0.28rem;
          }

          .sidebar-logo-mark {
            width: 34px;
            height: 34px;
            border-radius: 11px;
            background: linear-gradient(135deg, #1D9E75 0%, #5DCAA5 100%);
            color: #E1F5EE;
            font-size: 18px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
          }

          .sidebar-logo-title {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.02;
          }

          .sidebar-logo-sub {
            font-size: 13px;
            color: var(--text-tertiary);
            line-height: 1.05;
            margin-top: 2px;
          }

          .live-wrap {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--text-secondary);
            font-size: 11px;
          }

          .live-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--accent-teal);
            box-shadow: 0 0 0 0 rgba(29, 158, 117, 0.45);
            animation: pulse-dot 1.4s ease-out infinite;
          }

          @keyframes pulse-dot {
            0% { box-shadow: 0 0 0 0 rgba(29, 158, 117, 0.45); }
            100% { box-shadow: 0 0 0 8px rgba(29, 158, 117, 0); }
          }

          .section-label {
            color: var(--text-secondary);
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0.2rem 0 0.35rem 0;
          }

          .panel {
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.8rem;
            margin-bottom: 0.7rem;
          }

          .soft-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.65rem;
            margin-bottom: 0.6rem;
          }

          .counter {
            color: var(--text-tertiary);
            font-size: 10px;
            margin-bottom: 0.45rem;
          }

          .result-card {
            border-radius: 12px;
            border: 1px solid var(--border);
            background: var(--bg-primary);
            padding: 0.9rem;
            margin-top: 0.65rem;
            animation: fade-up 0.24s ease-out;
          }

          @keyframes fade-up {
            from { transform: translateY(4px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
          }

          .verdict-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            border-radius: 999px;
            padding: 0.2rem 0.65rem;
            font-size: 11px;
            font-weight: 500;
            border: 1px solid transparent;
            margin-bottom: 0.4rem;
          }

          .confidence-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 0.35rem;
          }

          .confidence-value {
            font-size: 18px;
            font-weight: 500;
          }

          .progress-track {
            height: 5px;
            border-radius: 3px;
            background: var(--bg-secondary);
            overflow: hidden;
            margin-bottom: 0.65rem;
          }

          .progress-fill {
            height: 100%;
            border-radius: 3px;
            transform-origin: left;
            animation: grow 0.5s ease-out;
          }

          @keyframes grow {
            from { transform: scaleX(0); }
            to { transform: scaleX(1); }
          }

          .meta-line {
            font-size: 10px;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
          }

          .table-wrap {
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 0.5rem;
          }

          .compare-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
          }

          .compare-table th,
          .compare-table td {
            text-align: left;
            padding: 0.4rem 0.45rem;
            border-bottom: 1px solid var(--border);
          }

          .compare-table th {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-secondary);
            font-weight: 500;
            background: var(--bg-secondary);
          }

          .compare-table tr:last-child td {
            border-bottom: none;
          }

          .mini-track {
            width: 62px;
            height: 4px;
            border-radius: 3px;
            background: var(--bg-secondary);
            overflow: hidden;
            display: inline-block;
            margin-left: 0.4rem;
          }

          .mini-fill {
            height: 100%;
            border-radius: 3px;
          }

          .note {
            font-size: 10px;
            color: var(--text-secondary);
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 0.4rem 0.5rem;
          }

          .platform-chip {
            border-radius: 10px;
            padding: 0.48rem 0.35rem;
            font-size: 10px;
            text-align: center;
            border: 1px solid var(--border);
            background: var(--bg-primary);
            color: var(--text-secondary);
            margin-bottom: 0.35rem;
          }

          .platform-chip.on {
            border-color: var(--accent-teal-border);
            background: var(--accent-teal-light);
            color: #085041;
          }

          .stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.45rem 0 0.65rem 0;
          }

          .stat-box {
            border-radius: 10px;
            background: var(--bg-secondary);
            text-align: center;
            padding: 0.65rem 0.3rem;
          }

          .stat-value {
            font-size: 20px;
            font-weight: 500;
            line-height: 1.1;
          }

          .stat-label {
            font-size: 10px;
            color: var(--text-secondary);
            margin-top: 0.18rem;
          }

          .feed-item {
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--bg-primary);
            padding: 0.62rem;
            margin-bottom: 0.45rem;
            border-left-width: 2px;
            animation: feed-enter 0.28s ease-out;
          }

          .feed-item:hover {
            background: var(--bg-secondary);
          }

          @keyframes feed-enter {
            from { opacity: 0; transform: translateY(-3px); }
            to { opacity: 1; transform: translateY(0); }
          }

          .feed-top {
            font-size: 10px;
            color: var(--text-secondary);
            margin-bottom: 0.2rem;
          }

          .feed-score {
            font-size: 10px;
            font-weight: 700;
          }

          .feed-headline {
            font-size: 11px;
            color: var(--text-primary);
            line-height: 1.45;
            margin-bottom: 0.32rem;
          }

          .tag-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.25rem;
            margin-bottom: 0.25rem;
          }

          .tag {
            padding: 0.12rem 0.44rem;
            border-radius: 999px;
            font-size: 9px;
            border: 1px solid var(--border-medium);
          }

          .feed-time {
            font-size: 9px;
            color: var(--text-tertiary);
          }

          .sidebar-row {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            margin-bottom: 0.2rem;
          }

          .sidebar-row .name {
            color: var(--text-secondary);
          }

          .sidebar-row .value {
            color: var(--text-primary);
            font-weight: 500;
          }

          .stTextArea textarea {
            min-height: 80px !important;
            border-radius: 10px !important;
            border: 1px solid var(--border) !important;
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            font-size: 12px !important;
          }

          .stTextArea textarea:focus {
            border: 1.5px solid var(--accent-teal-border) !important;
            box-shadow: none !important;
          }

          .stButton button {
            border-radius: 10px;
            font-size: 12px;
            font-weight: 500;
          }

          .stButton button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
          }

          .stButton button[kind="primary"] {
            background: var(--accent-teal);
            border-color: var(--accent-teal);
            color: #E1F5EE;
            height: 40px;
          }

          .stButton button[kind="primary"]:active {
            transform: scale(0.98);
          }

          .stRadio > div[role="radiogroup"] {
            gap: 0.3rem;
          }

          .stRadio > div[role="radiogroup"] label[data-baseweb="radio"] {
            margin-right: 0 !important;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--bg-primary);
            padding: 0.32rem 0.62rem;
            min-height: auto;
            cursor: pointer;
          }

          .stRadio > div[role="radiogroup"] label[data-baseweb="radio"] span {
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: 400;
          }

          .stRadio > div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] {
            border-color: var(--accent-teal-border);
            background: var(--accent-teal-light);
          }

          .stRadio > div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] span {
            color: #085041;
            font-weight: 500;
          }

          div[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-baseweb="radio"] {
            border: 1px solid var(--border);
            background: var(--bg-primary);
            border-radius: 14px;
            width: 100%;
            padding: 0.35rem 0.55rem;
            margin-bottom: 0.12rem;
          }

          div[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-baseweb="radio"] span {
            font-size: 11px;
            color: var(--text-primary);
            font-weight: 500;
          }

          div[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] {
            border-color: var(--accent-teal-border);
            background: var(--accent-teal-light);
          }

          div[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] span {
            color: #085041;
          }

          @media (max-width: 960px) {
            div[data-testid="stSidebar"] > div:first-child {
              width: 100%;
              min-width: 100%;
              max-width: 100%;
            }

            .stat-grid {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
