import hashlib
import time
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

TAB_OPTIONS = ["Phân tích thủ công", "Quét tự động"]
MODEL_OPTIONS = ["BaseLine", "TransFormer"]

# ======================================================================
# HEADER COMMENT:
# Chức năng "so sánh 4 mô hình" đang được tắt theo yêu cầu
# Để bật lại: đổi ENABLE_MODEL_COMPARISON = True và dùng lại block render
# trong render_result_card + phần summarize trong render_manual_analysis.
# ======================================================================
ENABLE_MODEL_COMPARISON = False

PLATFORM_OPTIONS = [
    ("x", "X / Twitter", "X"),
    ("rss", "RSS Feed", "RSS"),
    ("facebook", "Facebook", "FB"),
    ("web", "Web crawl", "WEB"),
]
TOPIC_OPTIONS = [
    "Chính trị",
    "Y tế / COVID",
    "Khoa học",
    "Kinh tế",
    "Môi trường",
    "Giáo dục",
    "Công nghệ",
]
INTERVAL_OPTIONS = {
    "5 phút": 5,
    "15 phút": 15,
    "1 giờ": 60,
    "Thủ công": 0,
}
HEADLINES = {
    "Chính trị": [
        "Nguồn ẩn danh nói có thay đổi chính sách qua đêm.",
        "Bài đăng cắt ghép phát biểu lãnh đạo gây tranh luận mạnh.",
        "Nội dung chưa kiểm chứng về nghị quyết mới lan truyền nhanh.",
    ],
    "Y tế / COVID": [
        "Tin đồn về thuốc tự pha chữa bệnh đang tăng nhanh trên mạng.",
        "Video cũ bị gắn sai ngữ cảnh về vaccine tại bệnh viện trung ương.",
        "Bản tin chưa xác minh khẳng định có biến thể mới tại nhiều tỉnh.",
    ],
    "Khoa học": [
        "Bài viết nói máy phát điện vĩnh cửu đã được chứng minh thành công.",
        "Ảnh thiên văn chỉnh sửa quá mức bị chia sẻ như nghiên cứu thật.",
        "Tin về thí nghiệm gây sốc thiếu dẫn nguồn từ tạp chí uy tín.",
    ],
    "Kinh tế": [
        "Tin nóng về ngân hàng ngừng chi trả bị lan rộng trên các nhóm kín.",
        "Bài đăng dự báo tỷ giá tăng sốc kèm số liệu không rõ nguồn.",
        "Tuyên bố cắt giảm thuế diện rộng chưa có văn bản chính thức.",
    ],
    "Môi trường": [
        "Tin cảnh báo ô nhiễm nguồn nước dùng hình ảnh từ năm trước.",
        "Video đám cháy rừng ở nước ngoài bị gắn thành sự kiện trong nước.",
        "Nội dung thời tiết cực đoan được chia sẻ lại với tiêu đề sai lệch.",
    ],
    "Giáo dục": [
        "Tin giả về lịch thi quốc gia bị chỉnh sửa lan trên nhóm phụ huynh.",
        "Ảnh thông báo tuyển sinh không dấu xác thực được chia sẻ rộng rãi.",
        "Thông tin học phí tăng đột biến chưa có văn bản chính thức.",
    ],
    "Công nghệ": [
        "Tin đồn rò rỉ dữ liệu từ ứng dụng lớn chưa được bên vận hành xác nhận.",
        "Bài viết dùng ảnh giả để nói về điện thoại mới chưa ra mắt.",
        "Nội dung quảng cáo AI thần kỳ không có bằng chứng kỹ thuật.",
    ],
}
VERDICT_THEME = {
    "Tin giả": {
        "bg": "#FCEBEB",
        "border": "#F09595",
        "text": "#791F1F",
        "bar": "#A32D2D",
    },
    "Tin thật": {
        "bg": "#E1F5EE",
        "border": "#5DCAA5",
        "text": "#085041",
        "bar": "#0F6E56",
    },
    "Chưa rõ": {
        "bg": "#FAEEDA",
        "border": "#EF9F27",
        "text": "#633806",
        "bar": "#854F0B",
    },
}


def inject_styles():
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


def init_session():
    defaults = {
        "active_tab": TAB_OPTIONS[0],
        "selected_model": MODEL_OPTIONS[0],
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
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def stable_noise(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}|{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def verdict_from_score(fake_score: float) -> tuple[str, float]:
    if fake_score >= 0.60:
        return "Tin giả", fake_score
    if fake_score <= 0.40:
        return "Tin thật", 1 - fake_score
    return "Chưa rõ", 0.50 + abs(fake_score - 0.50)


def simulate_model_prediction(text: str, model_name: str) -> dict:
    lower_text = text.lower()
    fake_tokens = ["giật gân", "bí mật", "chấn động", "fake", "giả", "100%"]
    real_tokens = ["theo báo cáo", "nguồn chính thức", "xác minh", "dữ liệu", "nghiên cứu"]

    score = 0.52
    for token in fake_tokens:
        if token in lower_text:
            score += 0.05
    for token in real_tokens:
        if token in lower_text:
            score -= 0.04

    model_bias = {
        "BaseLine": 0.03,
        "TransFormer": 0.06,
      
    }[model_name]
    score += model_bias
    score += (stable_noise(text, model_name) - 0.5) * 0.26
    score = clamp(score, 0.06, 0.94)

    verdict, confidence = verdict_from_score(score)
    if model_name == "BaseLine":
        elapsed = 1.1 + stable_noise(text, "latency") * 0.6
    else:
        elapsed = 0.03 + stable_noise(text, f"latency-{model_name}") * 0.07

    return {
        "model": model_name,
        "verdict": verdict,
        "confidence": confidence,
        "elapsed": elapsed,
    }


def summarize_result(predictions: list[dict]) -> dict:
    count = {"Tin giả": 0, "Tin thật": 0, "Chưa rõ": 0}
    for item in predictions:
        count[item["verdict"]] += 1

    majority_verdict = max(count, key=count.get)
    majority_count = count[majority_verdict]
    agree = [item["confidence"] for item in predictions if item["verdict"] == majority_verdict]
    avg_confidence = sum(agree) / len(agree) if agree else 0.55

    contradictions = [item for item in predictions if item["verdict"] != majority_verdict]
    note = f"Đa số {majority_count}/{len(predictions)}: {majority_verdict}"
    if contradictions:
        strongest = max(contradictions, key=lambda x: x["confidence"])
        note = f"{note} · {strongest['model']} trái chiều {int(strongest['confidence'] * 100)}%"

    return {
        "verdict": majority_verdict,
        "confidence": clamp(avg_confidence, 0.5, 0.99),
        "note": note,
    }


def update_stats(verdict: str):
    stats = st.session_state["stats"]
    stats["total"] += 1
    if verdict == "Tin giả":
        stats["fake"] += 1
    elif verdict == "Tin thật":
        stats["real"] += 1
    else:
        stats["unclear"] += 1


def render_topbar():
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


def render_sidebar():
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

    st.sidebar.markdown('<div class="section-label">Mô hình</div>', unsafe_allow_html=True)
    st.sidebar.radio(
        "Model",
        MODEL_OPTIONS,
        key="selected_model",
        format_func=lambda name: "DistilBERT ★" if name == "DistilBERT" else name,
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div class="section-label">Nền tảng quét</div>', unsafe_allow_html=True)
    for key, label, _ in PLATFORM_OPTIONS:
        st.session_state["platforms"][key] = st.sidebar.checkbox(
            label,
            value=st.session_state["platforms"][key],
            key=f"sidebar_platform_{key}",
        )

    # stats = st.session_state["stats"]
    # st.sidebar.markdown('<div class="section-label">Phiên này</div>', unsafe_allow_html=True)
    # st.sidebar.markdown(
    #     f"""
    #     <div class="soft-panel">
    #       <div class="sidebar-row"><span class="name">Tổng quét</span><span class="value">{stats['total']}</span></div>
    #       <div class="sidebar-row"><span class="name">Tin giả</span><span class="value" style="color:#A32D2D;">{stats['fake']}</span></div>
    #       <div class="sidebar-row"><span class="name">Tin thật</span><span class="value" style="color:#0F6E56;">{stats['real']}</span></div>
    #       <div class="sidebar-row"><span class="name">Chưa rõ</span><span class="value" style="color:#854F0B;">{stats['unclear']}</span></div>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )


def build_compare_table(predictions: list[dict]) -> str:
    rows = []
    for item in predictions:
        theme = VERDICT_THEME[item["verdict"]]
        star = "★ " if item["model"] == "DistilBERT" else ""
        confidence_pct = int(item["confidence"] * 100)
        latency = f"{item['elapsed']:.1f}s" if item["elapsed"] >= 0.1 else "<0.1s"
        rows.append(
            "<tr>"
            f"<td>{star}{item['model']}</td>"
            f"<td style='color:{theme['text']};'>{item['verdict']}</td>"
            f"<td>{confidence_pct}%<span class='mini-track'><span class='mini-fill' style='width:{confidence_pct}%;background:{theme['bar']};display:block;'></span></span></td>"
            f"<td>{latency}</td>"
            "</tr>"
        )

    return (
        "<div class='table-wrap'>"
        "<table class='compare-table'>"
        "<thead><tr><th>Mô hình</th><th>Verdict</th><th>Độ tin cậy</th><th>Xử lý</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def render_result_card(result: dict):
    primary = result["primary"]
    theme = VERDICT_THEME[primary["verdict"]]
    confidence_pct = int(primary["confidence"] * 100)
    chip_icon = "⚠" if primary["verdict"] == "Tin giả" else ("✓" if primary["verdict"] == "Tin thật" else "•")

    st.markdown(
        f"""
        <div class="result-card">
          <span class="verdict-chip" style="background:{theme['bg']};border-color:{theme['border']};color:{theme['text']};">
            {chip_icon} {primary['verdict']}
          </span>
          <div class="confidence-row">
            <span style="font-size:11px;color:#6B7280;">Độ tin cậy</span>
            <span class="confidence-value" style="color:{theme['text']};">{confidence_pct}%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" style="width:{confidence_pct}%;background:{theme['bar']};"></div>
          </div>
          <div class="meta-line">Mô hình chọn: {st.session_state['selected_model']} · Xử lý: {result['duration']:.1f}s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_manual_analysis():
    text = st.text_area(
        "Nội dung",
        key="manual_text",
        label_visibility="collapsed",
        height=120,
        max_chars=5000,
        placeholder="Dán nội dung bài báo cần kiểm tra vào đây... ",
    )
    char_count = len(text)
    counter_col, clear_col = st.columns([0.75, 0.25])
    with counter_col:
        st.markdown(f'<div class="counter">{char_count} / 5000 ký tự</div>', unsafe_allow_html=True)
    with clear_col:
        if text and st.button("Xóa", key="clear_manual_text", use_container_width=True):
            st.session_state["manual_text"] = ""
            st.session_state["manual_result"] = None
            st.rerun()

    analyze_clicked = st.button(
        "⊙ Phân tích ngay",
        key="analyze_now",
        type="primary",
        disabled=char_count < 50,
        use_container_width=True,
    )

    if char_count < 50:
        st.caption("Nhập tối thiểu 50 ký tự để bật nút phân tích.")

    if analyze_clicked and text.strip():
        with st.spinner("Đang phân tích..."):
            start = time.perf_counter()
            if ENABLE_MODEL_COMPARISON:
                predictions = [simulate_model_prediction(text, model_name) for model_name in MODEL_OPTIONS]
                primary = next(
                    item for item in predictions if item["model"] == st.session_state["selected_model"]
                )
                summary = summarize_result(predictions)
                st.session_state["manual_result"] = {
                    "primary": primary,
                    "summary": summary,
                    "predictions": predictions,
                    "duration": time.perf_counter() - start,
                }
                update_stats(primary["verdict"])
            else:
                primary = simulate_model_prediction(text, st.session_state["selected_model"])
                st.session_state["manual_result"] = {
                    "primary": primary,
                    "duration": time.perf_counter() - start,
                }
                update_stats(primary["verdict"])

    if st.session_state["manual_result"]:
        render_result_card(st.session_state["manual_result"])


def pick_verdict(seed: float) -> str:
    if seed < 0.42:
        return "Tin thật"
    if seed < 0.79:
        return "Tin giả"
    return "Chưa rõ"


def format_relative(ts: datetime) -> str:
    seconds = int((datetime.now() - ts).total_seconds())
    if seconds < 60:
        return f"{seconds} giây trước"
    if seconds < 3600:
        return f"{seconds // 60} phút trước"
    return f"{seconds // 3600} giờ trước"


def generate_feed_item() -> dict:
    st.session_state["scan_counter"] += 1
    seed = stable_noise(str(st.session_state["scan_counter"]), "feed-seed")
    active_platforms = [key for key, _, _ in PLATFORM_OPTIONS if st.session_state["platforms"][key]]
    if not active_platforms:
        active_platforms = ["x"]

    platform_key = active_platforms[int(seed * len(active_platforms)) % len(active_platforms)]
    platform_label = next(label for key, label, _ in PLATFORM_OPTIONS if key == platform_key)
    icon = next(icon for key, _, icon in PLATFORM_OPTIONS if key == platform_key)

    selected_topics = st.session_state["topics"] or TOPIC_OPTIONS
    topic = selected_topics[int(seed * 1000) % len(selected_topics)]
    headline = HEADLINES[topic][int(seed * 100) % len(HEADLINES[topic])]
    verdict = pick_verdict(seed)
    confidence = int((0.56 + seed * 0.4) * 100)
    model = st.session_state["selected_model"]

    return {
        "platform": platform_label,
        "icon": icon,
        "topic": topic,
        "headline": headline,
        "verdict": verdict,
        "confidence": confidence,
        "model": model,
        "timestamp": datetime.now(),
    }


def maybe_refresh_feed():
    if not st.session_state["scan_running"]:
        return
    minutes = INTERVAL_OPTIONS[st.session_state["scan_interval"]]
    if minutes <= 0:
        return

    now = time.time()
    if now - st.session_state["last_scan_ts"] >= minutes * 60:
        item = generate_feed_item()
        st.session_state["feed_items"] = [item] + st.session_state["feed_items"][:39]
        st.session_state["last_scan_ts"] = now
        update_stats(item["verdict"])


def enable_autorefresh():
    if not st.session_state["scan_running"]:
        return
    minutes = INTERVAL_OPTIONS[st.session_state["scan_interval"]]
    if minutes <= 0:
        return
    components.html(
        f"""
        <script>
          const rootWindow = window.parent;
          setTimeout(() => {{
            rootWindow.location.reload();
          }}, {minutes * 60 * 1000});
        </script>
        """,
        height=0,
    )


def render_stat_summary():
    stats = st.session_state["stats"]
    blocks = [
        ("Đã quét", stats["total"], "#111827"),
        ("Tin giả", stats["fake"], "#A32D2D"),
        ("Tin thật", stats["real"], "#0F6E56"),
        ("Chưa rõ", stats["unclear"], "#854F0B"),
    ]
    html_blocks = []
    for label, value, color in blocks:
        html_blocks.append(
            "<div class='stat-box'>"
            f"<div class='stat-value' style='color:{color};'>{value}</div>"
            f"<div class='stat-label'>{label}</div>"
            "</div>"
        )
    st.markdown(f"<div class='stat-grid'>{''.join(html_blocks)}</div>", unsafe_allow_html=True)


def render_feed():
    st.markdown('<div class="section-label">Feed thời gian thực</div>', unsafe_allow_html=True)
    if not st.session_state["feed_items"]:
        st.markdown(
            """
            <div class="panel">
              <span style="font-size:11px;color:#6B7280;">Chưa có kết quả quét. Bấm "Bắt đầu quét" để nhận feed.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for item in st.session_state["feed_items"][:18]:
        theme = VERDICT_THEME[item["verdict"]]
        st.markdown(
            f"""
            <div class="feed-item" style="border-left-color:{theme['border']};">
              <div class="feed-top">{item['icon']} · {item['platform']}</div>
              <div class="feed-score" style="color:{theme['bar']};">{item['confidence']}%</div>
              <div class="feed-headline">{item['headline']}</div>
              <div class="tag-row">
                <span class="tag" style="background:{theme['bg']};border-color:{theme['border']};color:{theme['text']};">{item['verdict']}</span>
                <span class="tag">{item['topic']}</span>
                <span class="tag">{item['model']} {item['confidence']}%</span>
              </div>
              <div class="feed-time">{item['timestamp'].strftime('%H:%M')} · {format_relative(item['timestamp'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_scanner_config():
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:0;">Nền tảng</div>', unsafe_allow_html=True)
    platform_cols = st.columns(4)
    for col, (key, label, _) in zip(platform_cols, PLATFORM_OPTIONS):
        with col:
            is_on = st.session_state["platforms"][key]
            st.markdown(
                f'<div class="platform-chip {"on" if is_on else ""}">{label}<br>{"✓" if is_on else ""}</div>',
                unsafe_allow_html=True,
            )
            st.session_state["platforms"][key] = st.checkbox(
                "Bật",
                key=f"platform_toggle_{key}",
                value=is_on,
                label_visibility="collapsed",
            )

    st.markdown('<div class="section-label">Chủ đề</div>', unsafe_allow_html=True)
    st.session_state["topics"] = st.multiselect(
        "Topics",
        TOPIC_OPTIONS,
        default=st.session_state["topics"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">Tần suất</div>', unsafe_allow_html=True)
    st.session_state["scan_interval"] = st.radio(
        "Interval",
        list(INTERVAL_OPTIONS.keys()),
        index=list(INTERVAL_OPTIONS.keys()).index(st.session_state["scan_interval"]),
        horizontal=True,
        label_visibility="collapsed",
        key="scan_interval_radio",
    )

    start_col, once_col, stop_col = st.columns([2.2, 1.6, 1.2])
    with start_col:
        if st.button("Bắt đầu quét", key="start_scan", use_container_width=True):
            st.session_state["scan_running"] = True
            st.session_state["last_scan_ts"] = 0.0
            item = generate_feed_item()
            st.session_state["feed_items"] = [item] + st.session_state["feed_items"][:39]
            update_stats(item["verdict"])
    with once_col:
        if st.button("Quét 1 lần", key="scan_once", use_container_width=True):
            item = generate_feed_item()
            st.session_state["feed_items"] = [item] + st.session_state["feed_items"][:39]
            update_stats(item["verdict"])
    with stop_col:
        if st.button("Dừng", key="stop_scan", use_container_width=True):
            st.session_state["scan_running"] = False

    st.markdown("</div>", unsafe_allow_html=True)


def render_auto_scanner():
    render_scanner_config()
    maybe_refresh_feed()
    render_stat_summary()
    render_feed()
    enable_autorefresh()


def main():
    st.set_page_config(page_title="Verify.AI", page_icon="📰", layout="wide")
    init_session()
    inject_styles()
    render_topbar()
    render_sidebar()

    if st.session_state["active_tab"] == TAB_OPTIONS[0]:
        render_manual_analysis()
    else:
        render_auto_scanner()


if __name__ == "__main__":
    main()
