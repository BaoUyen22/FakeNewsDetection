"""
Auto scanner tab component with feed and statistics
"""

import time
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
from .constants import (
    PLATFORM_OPTIONS, TOPIC_OPTIONS, INTERVAL_OPTIONS,
    HEADLINES, VERDICT_THEME
)
from .utils import stable_noise, format_relative, update_stats


def pick_verdict(seed: float) -> str:
    """Pick a verdict based on seed value"""
    if seed < 0.42:
        return "Tin thật"
    if seed < 0.79:
        return "Tin giả"
    return "Chưa rõ"


def generate_feed_item() -> dict:
    """Generate a simulated feed item for auto scanning"""
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
    """Check if it's time to refresh the feed based on interval"""
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
        update_stats(st, item["verdict"])


def enable_autorefresh():
    """Enable automatic page refresh for live feed updates"""
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
    """Render statistics summary grid"""
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
    """Render the live feed of scanned items"""
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
    """Render scanner configuration panel"""
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:0;">Nền tảng</div>', unsafe_allow_html=True)
    platform_cols = st.columns(4)
    for col, (key, label, _) in zip(platform_cols, PLATFORM_OPTIONS):
        with col:
            is_on = st.session_state["platforms"].get(key, False)
            st.markdown(
                f'<div class="platform-chip {"on" if is_on else ""}">{label}<br>{"✓" if is_on else ""}</div>',
                unsafe_allow_html=True,
            )
            current_value = st.checkbox(
                "Bật",
                key=f"platform_toggle_{key}",
                value=is_on,
                label_visibility="collapsed",
            )
            st.session_state["platforms"][key] = current_value

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
            update_stats(st, item["verdict"])
    with once_col:
        if st.button("Quét 1 lần", key="scan_once", use_container_width=True):
            item = generate_feed_item()
            st.session_state["feed_items"] = [item] + st.session_state["feed_items"][:39]
            update_stats(st, item["verdict"])
    with stop_col:
        if st.button("Dừng", key="stop_scan", use_container_width=True):
            st.session_state["scan_running"] = False

    st.markdown("</div>", unsafe_allow_html=True)


def render_auto_scanner():
    """Main render function for auto scanner tab"""
    render_scanner_config()
    maybe_refresh_feed()
    render_stat_summary()
    render_feed()
    enable_autorefresh()
