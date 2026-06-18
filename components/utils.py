"""
Utility functions for Verify.AI
"""

import hashlib
import time
from datetime import datetime


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between min and max"""
    return max(min_value, min(max_value, value))


def stable_noise(text: str, salt: str) -> float:
    """Generate stable pseudo-random noise from text using hash"""
    digest = hashlib.sha256(f"{salt}|{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def verdict_from_score(fake_score: float) -> tuple[str, float]:
    """Convert fake probability to verdict label and confidence"""
    if fake_score >= 0.60:
        return "Tin giả", fake_score
    if fake_score <= 0.40:
        return "Tin thật", 1 - fake_score
    return "Chưa rõ", 0.50 + abs(fake_score - 0.50)


def format_relative(ts: datetime) -> str:
    """Format timestamp as relative time"""
    seconds = int((datetime.now() - ts).total_seconds())
    if seconds < 60:
        return f"{seconds} giây trước"
    if seconds < 3600:
        return f"{seconds // 60} phút trước"
    return f"{seconds // 3600} giờ trước"


def init_session_state(st, defaults: dict):
    """Initialize Streamlit session state with default values"""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_stats(st, verdict: str):
    """Update statistics based on verdict"""
    stats = st.session_state["stats"]
    stats["total"] += 1
    if verdict == "Tin giả":
        stats["fake"] += 1
    elif verdict == "Tin thật":
        stats["real"] += 1
    else:
        stats["unclear"] += 1
