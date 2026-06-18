"""
Result card component for displaying prediction results
"""

import streamlit as st
from .constants import VERDICT_THEME


def render_result_card(result: dict):
    """
    Render prediction result card with verdict, confidence, and metadata
    
    Args:
        result: Dict containing primary prediction and optional comparison data
    """
    primary = result["primary"]
    theme = VERDICT_THEME[primary["verdict"]]
    confidence_pct = int(primary["confidence"] * 100)
    chip_icon = "⚠" if primary["verdict"] == "Tin giả" else ("✓" if primary["verdict"] == "Tin thật" else "•")
    
    # Get probabilities if available
    fake_pct = int(primary.get("fake_prob", 0) * 100) if "fake_prob" in primary else confidence_pct if primary["verdict"] == "Tin giả" else (100 - confidence_pct)
    real_pct = int(primary.get("real_prob", 0) * 100) if "real_prob" in primary else (100 - fake_pct)
    
    # Confidence warning
    confidence_warning = ""
    if confidence_pct < 70:
        confidence_warning = """
        <div style="background:#FEF3C7;border:1px solid #F59E0B;border-radius:8px;padding:0.5rem 0.6rem;margin-top:0.6rem;font-size:11px;color:#92400E;">
            ⚠️ <strong>Độ tin cậy thấp</strong> - Kết quả có thể không chính xác. Model chưa chắc chắn về văn bản này. Nên kiểm chứng thêm từ nguồn khác.
        </div>
        """
    elif primary["verdict"] == "Chưa rõ":
        confidence_warning = """
        <div style="background:#F3F4F6;border:1px solid #9CA3AF;border-radius:8px;padding:0.5rem 0.6rem;margin-top:0.6rem;font-size:11px;color:#374151;">
            ℹ️ <strong>Không thể kết luận</strong> - Văn bản này có đặc điểm không rõ ràng hoặc nằm ngoài phạm vi training data. Model không đủ tự tin để phân loại.
        </div>
        """

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
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#6B7280;margin-bottom:0.5rem;">
            <span>🚫 Tin giả: <strong style="color:#791F1F;">{fake_pct}%</strong></span>
            <span>✅ Tin thật: <strong style="color:#085041;">{real_pct}%</strong></span>
          </div>
          <div class="meta-line">Mô hình: {st.session_state['selected_model']} · Xử lý: {result['duration']:.2f}s</div>
          {confidence_warning}
        </div>
        """,
        unsafe_allow_html=True,
    )
