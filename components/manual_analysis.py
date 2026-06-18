"""
Manual analysis tab component
"""

import time
import streamlit as st
from .constants import ENABLE_MODEL_COMPARISON, MODEL_OPTIONS
from .result_card import render_result_card
from .model_loader import real_model_prediction
from .utils import update_stats
from utils.preprocessor import detect_vietnamese, get_vietnamese_ratio


def summarize_result(predictions: list[dict]) -> dict:
    """Summarize multiple model predictions into a consensus verdict"""
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
        "confidence": avg_confidence,
        "note": note,
    }


def render_manual_analysis():
    """Render the manual analysis tab with text input and prediction"""
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
            # Delete from session_state instead of setting to ""
            del st.session_state["manual_text"]
            if "manual_result" in st.session_state:
                del st.session_state["manual_result"]
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
    
    # Check for Vietnamese text
    if text and detect_vietnamese(text):
        vietnamese_ratio = get_vietnamese_ratio(text)
        
        if vietnamese_ratio > 0.3:  # More than 30% Vietnamese words
            st.warning(
                f"**Cảnh báo:** Phát hiện {vietnamese_ratio*100:.0f}% nội dung tiếng Việt!\n\n"
                "Model chỉ được huấn luyện trên **tiếng Anh**. "
                "Kết quả phân tích có thể **không chính xác** với văn bản tiếng Việt.\n\n"
                "**Khuyến nghị:** Dịch sang tiếng Anh trước khi phân tích để có kết quả tốt nhất.",
                icon="⚠️"
            )

    if analyze_clicked and text.strip():
        with st.spinner("Đang phân tích..."):
            start = time.perf_counter()
            if ENABLE_MODEL_COMPARISON:
                # Use real model prediction for comparison
                predictions = []
                for model_name in MODEL_OPTIONS:
                    pred = real_model_prediction(text, model_name)
                    if pred:  # Only add if prediction successful
                        predictions.append(pred)
                
                if not predictions:
                    st.error("❌ Không thể phân tích với bất kỳ model nào. Vui lòng thử lại.")
                else:
                    primary = next(
                        (item for item in predictions if item["model"] == st.session_state["selected_model"]),
                        predictions[0]  # Fallback to first prediction
                    )
                    summary = summarize_result(predictions)
                    st.session_state["manual_result"] = {
                        "primary": primary,
                        "summary": summary,
                        "predictions": predictions,
                        "duration": time.perf_counter() - start,
                    }
                    update_stats(st, primary["verdict"])
            else:
                # Use real model prediction for single model
                primary = real_model_prediction(text, st.session_state["selected_model"])
                if primary:
                    st.session_state["manual_result"] = {
                        "primary": primary,
                        "duration": time.perf_counter() - start,
                    }
                    update_stats(st, primary["verdict"])
                else:
                    st.error("❌ Không thể phân tích. Vui lòng kiểm tra lại nội dung.")

    if st.session_state["manual_result"]:
        render_result_card(st.session_state["manual_result"])
