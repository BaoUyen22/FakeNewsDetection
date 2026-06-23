"""
Model loading and prediction logic
"""

import time
import streamlit as st
from .constants import MODEL_NAME_MAPPING
from .utils import clamp, stable_noise, verdict_from_score


@st.cache_resource
def load_all_predictors():
    """Load all model predictors once and cache them"""
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from predictors import (
            BaselinePredictor,
            DeepLearningPredictor,
            BERTPredictor
        )
        
        baseline = BaselinePredictor(fake_threshold=0.60)
        deep_learning = DeepLearningPredictor()
        bert = BERTPredictor()
        return baseline, deep_learning, bert
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None


def simulate_model_prediction(text: str, model_name: str) -> dict:
    """Fallback simulation if real models fail"""
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

    # Different model characteristics
    model_bias = {
        "Linear SVC": 0.02,
        "Logistic Regression": 0.03,
        "LSTM": 0.05,
        "BERT": 0.06,
    }[model_name]
    score += model_bias
    score += (stable_noise(text, model_name) - 0.5) * 0.26
    score = clamp(score, 0.06, 0.94)

    verdict, confidence = verdict_from_score(score)

    
    # Different processing times for different models
    if model_name == "Linear SVC":
        elapsed = 0.8 + stable_noise(text, "latency") * 0.4
    elif model_name == "Logistic Regression":
        elapsed = 1.0 + stable_noise(text, "latency") * 0.5
    elif model_name == "LSTM":
        elapsed = 2.5 + stable_noise(text, "latency-lstm") * 0.8
    else:  # BERT
        elapsed = 3.2 + stable_noise(text, f"latency-{model_name}") * 1.0

    # Calculate fake/real probs
    if verdict == "Tin giả":
        fake_prob = confidence
        real_prob = 1 - confidence
    elif verdict == "Tin thật":
        real_prob = confidence
        fake_prob = 1 - confidence
    else:
        fake_prob = 0.5
        real_prob = 0.5

    return {
        "model": model_name,
        "verdict": verdict,
        "confidence": confidence,
        "elapsed": elapsed,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
    }


def real_model_prediction(text: str, model_name: str) -> dict:
    """
    Real prediction using trained models from compare_all_models
    
    Args:
        text: Raw text to predict (will be preprocessed by each model's predictor)
        model_name: Display name (e.g. "Linear SVC")
    
    Returns:
        dict with model, verdict, confidence, elapsed
    """
    # Get cached predictors (only loads once due to @st.cache_resource)
    baseline, deep_learning, bert = load_all_predictors()
    
    if not any([baseline, deep_learning, bert]):
        # Fallback to simulation if models not loaded
        return simulate_model_prediction(text, model_name)

    
    try:
        # Validate raw input length
        if not text or len(text.strip()) < 10:
            st.warning("⚠️ Text quá ngắn. Vui lòng nhập thêm nội dung.")
            return None
        
        # Get predictor type and key
        predictor_type, predictor_key = MODEL_NAME_MAPPING.get(model_name, ("baseline", "LinearSVC"))
        
        # Get predictions (each predictor applies its own model-specific preprocessing)
        start_time = time.perf_counter()
        
        if predictor_type == "baseline" and baseline.models:
            results = baseline.predict(text)
            result = results.get(predictor_key, {})
        elif predictor_type == "deep_learning" and deep_learning.model:
            results = deep_learning.predict(text)
            result = results.get(predictor_key, {})
        elif predictor_type == "bert" and bert.model:
            results = bert.predict(text)
            result = results.get(predictor_key, {})
        else:
            st.warning(f"⚠️ Model {model_name} not available. Using simulation.")
            return simulate_model_prediction(text, model_name)
        
        elapsed = time.perf_counter() - start_time
        
        if not result:
            st.warning(f"⚠️ No result from {model_name}. Using simulation.")
            return simulate_model_prediction(text, model_name)
        
        # Map verdict from English to Vietnamese
        verdict_map = {
            "FAKE": "Tin giả",
            "REAL": "Tin thật"
        }
        verdict = verdict_map.get(result["label"], "Chưa rõ")
        
        # Add confidence warning for low confidence predictions
        confidence = result["confidence"]
        if confidence < 0.60:
            # Low confidence - mark as uncertain
            verdict = "Chưa rõ"
            confidence = 0.50 + abs(confidence - 0.50) * 0.5
        
        return {
            "model": model_name,
            "verdict": verdict,
            "confidence": confidence,
            "elapsed": elapsed,
            "fake_prob": result["fake_prob"],
            "real_prob": result["real_prob"],
        }
        
    except Exception as e:
        st.warning(f"⚠️ Model prediction failed: {e}. Using simulation.")
        return simulate_model_prediction(text, model_name)
