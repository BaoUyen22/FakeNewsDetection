"""
Health check endpoints - Kiểm tra trạng thái server và models
"""
from fastapi import APIRouter
from datetime import datetime
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.model_load import get_all_models_status, MODELS

router = APIRouter()


@router.get("/")
async def health_check():
    """
    Basic health check - kiểm tra server đang chạy
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Fake News Detection API"
    }


@router.get("/models")
async def models_health():
    """
    Kiểm tra trạng thái các models đã load
    """
    models_status = get_all_models_status()
    
    # Kiểm tra xem có model nào loaded không
    all_healthy = all(
        status == "loaded" 
        for key, status in models_status.items() 
        if key not in ["total_models"]
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "models": models_status,
        "details": {
            "vectorizer": "required for baseline models",
            "logistic_regression": "Logistic Regression classifier",
            "linear_svc": "Linear SVC classifier", 
            "bert": "DistilBERT/BERT transformer model"
        }
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness check - kiểm tra server sẵn sàng nhận request
    """
    models_status = get_all_models_status()
    
    # Check if có ít nhất 1 model đã load
    has_models = models_status.get("total_models", 0) > 0
    has_vectorizer = "vectorizer" in MODELS
    
    is_ready = has_models and has_vectorizer
    
    return {
        "ready": is_ready,
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": models_status.get("total_models", 0),
        "message": "Server ready" if is_ready else "Models not loaded yet"
    }


@router.get("/live")
async def liveness_check():
    """
    Liveness check - kiểm tra server còn sống
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    } 