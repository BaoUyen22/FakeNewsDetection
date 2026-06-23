"""
Pydantic schemas - Định nghĩa cấu trúc response cho API
"""
from pydantic import BaseModel, Field
from typing import Optional


class PredictionDetails(BaseModel):
    """Chi tiết prediction scores"""
    fake_probability: float = Field(..., description="Xác suất là tin FAKE (0-1)")
    real_probability: float = Field(..., description="Xác suất là tin REAL (0-1)")


class URLPredictionResponse(BaseModel):
    """Response body cho prediction từ URL"""
    url: str = Field(..., description="URL gốc")
    title: str = Field(..., description="Tiêu đề bài báo")
    description: str = Field(..., description="Mô tả bài báo")
    text_length: int = Field(..., description="Độ dài text đã crawl (chars)")
    processed_text: str = Field(..., description="Text đã xử lý đưa vào model")
    processed_text_length: int = Field(..., description="Độ dài text đã xử lý (chars)")
    model: str = Field(..., description="Tên model đã sử dụng")
    label: str = Field(..., description="Nhãn dự đoán: FAKE hoặc REAL")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy của nhãn dự đoán (0-1)")
    prediction_details: PredictionDetails = Field(..., description="Chi tiết xác suất cho cả 2 class")


class TextPredictionRequest(BaseModel):
    """Request body cho prediction từ text."""
    text: str = Field(..., description="Raw news text để dự đoán")


class TextPredictionResponse(BaseModel):
    """Response body cho prediction từ text."""
    input_text: str = Field(..., description="Raw input text")
    processed_text: str = Field(..., description="Text đã xử lý đưa vào model")
    processed_text_length: int = Field(..., description="Độ dài text đã xử lý (chars)")
    model: str = Field(..., description="Tên model đã sử dụng")
    label: str = Field(..., description="Nhãn dự đoán: FAKE hoặc REAL")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy của nhãn dự đoán (0-1)")
    prediction_details: PredictionDetails = Field(..., description="Chi tiết xác suất cho cả 2 class")
