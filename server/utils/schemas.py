"""
Pydantic schemas - Định nghĩa cấu trúc response cho API
"""
from pydantic import BaseModel, Field


class URLPredictionResponse(BaseModel):
    """Response body cho prediction từ URL"""
    url: str = Field(..., description="URL gốc")
    title: str = Field(..., description="Tiêu đề bài báo")
    description: str = Field(..., description="Mô tả bài báo")
    text: str = Field(..., description="Text đã clean mà model xử lý")
    text_length: int = Field(..., description="Độ dài text đã crawl")
    model: str = Field(..., description="Tên model đã sử dụng")
    label: str = Field(..., description="Nhãn dự đoán: FAKE hoặc REAL")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy (0-1)")
