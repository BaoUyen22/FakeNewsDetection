"""
FastAPI Main Application
Server API cho Fake News Detection với multiple ML models
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.model_load import load_all_models
from routers import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager - chạy khi app startup và shutdown
    """
    print("\n🚀 Starting Fake News Detection API...")
    await load_all_models()
    print("✓ Models loaded successfully\n")
    yield
    print("\n⏹ Server shutting down...")


app = FastAPI(
    title="Fake News Detection API",
    description="API phát hiện tin giả sử dụng nhiều model ML (Logistic Regression, SVM, BERT)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - thông tin cơ bản về API
    """
    return {
        "message": "Fake News Detection API",
        "version": "1.0.0",
        "status": "running",
        "description": "Crawl URL và phân tích tin giả với ML models",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "models_status": "/health/models",
            "predict_url": "GET /predict/{model_name}/{url}"
        },
        "available_models": ["logistic", "svm", "bert"],
        "example": "GET /predict/bert/https://vnexpress.net/..."
    }


if __name__ == "__main__":
    import uvicorn
    import asyncio
    import sys
    
    # Fix for Windows + Playwright
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload for Windows + Playwright compatibility
    )
