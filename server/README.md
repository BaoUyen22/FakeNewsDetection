# Fake News Detection API Server

API server sử dụng FastAPI để phục vụ các models phát hiện tin giả.

## 📁 Cấu trúc

```
server/
├── main.py                 # FastAPI application chính
├── routers/
│   ├── health.py          # Health check endpoints
│   └── predict.py         # Prediction endpoints
└── utils/
    ├── model_load.py      # Load và cache models
    ├── schemas.py         # Pydantic schemas
    ├── clean_text.py      # Text preprocessing
    └── crawl.py          # Web crawling utilities
```

## 🚀 Chạy Server

### 1. Cài đặt dependencies

```bash
pip install fastapi uvicorn python-multipart
```

### 2. Khởi động server

```bash
# Từ thư mục gốc của project
cd server
python main.py

# Hoặc dùng uvicorn trực tiếp
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Truy cập API

- **API Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Root**: http://localhost:8000/

## 📡 API Endpoints

### Health Check

```bash
# Basic health check
GET /health

# Models status
GET /health/models

# Readiness check
GET /health/ready

# Liveness check
GET /health/live
```

### Prediction

```bash
# Predict với model cụ thể
POST /predict/{model_name}
Body: {
  "text": "Your news article text here..."
}

# So sánh tất cả models
POST /predict/compare
Body: {
  "text": "Your news article text here..."
}
```

### Available Models

- `logistic` - Logistic Regression
- `svm` - Linear SVC
- `bert` - BERT/DistilBERT

## 📝 Ví dụ sử dụng

### Python

```python
import requests

url = "http://localhost:8000/predict/bert"
data = {
    "text": "Breaking news: Scientists discover new planet in our solar system..."
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict/logistic" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here..."}'
```

### Response Format

```json
{
  "model": "bert",
  "label": "FAKE",
  "confidence": 0.92
}
```

## 🔧 Model Loading

Models được load tự động khi server khởi động thông qua `lifespan` context manager:

1. **Baseline Models**: Logistic Regression, Linear SVC, TF-IDF Vectorizer
2. **BERT Model**: DistilBERT hoặc BERT fine-tuned model
3. **LSTM Model**: (Optional) Deep learning model

Models được cache trong memory để tăng tốc độ prediction.

## ⚙️ Configuration

### CORS

Mặc định cho phép tất cả origins (`*`). Trong production, nên restrict:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Device Selection

BERT model tự động chọn GPU nếu có, ngược lại dùng CPU.

## 🐛 Troubleshooting

### Models không load được

- Kiểm tra đường dẫn models trong `models/` directory
- Đảm bảo đã train models trước
- Xem logs khi server startup

### Import errors

- Đảm bảo đang chạy từ đúng directory
- Kiểm tra Python path settings

### GPU không được sử dụng

- Kiểm tra PyTorch installation: `torch.cuda.is_available()`
- Cài đặt CUDA-enabled PyTorch nếu cần

## 📊 Monitoring

Server cung cấp các endpoints để monitor:

- `/health` - Basic health
- `/health/models` - Models status
- `/health/ready` - Kubernetes readiness probe
- `/health/live` - Kubernetes liveness probe

## 🔐 Security Notes

- Trong production, nên thêm authentication/authorization
- Rate limiting để tránh abuse
- Input validation (đã có basic validation trong Pydantic)
- Secure CORS configuration
