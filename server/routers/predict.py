"""
Prediction endpoints - Crawl URL và dự đoán tin giả
"""
from fastapi import APIRouter, HTTPException, Path as PathParam, Query
import sys
from pathlib import Path
import torch
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.schemas import URLPredictionResponse
from utils.model_load import get_model, MODELS
from utils.clean_text import clean_text
from utils.crawl import crawl_and_clean, CrawlError

# Try import playwright crawler
try:
    from utils.crawl_playwright import crawl_and_clean_playwright
    PLAYWRIGHT_AVAILABLE = True
except:
    PLAYWRIGHT_AVAILABLE = False

router = APIRouter()


def predict_with_baseline(model_name: str, text: str):
    """Dự đoán với baseline models (Logistic Regression, SVM)"""
    model = get_model(model_name)
    vectorizer = get_model("vectorizer")
    
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model {model_name} or vectorizer not loaded"
        )
    
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    label = "FAKE" if prediction == 0 else "REAL"
    confidence = float(proba[prediction])
    
    return label, confidence


def predict_with_bert(text: str):
    """Dự đoán với BERT model"""
    model = get_model("bert")
    tokenizer = get_model("bert_tokenizer")
    device = MODELS.get("device", "cpu")
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="BERT model not loaded"
        )
    
    cleaned = clean_text(text)
    
    inputs = tokenizer(
        cleaned,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Remove token_type_ids if model doesn't need it (DistilBERT)
    if "token_type_ids" in inputs and not hasattr(model.config, "type_vocab_size"):
        inputs.pop("token_type_ids")
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        proba = torch.softmax(logits, dim=1)
        prediction = torch.argmax(proba, dim=1).item()
        confidence = proba[0][prediction].item()
    
    label = "FAKE" if prediction == 0 else "REAL"
    
    return label, float(confidence)


@router.get(
    "/{model_name}/{url:path}",
    response_model=URLPredictionResponse,
    summary="Crawl URL và dự đoán tin giả"
)
async def predict_from_url(
    model_name: str = PathParam(..., description="Model name: logistic, svm, bert"),
    url: str = PathParam(..., description="URL của bài báo"),
    use_js: bool = Query(False, description="Dùng Playwright để render JavaScript")
):
    """
    Crawl nội dung từ URL và dự đoán tin giả
    
    Example: 
    - GET /predict/bert/https://vnexpress.net/...
    - GET /predict/bert/https://cnn.com/...?use_js=true (for JS-heavy sites)
    """
    valid_models = ["logistic", "svm", "bert"]
    if model_name not in valid_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {valid_models}"
        )
    
    # Decode URL nếu cần
    decoded_url = unquote(url)
    
    try:
        # 1. Crawl content - choose crawler based on use_js flag
        if use_js:
            if not PLAYWRIGHT_AVAILABLE:
                raise HTTPException(
                    status_code=400,
                    detail="Playwright not available. Install with: playwright install chromium"
                )
            crawl_result = await crawl_and_clean_playwright(url=decoded_url, timeout=30)
        else:
            crawl_result = crawl_and_clean(url=decoded_url, timeout=15)
        
        text = crawl_result.get("clean_text", "")
        
        if not text or len(text) < 10:
            raise HTTPException(
                status_code=400,
                detail="Crawled text is too short or empty. Try adding ?use_js=true for JavaScript sites."
            )
        
        # 2. Predict
        if model_name in ["logistic", "svm"]:
            label, confidence = predict_with_baseline(model_name, text)
        else:  # bert
            label, confidence = predict_with_bert(text)
        
        return URLPredictionResponse(
            url=crawl_result["url"],
            title=crawl_result["title"],
            description=crawl_result["description"],
            text=text,
            text_length=crawl_result["clean_text_length"],
            model=model_name,
            label=label,
            confidence=confidence
        )
    
    except CrawlError as e:
        raise HTTPException(status_code=400, detail=f"Crawl error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
