"""
Prediction endpoints - Crawl URL và dự đoán tin giả
"""
from fastapi import APIRouter, HTTPException, Path as PathParam, Query, Request, Form
import importlib.util
import sys
from pathlib import Path
import torch
import numpy as np
from urllib.parse import unquote
from scipy.sparse import hstack

# Ensure server/utils is available for local router helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load root project utils/preprocessor_for_model.py by path to avoid conflicting server/utils package
ROOT_PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT_UTILS_FILE = ROOT_PROJECT_DIR / "utils" / "preprocessor_for_model.py"
if ROOT_UTILS_FILE.exists():
    _spec = importlib.util.spec_from_file_location("root_preprocessor_for_model", ROOT_UTILS_FILE)
    _root_preprocessor = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_root_preprocessor)
    MLTextPreprocessor = _root_preprocessor.MLTextPreprocessor
else:
    MLTextPreprocessor = None

from utils.schemas import (
    URLPredictionResponse,
    PredictionDetails,
    TextPredictionRequest,
    TextPredictionResponse,
)
from utils.model_load import get_model, MODELS
from utils.crawl import crawl_and_clean, CrawlError

# Try import playwright crawler
try:
    from utils.crawl_playwright import crawl_and_clean_playwright
    PLAYWRIGHT_AVAILABLE = True
except:
    PLAYWRIGHT_AVAILABLE = False

router = APIRouter()


def predict_with_baseline(model_name: str, text: str):
    """
    Dự đoán với baseline models (Logistic Regression, SVM)
    
    Args:
        model_name: Model name (logistic, svm)
        text: Raw text to predict (will be preprocessed)
    
    Returns: (label, confidence, fake_prob, real_prob, cleaned_text)
    """
    model = get_model(model_name)
    vectorizer = get_model("vectorizer")
    
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model {model_name} or vectorizer not loaded"
        )
    
    if MLTextPreprocessor is None:
        raise HTTPException(status_code=500, detail="MLTextPreprocessor module not available")

    # Apply ML-specific preprocessing: lowercase, count !?, remove punctuation, stopwords, lemmatization
    cleaned_text, num_excl, num_quest = MLTextPreprocessor.transform(text)
    
    # TF-IDF vectorization
    X_tfidf = vectorizer.transform([cleaned_text])
    
    # Add num_exclamation and num_question features
    excl_array = np.array([[num_excl]])
    quest_array = np.array([[num_quest]])
    X = hstack([X_tfidf, excl_array, quest_array])
    
    prediction = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        # proba[0] = REAL (class 0), proba[1] = FAKE (class 1)
        real_prob = float(proba[0])
        fake_prob = float(proba[1])
    elif hasattr(model, "decision_function"):
        decision = float(model.decision_function(X)[0])
        fake_prob = 1 / (1 + np.exp(-decision))
        real_prob = 1 - fake_prob
    else:
        fake_prob = 0.5
        real_prob = 0.5
    
    label = "FAKE" if prediction == 1 else "REAL"
    confidence = fake_prob if label == "FAKE" else real_prob
    
    return label, confidence, fake_prob, real_prob, cleaned_text


def predict_with_bert(model_name: str, text: str):
    """
    Dự đoán với BERT model
    
    Args:
        model_name: Model name (bert)
        text: Raw text to predict (will be preprocessed)
    
    Returns: (label, confidence, fake_prob, real_prob, cleaned_text)
    """
    model = get_model("bert")
    tokenizer = get_model("bert_tokenizer")
    device = MODELS.get("device", "cpu")
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="BERT model not loaded"
        )
    
    # Load BERTTextPreprocessor
    ROOT_BERT_PREPROCESSOR_FILE = ROOT_PROJECT_DIR / "utils" / "preprocessor_for_model.py"
    if ROOT_BERT_PREPROCESSOR_FILE.exists():
        _spec = importlib.util.spec_from_file_location("bert_preprocessor", ROOT_BERT_PREPROCESSOR_FILE)
        _bert_preprocessor = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_bert_preprocessor)
        BERTTextPreprocessor = _bert_preprocessor.BERTTextPreprocessor
    else:
        raise HTTPException(status_code=500, detail="BERTTextPreprocessor not available")
    
    # Apply BERT-specific preprocessing: only normalize whitespace
    cleaned = BERTTextPreprocessor.transform(text)
    
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
        
        # proba[0][0] = Class 0 = REAL, proba[0][1] = Class 1 = FAKE
        real_prob = float(proba[0][0].item())
        fake_prob = float(proba[0][1].item())
        confidence = proba[0][prediction].item()
    
    label = "FAKE" if prediction == 1 else "REAL"
    
    return label, float(confidence), fake_prob, real_prob, cleaned


def predict_with_lstm(model_name: str, text: str):
    """
    Dự đoán với LSTM model
    
    Args:
        model_name: Model name (lstm)
        text: Raw text to predict (will be preprocessed)
    
    Returns: (label, confidence, fake_prob, real_prob, cleaned_text)
    """
    model = get_model("lstm")
    word2idx = MODELS.get("lstm_word2idx")
    max_len = MODELS.get("lstm_max_len", 200)
    device = MODELS.get("lstm_device", "cpu")
    
    if model is None or word2idx is None:
        raise HTTPException(
            status_code=503,
            detail="LSTM model not loaded"
        )
    
    # Load LSTMTextPreprocessor
    ROOT_LSTM_PREPROCESSOR_FILE = ROOT_PROJECT_DIR / "utils" / "preprocessor_for_model.py"
    if ROOT_LSTM_PREPROCESSOR_FILE.exists():
        _spec = importlib.util.spec_from_file_location("lstm_preprocessor", ROOT_LSTM_PREPROCESSOR_FILE)
        _lstm_preprocessor = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_lstm_preprocessor)
        LSTMTextPreprocessor = _lstm_preprocessor.LSTMTextPreprocessor
    else:
        raise HTTPException(status_code=500, detail="LSTMTextPreprocessor not available")
    
    # Apply LSTM-specific preprocessing: lowercase, keep !?, normalize whitespace
    cleaned = LSTMTextPreprocessor.transform(text)
    
    # Convert text to sequence
    words = cleaned.lower().split()
    seq = [word2idx.get(word, word2idx.get("<UNK>", 1)) for word in words]
    
    # Truncate or pad
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [word2idx.get("<PAD>", 0)] * (max_len - len(seq))
    
    # Convert to tensor
    seq_tensor = torch.LongTensor([seq]).to(device)
    
    with torch.no_grad():
        output = torch.sigmoid(model(seq_tensor)).item()
    
    # output = fake probability
    fake_prob = float(output)
    real_prob = 1 - fake_prob
    
    prediction = 1 if output >= 0.5 else 0
    label = "FAKE" if prediction == 1 else "REAL"
    confidence = fake_prob if label == "FAKE" else real_prob
    
    return label, float(confidence), fake_prob, real_prob, cleaned


@router.get(
    "/{model_name}/{url:path}",
    response_model=URLPredictionResponse,
    summary="Crawl URL và dự đoán tin giả"
)
async def predict_from_url(
    model_name: str = PathParam(..., description="Model name: logistic, svm, bert, lstm"),
    url: str = PathParam(..., description="URL của bài báo"),
    use_js: bool = Query(False, description="Dùng Playwright để render JavaScript")
):
    """
    Crawl nội dung từ URL và dự đoán tin giả
    Example: 
    - GET /predict/bert/https://vnexpress.net/...
    - GET /predict/bert/https://cnn.com/...?use_js=true (for JS-heavy sites)
    """
    valid_models = ["logistic", "svm", "bert", "lstm"]
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
            crawl_result = await crawl_and_clean_playwright(url=decoded_url, timeout=60)
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
            label, confidence, fake_prob, real_prob, cleaned_text = predict_with_baseline(model_name, text)
        elif model_name == "bert":
            label, confidence, fake_prob, real_prob, cleaned_text = predict_with_bert(model_name, text)
        elif model_name == "lstm":
            label, confidence, fake_prob, real_prob, cleaned_text = predict_with_lstm(model_name, text)
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not supported")
        
        # Return full processed text (no limit)
        
        return URLPredictionResponse(
            url=crawl_result["url"],
            title=crawl_result["title"],
            description=crawl_result["description"],
            text_length=crawl_result["clean_text_length"],
            processed_text=cleaned_text,  # Full text, no preview
            processed_text_length=len(cleaned_text),
            model=model_name,
            label=label,
            confidence=confidence,
            prediction_details=PredictionDetails(
                fake_probability=fake_prob,
                real_probability=real_prob
            )
        )
    
    except CrawlError as e:
        raise HTTPException(status_code=400, detail=f"Crawl error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post(
    "/text/{model_name}",
    response_model=TextPredictionResponse,
    summary="Predict news text directly"
)
async def predict_from_text(
    model_name: str = PathParam(..., description="Model name: logistic, svm, bert, lstm"),
    request: Request = None,
    text: str | None = Form(None),
):
    """Predict from raw news text input."""
    valid_models = ["logistic", "svm", "bert", "lstm"]
    if model_name not in valid_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {valid_models}"
        )

    body_text = None
    if text:
        body_text = text
    elif request is not None:
        try:
            body = await request.json()
            body_text = body.get("text") if isinstance(body, dict) else None
        except Exception:
            form = await request.form()
            body_text = form.get("text")

    if not body_text or len(str(body_text).strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Input text is required and must be at least 10 characters long."
        )

    text = str(body_text)

    if model_name in ["logistic", "svm"]:
        label, confidence, fake_prob, real_prob, cleaned_text = predict_with_baseline(model_name, text)
    elif model_name == "bert":
        label, confidence, fake_prob, real_prob, cleaned_text = predict_with_bert(model_name, text)
    elif model_name == "lstm":
        label, confidence, fake_prob, real_prob, cleaned_text = predict_with_lstm(model_name, text)
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not supported")

    return TextPredictionResponse(
        input_text=text,
        processed_text=cleaned_text,
        processed_text_length=len(cleaned_text),
        model=model_name,
        label=label,
        confidence=confidence,
        prediction_details=PredictionDetails(
            fake_probability=fake_prob,
            real_probability=real_prob
        )
    )
