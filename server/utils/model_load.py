"""
Model loader - Load và cache các ML models khi server khởi động
"""
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# Global dictionary để lưu các models đã load
MODELS = {}

# Base path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


async def load_baseline_models():
    """Load Logistic Regression và LinearSVC models"""
    try:
        baseline_dir = MODELS_DIR / "baseline"
        
        # Load vectorizer
        vectorizer_path = baseline_dir / "tfidf_vectorizer.pkl"
        print(f"Loading vectorizer from {vectorizer_path}...")
        MODELS["vectorizer"] = joblib.load(vectorizer_path)
        
        # Load Logistic Regression
        logistic_path = baseline_dir / "logistic_regression.pkl"
        print(f"Loading logistic regression from {logistic_path}...")
        MODELS["logistic"] = joblib.load(logistic_path)
        
        # Load LinearSVC
        svm_path = baseline_dir / "linear_svc.pkl"
        print(f"Loading linear svc from {svm_path}...")
        MODELS["linear_svc"] = joblib.load(svm_path)
        
        print("✓ Baseline models loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading baseline models: {e}")
        import traceback
        traceback.print_exc()
        return False


async def load_bert_model():
    """Load BERT/DistilBERT model"""
    try:
        bert_dir = MODELS_DIR / "bert_output"
        
        print(f"Loading BERT from {bert_dir}...")
        
        # Load tokenizer
        MODELS["bert_tokenizer"] = AutoTokenizer.from_pretrained(str(bert_dir))
        
        # Load model
        MODELS["bert"] = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))
        MODELS["bert"].eval()
        
        # Check if GPU available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODELS["bert"] = MODELS["bert"].to(device)
        MODELS["device"] = device
        
        print(f"✓ BERT model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"✗ Error loading BERT model: {e}")
        import traceback
        traceback.print_exc()
        return False


async def load_lstm_model():
    """Load LSTM model"""
    try:
        lstm_path = MODELS_DIR / "deep_learning" / "lstm_model.pth"
        
        if not lstm_path.exists():
            print("⚠ LSTM model file not found")
            return True
        
        print(f"Loading LSTM from {lstm_path}...")
        
        # Import model architecture from models package
        import sys
        project_root = str(BASE_DIR)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import LSTMClassifier
        from models import LSTMClassifier
        
        # Load checkpoint
        checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=False)
        
        # Extract config
        config = checkpoint.get("config", {})
        vocab_size = config.get("vocab_size", 10000)
        embedding_dim = config.get("embedding_dim", 128)
        hidden_dim = config.get("hidden_dim", 64)
        max_len = config.get("max_len", 200)
        
        # Create model
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=1
        )
        
        # Load weights
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        # Store model and metadata
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        MODELS["lstm"] = model
        MODELS["lstm_word2idx"] = checkpoint.get("word2idx", {})
        MODELS["lstm_max_len"] = max_len
        MODELS["lstm_device"] = device
        
        print(f"✓ LSTM model loaded successfully on {device}")
        print(f"  Vocab size: {vocab_size}, Max len: {max_len}")
        return True
        
    except Exception as e:
        print(f"⚠ LSTM model load failed: {e}")
        # Don't crash server if LSTM fails
        return True


async def load_all_models():
    """Load tất cả models"""
    print("\n" + "="*50)
    print("Loading Models...")
    print("="*50)
    
    await load_baseline_models()
    await load_bert_model()
    await load_lstm_model()
    
    print("="*50)
    model_names = ['logistic', 'linear_svc', 'bert', 'lstm']
    loaded_models = [m for m in model_names if m in MODELS]
    print(f"Total models loaded: {len(loaded_models)} - {', '.join(loaded_models)}")
    print("="*50 + "\n")


def get_model(model_name: str):
    """Lấy model từ cache"""
    return MODELS.get(model_name)


def get_all_models_status():
    """Kiểm tra trạng thái tất cả models"""
    model_names = ['logistic', 'linear_svc', 'bert', 'lstm']
    loaded_count = len([m for m in model_names if m in MODELS])
    return {
        "logistic_regression": "loaded" if "logistic" in MODELS else "not_loaded",
        "linear_svc": "loaded" if "linear_svc" in MODELS else "not_loaded",
        "bert": "loaded" if "bert" in MODELS else "not_loaded",
        "lstm": "loaded" if "lstm" in MODELS else "not_loaded",
        "vectorizer": "loaded" if "vectorizer" in MODELS else "not_loaded",
        "total_models": loaded_count
    }
