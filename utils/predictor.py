"""
=============================================================================
PREDICTOR - Module xử lý prediction logic
=============================================================================

CHỨC NĂNG:
- Load models (baseline và transformer)
- Thực hiện prediction
- Trả về kết quả với confidence scores

NGƯỜI PHỤ TRÁCH: M4
=============================================================================
"""

import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelPredictor:
    """
    Class quản lý việc load và predict với các models
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Khởi tạo predictor
        
        Args:
            models_dir: Thư mục chứa models
        """
        self.models_dir = Path(models_dir)
        self.baseline_models = {}
        self.vectorizers = {}
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_baseline_models(self):
 
        baseline_dir = self.models_dir / "baseline"
        
        if not baseline_dir.exists():
            print(f"⚠️  Baseline models not found at {baseline_dir}")
            return
        
        # Load vectorizer
        vectorizer_path = baseline_dir / "tfidf_vectorizer.pkl"
        if vectorizer_path.exists():
            try:
                self.vectorizers["tfidf"] = joblib.load(vectorizer_path)
                print(f"✅ Loaded TF-IDF vectorizer")
            except Exception as e:
                print(f"❌ Error loading TF-IDF vectorizer: {e}")
                print(f"   Try retraining models with: python src/train_baseline.py")
                return
        
        # Load models
        model_files = {
            "logistic_regression": "logistic_regression.pkl",
            "linear_svc": "linear_svc.pkl",
            "naive_bayes": "naive_bayes.pkl",
        }
        
        for name, filename in model_files.items():
            model_path = baseline_dir / filename
            if model_path.exists():
                try:
                    self.baseline_models[name] = joblib.load(model_path)
                    print(f"✅ Loaded {name}")
                except Exception as e:
                    print(f"❌ Error loading {name}: {e}")
            else:
                print(f"⚠️  {name} not found at {model_path}")
    
    def load_transformer_model(self, model_name: str = "bert_output"):
        """
        Load transformer model (DistilBERT/PhoBERT)
        
        Args:
            model_name: Tên thư mục model (vd: "bert_output", "distilbert_model")
        """
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            print(f"⚠️  Transformer model not found at {model_dir}")
            return
        
        try:
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.transformer_model.to(self.device)
            self.transformer_model.eval()
            print(f"✅ Loaded transformer model from {model_dir}")
            print(f"   Device: {self.device}")
        except Exception as e:
            print(f"❌ Error loading transformer model: {e}")
    
    def predict_baseline(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Predict với baseline model
        
        Args:
            text: Text đã clean
            model_name: Tên model (logistic_regression, linear_svc, naive_bayes)
        
        Returns:
            {
                "verdict": "FAKE" hoặc "REAL",
                "confidence": 0.85,
                "label": 1 hoặc 0,
                "model": "logistic_regression"
            }
        """
        if model_name not in self.baseline_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        if "tfidf" not in self.vectorizers:
            raise ValueError("TF-IDF vectorizer not loaded")
        
        # Vectorize
        X = self.vectorizers["tfidf"].transform([text])
        
        # Predict
        model = self.baseline_models[model_name]
        prediction = model.predict(X)[0]
        
        # Get confidence (nếu có predict_proba)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(proba[prediction])
        else:
            # LinearSVC không có predict_proba, dùng decision_function
            decision = model.decision_function(X)[0]
            confidence = float(1 / (1 + np.exp(-decision)))  # Sigmoid
            if prediction == 0:
                confidence = 1 - confidence
        
        return {
            "verdict": "FAKE" if prediction == 1 else "REAL",
            "confidence": round(confidence, 4),
            "label": int(prediction),
            "model": model_name
        }
    
    def predict_transformer(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Predict với transformer model
        
        Args:
            text: Text đã clean
            max_length: Max sequence length
        
        Returns:
            {
                "verdict": "FAKE" hoặc "REAL",
                "confidence": 0.92,
                "label": 1 hoặc 0,
                "model": "transformer"
            }
        """
        if self.transformer_model is None or self.transformer_tokenizer is None:
            raise ValueError("Transformer model not loaded")
        
        # Tokenize (remove token_type_ids for DistilBERT)
        inputs = self.transformer_tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False  # DistilBERT doesn't use token_type_ids
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.transformer_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        return {
            "verdict": "FAKE" if prediction == 1 else "REAL",
            "confidence": round(confidence, 4),
            "label": int(prediction),
            "model": "transformer"
        }
    
    def predict_single(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Predict với 1 model cụ thể
        
        Args:
            text: Text đã clean
            model_name: "logistic_regression", "linear_svc", "naive_bayes", "transformer"
        """
        if model_name == "transformer":
            return self.predict_transformer(text)
        else:
            return self.predict_baseline(text, model_name)
    
    def predict_all(self, text: str) -> Dict[str, Any]:
        """
        Predict với tất cả models và so sánh
        
        Args:
            text: Text đã clean
        
        Returns:
            {
                "results": {
                    "logistic_regression": {...},
                    "linear_svc": {...},
                    "naive_bayes": {...},
                    "transformer": {...}
                },
                "consensus": {
                    "verdict": "FAKE",
                    "count": "3/4",
                    "confidence_avg": 0.87
                }
            }
        """
        results = {}
        
        # Predict với baseline models
        for model_name in self.baseline_models.keys():
            try:
                results[model_name] = self.predict_baseline(text, model_name)
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        # Predict với transformer
        if self.transformer_model is not None:
            try:
                results["transformer"] = self.predict_transformer(text)
            except Exception as e:
                results["transformer"] = {"error": str(e)}
        
        # Tính consensus
        verdicts = [r["verdict"] for r in results.values() if "verdict" in r]
        confidences = [r["confidence"] for r in results.values() if "confidence" in r]
        
        fake_count = verdicts.count("FAKE")
        real_count = verdicts.count("REAL")
        total = len(verdicts)
        
        consensus = {
            "verdict": "FAKE" if fake_count > real_count else "REAL",
            "count": f"{max(fake_count, real_count)}/{total}",
            "confidence_avg": round(np.mean(confidences), 4) if confidences else 0
        }
        
        return {
            "results": results,
            "consensus": consensus
        }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Lấy thông tin về model
        
        Args:
            model_name: Tên model
        
        Returns:
            {
                "name": "logistic_regression",
                "type": "baseline",
                "loaded": True,
                "recommended": False
            }
        """
        info = {
            "name": model_name,
            "type": "transformer" if model_name == "transformer" else "baseline",
            "loaded": False,
            "recommended": model_name == "transformer"
        }
        
        if model_name == "transformer":
            info["loaded"] = self.transformer_model is not None
        else:
            info["loaded"] = model_name in self.baseline_models
        
        return info


"""
USAGE:

1. Khởi tạo:
   predictor = ModelPredictor()
   predictor.load_baseline_models()
   predictor.load_transformer_model()

2. Predict với 1 model:
   result = predictor.predict_single(cleaned_text, "transformer")
   print(result["verdict"])  # "FAKE"
   print(result["confidence"])  # 0.83

3. Predict với tất cả models:
   results = predictor.predict_all(cleaned_text)
   print(results["consensus"]["verdict"])  # "FAKE"
   print(results["consensus"]["count"])  # 3/4

4. Lấy model info:
   info = predictor.get_model_info("transformer")
   print(info["recommended"])  # True

EXAMPLE:
   from utils.predictor import ModelPredictor
   from utils.preprocessor import TextPreprocessor
   
   # Setup
   predictor = ModelPredictor()
   predictor.load_baseline_models()
   predictor.load_transformer_model("bert_output")
   
   preprocessor = TextPreprocessor()
   
   # Predict
   text = "Breaking news: This is fake!"
   cleaned = preprocessor.clean_text(text)
   result = predictor.predict_single(cleaned, "transformer")
   
   print(f"Verdict: {result['verdict']}")
   print(f"Confidence: {result['confidence']}")
"""
