"""
Baseline Model Predictor
Logistic Regression and LinearSVC models
"""

import time
import joblib
import numpy as np
from pathlib import Path


class BaselinePredictor:
    """Baseline models: Logistic Regression, LinearSVC"""
    
    def __init__(self, fake_threshold=0.70):
        """
        Initialize baseline predictor
        
        Args:
            fake_threshold: Threshold for classifying as FAKE (default 0.70)
                           Higher = more conservative, fewer false positives
        """
        self.models = {}
        self.vectorizer = None
        self.fake_threshold = fake_threshold
        self.load_models()
    
    def load_models(self):
        """Load baseline models from disk"""
        base_dir = Path(__file__).parent.parent
        models_dir = base_dir / "models" / "baseline"
        
        try:
            # Load TF-IDF vectorizer
            self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
            
            # Load models
            self.models["Logistic Regression"] = joblib.load(
                models_dir / "logistic_regression.pkl"
            )
            self.models["LinearSVC"] = joblib.load(
                models_dir / "linear_svc.pkl"
            )
            
            print("✅ Baseline models loaded")
        except Exception as e:
            print(f"❌ Error loading baseline models: {e}")
    
    def predict(self, text):
        """
        Predict with all baseline models
        
        Args:
            text: Cleaned text to predict
        
        Returns:
            dict: Model name -> prediction results
        """
        results = {}
        
        if not self.vectorizer or not self.models:
            return results
        
        # Vectorize text
        text_vec = self.vectorizer.transform([text])
        
        # Predict with each model
        for name, model in self.models.items():
            start_time = time.time()
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(text_vec)[0]
                real_prob = prob[0]  # Class 0 = REAL
                fake_prob = prob[1]  # Class 1 = FAKE
                
                # Use custom threshold instead of default 0.5
                pred = 1 if fake_prob >= self.fake_threshold else 0
                confidence = prob[pred]
                
            elif hasattr(model, 'decision_function'):
                # For SVC - decision function returns signed distance
                decision = model.decision_function(text_vec)[0]
                
                # Convert to probability using Platt scaling (sigmoid)
                # Positive decision → FAKE (class 1)
                # Negative decision → REAL (class 0)
                fake_prob = 1 / (1 + np.exp(-decision))
                real_prob = 1 - fake_prob
                
                # Use custom threshold
                pred = 1 if fake_prob >= self.fake_threshold else 0
                
                # Confidence based on actual prediction
                confidence = fake_prob if pred == 1 else real_prob
            else:
                # Fallback if no probability available
                pred = model.predict(text_vec)[0]
                fake_prob = 1.0 if pred == 1 else 0.0
                real_prob = 1.0 - fake_prob
                confidence = 1.0
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            results[name] = {
                "label": "FAKE" if pred == 1 else "REAL",
                "confidence": confidence,
                "fake_prob": fake_prob,
                "real_prob": real_prob,
                "inference_time": inference_time
            }
        
        return results
