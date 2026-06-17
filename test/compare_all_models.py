"""
=============================================================================
COMPARE ALL MODELS - INTERACTIVE TESTING
=============================================================================
Test a single news article with all available models and compare results
- Baseline: Logistic Regression, LinearSVC
- Deep Learning: LSTM + Word Embedding
- Transformer: BERT (DistilBERT)

Compare based on:
- Prediction result (REAL/FAKE)
- Confidence score
- Inference time
=============================================================================
"""

import os
import sys
import time
import torch
import joblib
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessor import clean_text


# =============================================================================
# MODEL LOADERS
# =============================================================================

class BaselinePredictor:
    """Baseline models: Logistic Regression, LinearSVC"""
    
    def __init__(self, fake_threshold=0.70):
        """
        Args:
            fake_threshold: Threshold for classifying as FAKE (default 0.70)
                           Higher = more conservative, fewer false positives
        """
        self.models = {}
        self.vectorizer = None
        self.fake_threshold = fake_threshold
        self.load_models()
    
    def load_models(self):
        """Load baseline models"""
        base_dir = Path(__file__).parent.parent
        models_dir = base_dir / "models" / "baseline"
        
        try:
            # Load vectorizer
            self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
            
            # Load models
            self.models["Logistic Regression"] = joblib.load(models_dir / "logistic_regression.pkl")
            self.models["LinearSVC"] = joblib.load(models_dir / "linear_svc.pkl")
            
            print("✅ Baseline models loaded")
        except Exception as e:
            print(f"❌ Error loading baseline models: {e}")
    
    def predict(self, text):
        """Predict with all baseline models"""
        results = {}
        
        if not self.vectorizer or not self.models:
            return results
        
        # Vectorize
        text_vec = self.vectorizer.transform([text])
        
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


class DeepLearningPredictor:
    """Deep Learning: LSTM + Word Embedding"""
    
    def __init__(self):
        self.model = None
        self.word2idx = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load LSTM model"""
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / "models" / "deep_learning" / "lstm_model.pth"
        
        try:
            # Import model class
            sys.path.append(str(base_dir / "src"))
            from train_deep_learning import LSTMClassifier
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.word2idx = checkpoint["word2idx"]
            self.config = checkpoint["config"]
            
            # Initialize model
            self.model = LSTMClassifier(
                vocab_size=self.config["vocab_size"],
                embedding_dim=self.config["embedding_dim"],
                hidden_dim=self.config["hidden_dim"]
            )
            self.model.load_state_dict(checkpoint["model_state"])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("✅ Deep Learning model loaded")
        except Exception as e:
            print(f"❌ Error loading Deep Learning model: {e}")
    
    def texts_to_sequences(self, text, max_len=200):
        """Convert text to sequence"""
        words = text.lower().split()
        seq = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        
        # Truncate or pad
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [self.word2idx["<PAD>"]] * (max_len - len(seq))
        
        return seq
    
    def predict(self, text):
        """Predict with LSTM model"""
        if not self.model or not self.word2idx:
            return {}
        
        try:
            # Convert to sequence
            seq = self.texts_to_sequences(text, max_len=self.config["max_len"])
            seq_tensor = torch.LongTensor([seq]).to(self.device)
            
            # Predict
            start_time = time.time()
            with torch.no_grad():
                output = torch.sigmoid(self.model(seq_tensor)).item()
            inference_time = (time.time() - start_time) * 1000  # ms
            
            pred = 1 if output >= 0.5 else 0
            confidence = output if pred == 1 else (1 - output)
            
            return {
                "LSTM": {
                    "label": "FAKE" if pred == 1 else "REAL",
                    "confidence": confidence,
                    "fake_prob": output,
                    "real_prob": 1 - output,
                    "inference_time": inference_time
                }
            }
        except Exception as e:
            print(f"❌ Error predicting with Deep Learning: {e}")
            return {}


class BERTPredictor:
    """Transformer: BERT (DistilBERT)"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load BERT model"""
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / "models" / "bert_output"
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("✅ BERT model loaded")
        except Exception as e:
            print(f"❌ Error loading BERT model: {e}")
    
    def predict(self, text):
        """Predict with BERT model"""
        if not self.model or not self.tokenizer:
            return {}
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]
            inference_time = (time.time() - start_time) * 1000  # ms
            
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()
            
            return {
                "BERT (DistilBERT)": {
                    "label": "FAKE" if pred == 1 else "REAL",
                    "confidence": confidence,
                    "fake_prob": probs[1].item(),
                    "real_prob": probs[0].item(),
                    "inference_time": inference_time
                }
            }
        except Exception as e:
            print(f"❌ Error predicting with BERT: {e}")
            return {}


# =============================================================================
# COMPARISON DISPLAY
# =============================================================================

def display_comparison(text, all_results):
    """Display comparison table for all models"""
    print("\n" + "="*80)
    print("PREDICTION RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n📰 Input Text (cleaned):")
    print(f"   {text[:200]}..." if len(text) > 200 else f"   {text}")
    
    print("\n" + "="*80)
    print("📊 PREDICTIONS:")
    print("="*80)
    print(f"{'Model':<25} {'Prediction':<12} {'Confidence':<12} {'FAKE %':<12} {'REAL %':<12} {'Time (ms)':<12}")
    print("-"*80)
    
    for model_name, result in all_results.items():
        print(f"{model_name:<25} "
              f"{result['label']:<12} "
              f"{result['confidence']*100:>10.2f}% "
              f"{result['fake_prob']*100:>10.2f}% "
              f"{result['real_prob']*100:>10.2f}% "
              f"{result['inference_time']:>10.3f}")
    
    print("\n" + "="*80)
    print("📈 ANALYSIS:")
    print("="*80)
    
    # Consensus
    predictions = [r["label"] for r in all_results.values()]
    fake_count = predictions.count("FAKE")
    real_count = predictions.count("REAL")
    
    print(f"   Consensus: {fake_count} FAKE, {real_count} REAL")
    
    if fake_count == len(predictions):
        print("   ✅ All models agree: FAKE NEWS")
    elif real_count == len(predictions):
        print("   ✅ All models agree: REAL NEWS")
    else:
        print("   ⚠️  Models disagree!")
    
    # Average confidence
    avg_confidence = np.mean([r["confidence"] for r in all_results.values()])
    print(f"   Average confidence: {avg_confidence*100:.2f}%")
    
    # Fastest model
    fastest = min(all_results.items(), key=lambda x: x[1]["inference_time"])
    print(f"   Fastest model: {fastest[0]} ({fastest[1]['inference_time']:.3f}ms)")
    
    # Most confident
    most_confident = max(all_results.items(), key=lambda x: x[1]["confidence"])
    print(f"   Most confident: {most_confident[0]} ({most_confident[1]['confidence']*100:.2f}%)")


# =============================================================================
# MAIN INTERACTIVE TESTING
# =============================================================================

def main():
    """Main interactive testing loop"""
    print("="*80)
    print("FAKE NEWS DETECTION - MODEL COMPARISON")
    print("="*80)
    print("\nLoading models...")
    
    # Load all models
    baseline = BaselinePredictor()
    deep_learning = DeepLearningPredictor()
    bert = BERTPredictor()
    
    print("\n" + "="*80)
    print("READY FOR TESTING")
    print("="*80)
    print("\n📝 Instructions:")
    print("   1. Enter a news article (title + text)")
    print("   2. Text will be automatically cleaned")
    print("   3. All models will predict and results will be compared")
    print("   4. Type 'exit' to quit")
    print("\n" + "="*80)
    
    while True:
        print("\n" + "-"*80)
        user_input = input("\n📰 Enter news article (or 'exit' to quit):\n> ")
        
        if user_input.lower().strip() == "exit":
            print("\n👋 Goodbye!")
            break
        
        if not user_input.strip():
            print("⚠️  Please enter some text!")
            continue
        
        # Clean text
        cleaned_text = clean_text(user_input)
        
        if not cleaned_text.strip():
            print("⚠️  Text is empty after cleaning!")
            continue
        
        # Collect predictions from all models
        all_results = {}
        
        # Baseline models
        baseline_results = baseline.predict(cleaned_text)
        all_results.update(baseline_results)
        
        # Deep Learning
        dl_results = deep_learning.predict(cleaned_text)
        all_results.update(dl_results)
        
        # BERT
        bert_results = bert.predict(cleaned_text)
        all_results.update(bert_results)
        
        # Display comparison
        if all_results:
            display_comparison(cleaned_text, all_results)
        else:
            print("❌ No models available for prediction!")


if __name__ == "__main__":
    main()
