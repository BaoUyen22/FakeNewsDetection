"""
BERT Model Predictor
Transformer-based model (DistilBERT)
"""

import time
import torch
from pathlib import Path


class BERTPredictor:
    """Transformer: BERT (DistilBERT)"""
    
    def __init__(self):
        """Initialize BERT predictor"""
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load BERT model from disk"""
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
        """
        Predict with BERT model
        
        Args:
            text: Raw text to predict (will be preprocessed)
        
        Returns:
            dict: Model name -> prediction results
        """
        if not self.model or not self.tokenizer:
            return {}
        
        try:
            # Apply BERT-specific preprocessing (minimal)
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            
            from utils.preprocessor_for_model import BERTTextPreprocessor
            
            text_bert = BERTTextPreprocessor.transform(text)
            
            # Tokenize
            inputs = self.tokenizer(
                text_bert,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Remove token_type_ids if model doesn't need it (DistilBERT)
            if "token_type_ids" in inputs and not hasattr(self.model.config, "type_vocab_size"):
                inputs.pop("token_type_ids")
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
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
