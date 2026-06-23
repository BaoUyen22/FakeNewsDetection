"""
Deep Learning Model Predictor
LSTM with Word Embeddings
"""

import sys
import time
import torch
from pathlib import Path


class DeepLearningPredictor:
    """Deep Learning: LSTM + Word Embedding"""
    
    def __init__(self):
        """Initialize deep learning predictor"""
        self.model = None
        self.word2idx = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load LSTM model from disk"""
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
        """
        Convert text to sequence of word indices
        
        Args:
            text: Input text
            max_len: Maximum sequence length
        
        Returns:
            list: Sequence of word indices
        """
        words = text.lower().split()
        seq = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        
        # Truncate or pad
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [self.word2idx["<PAD>"]] * (max_len - len(seq))
        
        return seq
    
    def predict(self, text):
        """
        Predict with LSTM model
        
        Args:
            text: Raw text to predict (will be preprocessed)
        
        Returns:
            dict: Model name -> prediction results
        """
        if not self.model or not self.word2idx:
            return {}
        
        try:
            # Apply LSTM-specific preprocessing
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            
            from utils.preprocessor_for_model import LSTMTextPreprocessor
            
            text_lstm = LSTMTextPreprocessor.transform(text)
            
            # Convert to sequence
            seq = self.texts_to_sequences(text_lstm, max_len=self.config["max_len"])
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
