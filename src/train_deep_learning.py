"""
=============================================================================
DEEP LEARNING MODEL TRAINING (LSTM + WORD EMBEDDING)
=============================================================================
Train LSTM model with static word embeddings
- Word Embedding: Each word → fixed vector (trainable)
- LSTM: Understand context from sequence
- Classification: Binary (Real/Fake)
=============================================================================
"""

import os
import torch
import pandas as pd
import time
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

     
# CONFIG
     
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models/deep_learning"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pth")

# Hyperparameters
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
MAX_LEN = 200
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

     
# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load train, val, test datasets"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    train_path = os.path.join(base_dir, TRAIN_PATH)
    val_path = os.path.join(base_dir, VAL_PATH)
    test_path = os.path.join(base_dir, TEST_PATH)
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Handle NaN
    for df in [train_df, val_df, test_df]:
        df["text"] = df["text"].fillna("").astype(str)
        df.drop(df[df["text"].str.strip().str.len() == 0].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    return train_df, val_df, test_df


# =============================================================================
# TOKENIZATION
# =============================================================================

def build_vocabulary(texts, vocab_size=10000):
    """
    Build vocabulary from texts (FIT ONLY ON TRAIN)
    
    Returns:
        word2idx: dict mapping word -> index
        idx2word: dict mapping index -> word
    """
    from collections import Counter
    
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Get top vocab_size words
    most_common = word_counts.most_common(vocab_size - 2)  # Reserve 2 slots
    
    # Create mappings
    word2idx = {"<PAD>": 0, "<UNK>": 1}  # Special tokens
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    
    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx
        idx2word[idx] = word
    
    return word2idx, idx2word


def texts_to_sequences(texts, word2idx, max_len=200):
    """
    Convert texts to sequences of token IDs
    
    Args:
        texts: List of text strings
        word2idx: Word to index mapping
        max_len: Maximum sequence length (truncate/pad)
    
    Returns:
        sequences: numpy array of shape (num_texts, max_len)
    """
    sequences = []
    
    for text in texts:
        words = text.lower().split()
        # Convert to indices (use <UNK> for unknown words)
        seq = [word2idx.get(word, word2idx["<UNK>"]) for word in words]
        
        # Truncate or pad
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [word2idx["<PAD>"]] * (max_len - len(seq))
        
        sequences.append(seq)
    
    return np.array(sequences)


# =============================================================================
# DEEP LEARNING MODEL (LSTM + EMBEDDING)
# =============================================================================

class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier with word embeddings
    
    Architecture:
        1. Embedding Layer: words → vectors
        2. LSTM Layer: process sequence → hidden state
        3. Dense Layer: classification
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1):
        super(LSTMClassifier, self).__init__()
        
        # Word Embedding (trainable static embeddings)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True  # Bidirectional LSTM
        )
        
        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, seq_len) - token IDs
        
        Returns:
            out: (batch_size, 1) - logits
        """
        # Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch, seq_len, embed_dim) → (batch, seq_len, hidden_dim*2)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state from both directions
        # hidden: (num_layers*2, batch, hidden_dim)
        # Concatenate last layer's forward and backward hidden states
        hidden_fwd = hidden[-2, :, :]  # Forward direction
        hidden_bwd = hidden[-1, :, :]  # Backward direction
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Classification: (batch, hidden_dim*2) → (batch, 1)
        out = self.fc(hidden_concat)
        
        return out.squeeze()


# =============================================================================
# TRAINING
# =============================================================================

def train():
    """Train LSTM model with word embeddings"""
    print("="*60)
    print("DEEP LEARNING (LSTM + WORD EMBEDDING) TRAINING")
    print("="*60)
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    train_df, val_df, test_df = load_data()
    
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    
    # 2. Build vocabulary (FIT ONLY ON TRAIN)
    print(f"\n[2/7] Building vocabulary (vocab_size={VOCAB_SIZE})...")
    word2idx, idx2word = build_vocabulary(train_df["text"], vocab_size=VOCAB_SIZE)
    
    print(f"   Vocabulary size: {len(word2idx)}")
    print(f"   Sample words: {list(word2idx.keys())[:10]}")
    
    # 3. Convert texts to sequences
    print(f"\n[3/7] Converting texts to sequences (max_len={MAX_LEN})...")
    X_train = texts_to_sequences(train_df["text"], word2idx, max_len=MAX_LEN)
    X_val = texts_to_sequences(val_df["text"], word2idx, max_len=MAX_LEN)
    X_test = texts_to_sequences(test_df["text"], word2idx, max_len=MAX_LEN)
    
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values
    
    print(f"   Train sequences: {X_train.shape}")
    print(f"   Val sequences:   {X_val.shape}")
    print(f"   Test sequences:  {X_test.shape}")
    
    # 4. Calculate class weights
    fake_count = sum(y_train)
    real_count = len(y_train) - fake_count
    pos_weight = torch.tensor([real_count / fake_count])
    
    print(f"\n   Class distribution:")
    print(f"   Real: {real_count} | Fake: {fake_count}")
    print(f"   Pos weight: {pos_weight.item():.4f}")
    
    # 5. Convert to PyTorch tensors
    X_train_tensor = torch.LongTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    X_val_tensor = torch.LongTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    X_test_tensor = torch.LongTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 6. Create DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    # 7. Define model
    print(f"\n[4/7] Building LSTM model...")
    print(f"   Embedding dim: {EMBEDDING_DIM}")
    print(f"   Hidden dim: {HIDDEN_DIM}")
    print(f"   LSTM layers: 2 (bidirectional)")
    
    model = LSTMClassifier(
        vocab_size=len(word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=1
    )
    
    # Check if CUDA available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   Device: {device}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # 8. Training loop
    print(f"\n[5/7] Training for {EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation accuracy
        model.eval()
        with torch.no_grad():
            X_val_device = X_val_tensor.to(device)
            val_outputs = torch.sigmoid(model(X_val_device))
            val_preds = (val_outputs >= 0.5).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)
        
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    train_time = time.time() - start_time
    print(f"\n   Training time: {train_time:.2f}s")
    
    # 9. Evaluate on test set
    print("\n[6/7] Evaluating on test set...")
    model.eval()
    
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        
        start_time = time.time()
        test_outputs = torch.sigmoid(model(X_test_device))
        inference_time = (time.time() - start_time) / len(y_test) * 1000
        
        test_preds = (test_outputs >= 0.5).cpu().numpy()
    
    # Metrics
    accuracy = accuracy_score(y_test, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='binary')
    
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Inference: {inference_time:.3f}ms/sample")
    
    print("\n   Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["Real (0)", "Fake (1)"], digits=4))
    
    print("\n   Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))
    
    # 10. Save model
    print("\n[7/7] Saving model...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(base_dir, MODEL_DIR)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(base_dir, MODEL_PATH)
    torch.save({
        "model_state": model.state_dict(),
        "word2idx": word2idx,
        "idx2word": idx2word,
        "config": {
            "vocab_size": len(word2idx),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "max_len": MAX_LEN
        }
    }, save_path)
    
    print(f"   ✅ Model saved: {save_path}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    print(f"Model: LSTM + Word Embedding")
    print(f"Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")
    
    return model, word2idx


# =============================================================================
# PREDICTION
# =============================================================================

def predict(text, model, word2idx, device="cpu"):
    """
    Predict with LSTM model
    
    Args:
        text: Text ĐÃ CLEANED (use utils/preprocessor.py to clean first)
    """
    # Convert to sequence
    seq = texts_to_sequences([text], word2idx, max_len=MAX_LEN)
    seq_tensor = torch.LongTensor(seq).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model(seq_tensor)).item()
    
    # Use standard threshold 0.5
    label = "FAKE" if output >= 0.5 else "REAL"
    confidence = output if output >= 0.5 else (1 - output)
    
    print(f"\n{label} NEWS | Confidence: {confidence*100:.2f}%")
    print(f"   FAKE: {output*100:.2f}%")
    print(f"   REAL: {(1-output)*100:.2f}%")


# =============================================================================
# MAIN
# =============================================================================
     
if __name__ == "__main__":
    train()
