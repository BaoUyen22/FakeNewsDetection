import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from pathlib import Path

# ============================================================
# CONFIGURATION - Optimized for Google Colab GPU
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

# Paths (relative - works on Colab and local)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models" / "bert_output"

# Training Config
EPOCHS = 3
BATCH_SIZE = 32          # Tối ưu cho GPU T4
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
GRADIENT_ACCUMULATION = 2  # Effective batch = 32*2 = 64

# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, f1"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_transformer():
    """
    Train DistilBERT on full dataset
    Optimized for Google Colab GPU (T4)
    """
    
    print("="*60)
    print("BERT TRAINING - FULL DATASET")
    print("="*60)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*60 + "\n")
    
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    
    print("[1/6] Loading data...")
    
    if not train_path.exists():
        print(f"❌ Error: {train_path} not found")
        return
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Clean data
    train_df = train_df.dropna(subset=['text', 'label'])
    train_df['text'] = train_df['text'].astype(str)
    
    val_df = val_df.dropna(subset=['text', 'label'])
    val_df['text'] = val_df['text'].astype(str)
    
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val:   {len(val_df):,} samples")
    
    # ============================================================
    # 2. CREATE DATASETS
    # ============================================================
    
    print("\n[2/6] Creating HuggingFace datasets...")
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    # ============================================================
    # 3. TOKENIZER
    # ============================================================
    
    print(f"\n[3/6] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    print("[4/6] Tokenizing...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # ============================================================
    # 5. MODEL
    # ============================================================
    
    print(f"\n[5/6] Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    
    # ============================================================
    # 6. TRAINING ARGUMENTS
    # ============================================================
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        
        # Training
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        # Gradient accumulation
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        
        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        
        # Logging
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=100,
        logging_strategy="steps",
        report_to="none",  # Disable wandb
        
        # Performance (GPU optimization)
        fp16=torch.cuda.is_available(),  # Mixed precision
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        
        # Save
        save_total_limit=2,
    )
    
    # ============================================================
    # 7. TRAINER
    # ============================================================
    
    print("\n[6/6] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )
    
    # ============================================================
    # 8. TRAIN
    # ============================================================
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Model:      {MODEL_NAME}")
    print(f"Device:     {device}")
    print(f"Epochs:     {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Effective:  {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"LR:         {LEARNING_RATE}")
    print(f"Samples:    {len(train_df):,}")
    print("="*60 + "\n")
    
    trainer.train()
    
    # ============================================================
    # 9. EVALUATE
    # ============================================================
    
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    eval_results = trainer.evaluate()
    
    print(f"\nValidation Results:")
    print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall:    {eval_results['eval_recall']:.4f}")
    print(f"  F1-Score:  {eval_results['eval_f1']:.4f}")
    
    # ============================================================
    # 10. SAVE MODEL
    # ============================================================
    
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    print(f"✅ Model saved: {OUTPUT_DIR}")
    print(f"✅ Tokenizer saved: {OUTPUT_DIR}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_transformer()
