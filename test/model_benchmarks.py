"""
=============================================================================
MODEL BENCHMARKS - COMPREHENSIVE COMPARISON
=============================================================================
Evaluate all models on test set and compare using multiple criteria:
- Performance: Accuracy, Precision, Recall, F1-Score
- Efficiency: Training Time, Inference Time, Throughput
- Resources: CPU/GPU, Memory, Storage
- Operations: Ease of Deploy, Interpretability
=============================================================================
"""

import os
import sys
import time
import torch
import joblib
import pandas as pd
import numpy as np
import psutil
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


# =============================================================================
# LOAD TEST DATA
# =============================================================================

def load_test_data():
    """Load test dataset"""
    base_dir = Path(__file__).parent.parent
    test_path = base_dir / "data" / "processed" / "test.csv"
    
    test_df = pd.read_csv(test_path)
    test_df["text"] = test_df["text"].fillna("").astype(str)
    test_df = test_df[test_df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    
    return test_df


# =============================================================================
# BASELINE EVALUATION
# =============================================================================

def evaluate_baseline(test_df):
    """Evaluate baseline models"""
    print("\n" + "="*60)
    print("EVALUATING BASELINE MODELS")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models" / "baseline"
    
    results = {}
    
    try:
        # Load vectorizer
        vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
        X_test = vectorizer.transform(test_df["text"])
        y_test = test_df["label"].values
        
        # Models (use exact filenames)
        models = {
            "Logistic Regression": joblib.load(models_dir / "logistic_regression.pkl"),
            "LinearSVC": joblib.load(models_dir / "linear_svc.pkl")
        }
        
        for name, model in models.items():
            print(f"\n{name}:")
            
            # Inference time
            start_time = time.time()
            y_pred = model.predict(X_test)
            total_time = time.time() - start_time
            inference_time = total_time / len(y_test) * 1000  # ms per sample
            throughput = len(y_test) / total_time  # samples per second
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            
            # Model size (map to actual filenames)
            filename_map = {
                "Logistic Regression": "logistic_regression.pkl",
                "LinearSVC": "linear_svc.pkl"
            }
            model_path = models_dir / filename_map[name]
            model_size = model_path.stat().st_size / (1024 * 1024)  # MB
            
            results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "inference_time": inference_time,
                "throughput": throughput,
                "model_size": model_size,
                "device": "CPU",
                "interpretability": "HIGH"
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   Inference: {inference_time:.3f}ms/sample")
            print(f"   Throughput: {throughput:.0f} samples/sec")
        
        # Add TF-IDF size
        tfidf_size = (models_dir / "tfidf_vectorizer.pkl").stat().st_size / (1024 * 1024)
        print(f"\n   TF-IDF Vectorizer: {tfidf_size:.2f}MB")
        
    except Exception as e:
        print(f"❌ Error evaluating baseline: {e}")
    
    return results


# =============================================================================
# DEEP LEARNING EVALUATION
# =============================================================================

def evaluate_deep_learning(test_df):
    """Evaluate LSTM model"""
    print("\n" + "="*60)
    print("EVALUATING DEEP LEARNING MODEL")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "models" / "deep_learning" / "lstm_model.pth"
    
    results = {}
    
    try:
        # Import model class
        sys.path.append(str(base_dir / "src"))
        from train_deep_learning import LSTMClassifier, texts_to_sequences
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        model = LSTMClassifier(
            vocab_size=checkpoint["config"]["vocab_size"],
            embedding_dim=checkpoint["config"]["embedding_dim"],
            hidden_dim=checkpoint["config"]["hidden_dim"]
        )
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        model.eval()
        
        # Prepare data
        word2idx = checkpoint["word2idx"]
        max_len = checkpoint["config"]["max_len"]
        X_test = texts_to_sequences(test_df["text"], word2idx, max_len=max_len)
        X_test_tensor = torch.LongTensor(X_test).to(device)
        y_test = test_df["label"].values
        
        print(f"\nLSTM + Word Embedding:")
        
        # Inference time
        start_time = time.time()
        with torch.no_grad():
            outputs = torch.sigmoid(model(X_test_tensor))
            y_pred = (outputs >= 0.5).cpu().numpy()
        total_time = time.time() - start_time
        inference_time = total_time / len(y_test) * 1000  # ms per sample
        throughput = len(y_test) / total_time  # samples per second
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        
        # Model size
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        
        results["LSTM"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "inference_time": inference_time,
            "throughput": throughput,
            "model_size": model_size,
            "device": str(device).upper(),
            "interpretability": "LOW"
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Inference: {inference_time:.3f}ms/sample")
        print(f"   Throughput: {throughput:.0f} samples/sec")
        print(f"   Device: {device}")
        
    except Exception as e:
        print(f"❌ Error evaluating Deep Learning: {e}")
    
    return results


# =============================================================================
# BERT EVALUATION
# =============================================================================

def evaluate_bert(test_df, batch_size=32):
    """Evaluate BERT model"""
    print("\n" + "="*60)
    print("EVALUATING BERT MODEL")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "models" / "bert_output"
    
    results = {}
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        
        y_test = test_df["label"].values
        y_pred = []
        
        print(f"\nBERT (DistilBERT):")
        
        # Batch inference
        start_time = time.time()
        for i in range(0, len(test_df), batch_size):
            batch_texts = test_df["text"].iloc[i:i+batch_size].tolist()
            
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                y_pred.extend(preds)
        
        total_time = time.time() - start_time
        inference_time = total_time / len(y_test) * 1000  # ms per sample
        throughput = len(y_test) / total_time  # samples per second
        
        y_pred = np.array(y_pred)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        
        # Model size
        model_size = 0
        for file in model_path.glob("*.safetensors"):
            model_size += file.stat().st_size / (1024 * 1024)
        for file in model_path.glob("*.json"):
            model_size += file.stat().st_size / (1024 * 1024)
        
        results["BERT"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "inference_time": inference_time,
            "throughput": throughput,
            "model_size": model_size,
            "device": str(device).upper(),
            "interpretability": "VERY LOW"
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Inference: {inference_time:.3f}ms/sample")
        print(f"   Throughput: {throughput:.0f} samples/sec")
        print(f"   Device: {device}")
        
    except Exception as e:
        print(f"❌ Error evaluating BERT: {e}")
    
    return results


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def display_comparison_table(all_results):
    """Display comprehensive comparison table"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    if not all_results:
        print("❌ No results to display!")
        return
    
    # Performance Metrics
    print("\n┌" + "─"*78 + "┐")
    print("│ 📊 PERFORMANCE METRICS" + " "*54 + "│")
    print("├" + "─"*78 + "┤")
    print(f"│ {'Model':<20} │ {'Accuracy':<12} │ {'Precision':<12} │ {'Recall':<12} │ {'F1-Score':<12} │")
    print("├" + "─"*78 + "┤")
    
    for model_name, metrics in all_results.items():
        print(f"│ {model_name:<20} │ "
              f"{metrics['accuracy']:>11.4f} │ "
              f"{metrics['precision']:>11.4f} │ "
              f"{metrics['recall']:>11.4f} │ "
              f"{metrics['f1']:>11.4f} │")
    
    print("└" + "─"*78 + "┘")
    
    # Efficiency Metrics
    print("\n┌" + "─"*78 + "┐")
    print("│ ⚡ EFFICIENCY METRICS" + " "*56 + "│")
    print("├" + "─"*78 + "┤")
    print(f"│ {'Model':<20} │ {'Inference (ms)':<20} │ {'Throughput (samples/s)':<30} │")
    print("├" + "─"*78 + "┤")
    
    for model_name, metrics in all_results.items():
        print(f"│ {model_name:<20} │ "
              f"{metrics['inference_time']:>19.3f} │ "
              f"{metrics['throughput']:>29.0f} │")
    
    print("└" + "─"*78 + "┘")
    
    # Resource Metrics
    print("\n┌" + "─"*78 + "┐")
    print("│ 💻 RESOURCE REQUIREMENTS" + " "*52 + "│")
    print("├" + "─"*78 + "┤")
    print(f"│ {'Model':<20} │ {'Device':<15} │ {'Model Size (MB)':<20} │ {'Memory':<15} │")
    print("├" + "─"*78 + "┤")
    
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    for model_name, metrics in all_results.items():
        print(f"│ {model_name:<20} │ "
              f"{metrics['device']:<15} │ "
              f"{metrics['model_size']:>19.2f} │ "
              f"{'~' + str(int(memory_mb)) + ' MB':<15} │")
    
    print("└" + "─"*78 + "┘")
    
    # Operations Metrics
    print("\n┌" + "─"*78 + "┐")
    print("│ 🔧 OPERATIONAL CHARACTERISTICS" + " "*46 + "│")
    print("├" + "─"*78 + "┤")
    print(f"│ {'Model':<20} │ {'Interpretability':<25} │ {'Ease of Deploy':<25} │")
    print("├" + "─"*78 + "┤")
    
    deploy_ease = {
        "Logistic Regression": "EASY",
        "LinearSVC": "EASY",
        "LSTM": "MODERATE",
        "BERT": "MODERATE"
    }
    
    for model_name, metrics in all_results.items():
        ease = deploy_ease.get(model_name, "MODERATE")
        print(f"│ {model_name:<20} │ "
              f"{metrics['interpretability']:<25} │ "
              f"{ease:<25} │")
    
    print("└" + "─"*78 + "┘")
    
    # Summary
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)
    
    best_accuracy = max(all_results.items(), key=lambda x: x[1]["accuracy"])
    best_f1 = max(all_results.items(), key=lambda x: x[1]["f1"])
    fastest = min(all_results.items(), key=lambda x: x[1]["inference_time"])
    smallest = min(all_results.items(), key=lambda x: x[1]["model_size"])
    
    print(f"   🏆 Best Accuracy:     {best_accuracy[0]:<20} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"   🏆 Best F1-Score:     {best_f1[0]:<20} ({best_f1[1]['f1']:.4f})")
    print(f"   ⚡ Fastest Inference:  {fastest[0]:<20} ({fastest[1]['inference_time']:.3f}ms)")
    print(f"   💾 Smallest Model:    {smallest[0]:<20} ({smallest[1]['model_size']:.2f}MB)")
    
    # Interpretability explanation
    print("\n" + "="*80)
    print("📖 INTERPRETABILITY GUIDE")
    print("="*80)
    print("   HIGH:     Can inspect feature weights and understand decision logic")
    print("             Example: Logistic Regression shows which words contribute to prediction")
    print("\n   LOW:      Complex model with learned representations, harder to interpret")
    print("             Example: LSTM learns hidden patterns, requires specialized tools")
    print("\n   VERY LOW: Black box model, requires complex interpretation techniques")
    print("             Example: BERT attention visualization, layer analysis needed")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main benchmark pipeline"""
    print("="*80)
    print("MODEL BENCHMARKING - COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    test_df = load_test_data()
    print(f"   Test samples: {len(test_df)}")
    
    # Evaluate all models
    all_results = {}
    
    # Baseline
    baseline_results = evaluate_baseline(test_df)
    all_results.update(baseline_results)
    
    # Deep Learning
    dl_results = evaluate_deep_learning(test_df)
    all_results.update(dl_results)
    
    # BERT
    bert_results = evaluate_bert(test_df)
    all_results.update(bert_results)
    
    # Display comparison
    display_comparison_table(all_results)
    
    print("\n" + "="*80)
    print("✅ BENCHMARKING COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
