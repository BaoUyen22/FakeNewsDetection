# Liner , TF-IDF , Logistic 
import os
import sys
import joblib
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report, 
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
from scipy.sparse import hstack

# Add parent directory to path để import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocessor_for_model import MLTextPreprocessor

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load train, val, test datasets"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    train_path = os.path.join(base_dir, "data", "processed", "train.csv")
    val_path = os.path.join(base_dir, "data", "processed", "val.csv")
    test_path = os.path.join(base_dir, "data", "processed", "test.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, val_df, test_df


def preprocess_data(train_df, val_df, test_df):
    """Clean and apply ML-specific preprocessing"""
    for df in [train_df, val_df, test_df]:
        df["text"] = df["text"].fillna("").astype(str)
        df.drop(df[df["text"].str.strip().str.len() == 0].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    for df in [train_df, val_df, test_df]:
        # Apply MLTextPreprocessor, returns tuple (cleaned_text, num_excl, num_quest)
        df[["text_ml", "num_exclamation", "num_question"]] = df["text"].apply(
            lambda x: pd.Series(MLTextPreprocessor.transform(x))
        )
        # Remove rows where cleaned text is empty
        df.drop(df[df["text_ml"].str.strip().str.len() == 0].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


# =============================================================================
# TRAINING
# =============================================================================

def train_models(X_train, y_train):
    """
    Train multiple baseline models
    
    Returns:
        dict: Trained models
    """
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
            C=1.0  # Regularization strength
        ),
        "linear_svc": LinearSVC(
            max_iter=1000, 
            random_state=42, 
            dual='auto',
            class_weight='balanced'  # Handle class imbalance
        ),
    }
    trained_models = {}
    training_times = {}
    
    for i, (name, model) in enumerate(models.items(), start=1):
        print(f"\n[{i}/{len(models)}] Training {name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        trained_models[name] = model
        training_times[name] = train_time
        
        print(f"   Training time: {train_time:.2f}s")
    
    return trained_models, training_times


# =============================================================================
# VALIDATION
# =============================================================================

def evaluate_on_validation(models, X_val, y_val):
    """
    Evaluate models on validation set
    
    Returns:
        dict: Validation metrics for each model
    """
    print("\n" + "="*60)
    print("VALIDATION EVALUATION")
    print("="*60)
    
    val_results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='binary'
        )
        
        val_results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
    
    return val_results


# =============================================================================
# FINAL TEST EVALUATION
# =============================================================================

def evaluate_on_test(models, X_test, y_test):
    """
    Final evaluation on test set
    
    Returns:
        dict: Test metrics for each model
    """
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    test_results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Predict / dự đoán
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(y_test) * 1000  # ms per sample
        
        # Metrics / số liệu 
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        
        test_results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "y_pred": y_pred,
            "inference_time": inference_time
        }
        
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   Inference: {inference_time:.3f}ms/sample")
    
    return test_results


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_models(models, train_times, val_results, test_results):
    """
    Compare all models using multiple criteria
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    print("\n📊 PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"{'Model':<25} {'Val Acc':<12} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)
    
    for name in models.keys():
        val_acc = val_results[name]["accuracy"]
        test_acc = test_results[name]["accuracy"]
        precision = test_results[name]["precision"]
        recall = test_results[name]["recall"]
        f1 = test_results[name]["f1"]
        
        print(f"{name:<25} {val_acc:<12.4f} {test_acc:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    print("\n⚡ EFFICIENCY METRICS:")
    print("-" * 60)
    print(f"{'Model':<25} {'Train Time':<15} {'Inference Time':<20}")
    print("-" * 60)
    
    for name in models.keys():
        train_time = train_times[name]
        inference_time = test_results[name]["inference_time"]
        
        print(f"{name:<25} {train_time:<15.2f}s {inference_time:<20.3f}ms/sample")
    
    print("\n💾 MODEL SIZE:")
    print("-" * 40)
    print(f"{'Model':<25} {'Size':<15}")
    print("-" * 40)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models", "baseline")
    
    for name in models.keys():
        model_path = os.path.join(models_dir, f"{name}.pkl")
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"{name:<25} {size_mb:<15.2f}MB")
    
    # TF-IDF vectorizer size
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_path):
        tfidf_size = os.path.getsize(tfidf_path) / (1024 * 1024)
        print(f"{'TF-IDF Vectorizer':<25} {tfidf_size:<15.2f}MB")


# =============================================================================
# BEST MODEL SELECTION
# =============================================================================

def select_best_model(test_results, y_test):
    """
    Select and display best model based on test accuracy
    """
    best_model_name = max(test_results.keys(), key=lambda k: test_results[k]["accuracy"])
    best_accuracy = test_results[best_model_name]["accuracy"]
    best_f1 = test_results[best_model_name]["f1"]
    
    print("\n" + "="*60)
    print("BEST MODEL")
    print("="*60)
    print(f"Model: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"F1-Score: {best_f1:.4f}")
    
    print("\n" + "-"*60)
    print("DETAILED CLASSIFICATION REPORT:")
    print("-"*60)
    print(classification_report(
        y_test, 
        test_results[best_model_name]["y_pred"],
        target_names=["Real (0)", "Fake (1)"],
        digits=4
    ))
    
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_test, test_results[best_model_name]["y_pred"]))
    
    return best_model_name


# =============================================================================
# MODEL SAVING
# =============================================================================

def save_models(models, tfidf):
    """Save trained models and vectorizer"""
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models", "baseline")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save vectorizer
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    joblib.dump(tfidf, tfidf_path)
    print(f"✅ TF-IDF Vectorizer: {tfidf_path}")
    
    # Save models
    for name, model in models.items():
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ {name}: {model_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main training pipeline"""
    print("="*60)
    print("BASELINE MODELS TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    train_df, val_df, test_df = load_data()
    train_df, val_df, test_df = preprocess_data(train_df, val_df, test_df)
    
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    
    # 2. Vectorization
    print("\n[2/7] TF-IDF Vectorization...")
    tfidf = TfidfVectorizer(
        max_features=15000,  # Tăng từ 10000
        ngram_range=(1, 3),  # Thêm trigrams
        min_df=1,  # Giảm từ 2 (giữ nhiều từ hơn)
        max_df=0.85,  # Giảm từ 0.9 (loại bỏ từ quá phổ biến)
        sublinear_tf=True  # Apply sublinear tf scaling
    )
    
    X_train_tfidf = tfidf.fit_transform(train_df["text_ml"])
    X_val_tfidf = tfidf.transform(val_df["text_ml"])
    X_test_tfidf = tfidf.transform(test_df["text_ml"])
    # Lấy đặc trưng số
    train_excl = np.array(train_df["num_exclamation"]).reshape(-1, 1)
    train_quest = np.array(train_df["num_question"]).reshape(-1, 1)
    val_excl   = np.array(val_df["num_exclamation"]).reshape(-1, 1)
    val_quest  = np.array(val_df["num_question"]).reshape(-1, 1)
    test_excl  = np.array(test_df["num_exclamation"]).reshape(-1, 1)
    test_quest = np.array(test_df["num_question"]).reshape(-1, 1)

    # Ghép TF-IDF với 2 cột số
    X_train = hstack([X_train_tfidf, train_excl, train_quest])
    X_val   = hstack([X_val_tfidf, val_excl, val_quest])
    X_test  = hstack([X_test_tfidf, test_excl, test_quest])
    print(f"   Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"   Train matrix shape: {X_train_tfidf.shape}")
    print(f"   Val matrix shape:   {X_val_tfidf.shape}")
    print(f"   Test matrix shape:  {X_test_tfidf.shape}")
    
    # 3. Training
    print("\n[3/7] Training models...")
    models, train_times = train_models(X_train, train_df["label"])
    
    # 4. Validation
    print("\n[4/7] Validation evaluation...")
    val_results = evaluate_on_validation(models, X_val, val_df["label"])
    
    # 5. Final test
    print("\n[5/7] Final test evaluation...")
    test_results = evaluate_on_test(models, X_test, test_df["label"])
    
    # 6. Comparison
    print("\n[6/7] Comparing models...")
    compare_models(models, train_times, val_results, test_results)
    
    # 7. Best model
    print("\n[7/7] Selecting best model...")
    best_model = select_best_model(test_results, test_df["label"])
    
    # Save models
    save_models(models, tfidf)
    
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETED!")
    print("="*60)
    print(f"\nBest Model: {best_model}")
    print(f"Accuracy: {test_results[best_model]['accuracy']:.4f}")
    print(f"F1-Score: {test_results[best_model]['f1']:.4f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
