import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


# LOAD DATA (from data_loader)

def load_data():
    """Load train, val, test từ data/processed/"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_path = os.path.join(base_dir, "data", "processed", "train.csv")
    val_path = os.path.join(base_dir, "data", "processed", "val.csv")
    test_path = os.path.join(base_dir, "data", "processed", "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


# TRAIN MODEL

def train():
    """Train baseline models: Logistic Regression, LinearSVC, Naive Bayes"""
    train_df, val_df, test_df = load_data()

    # Sử dụng cột 'text' (đã cleaned trong data_loader)
    # Xử lý NaN và chuỗi rỗng
    train_df["text"] = train_df["text"].fillna("").astype(str)
    val_df["text"] = val_df["text"].fillna("").astype(str)
    test_df["text"] = test_df["text"].fillna("").astype(str)
    
    # Lọc bỏ các dòng có text rỗng
    train_df = train_df[train_df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    val_df = val_df[val_df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    test_df = test_df[test_df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    
    X_train = train_df["text"]
    y_train = train_df["label"]

    X_val = val_df["text"]
    y_val = val_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # TF-IDF Vectorizer
    print("\n[1/4] Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )

    X_train_tfidf = tfidf.fit_transform(X_train)  # Fit chỉ trên TRAIN
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"   Vocabulary size: {len(tfidf.vocabulary_)}")

    # Train multiple models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "linear_svc": LinearSVC(max_iter=1000, random_state=42),
        "naive_bayes": MultinomialNB()
    }

    results = {}

    for i, (name, model) in enumerate(models.items(), start=2):
        print(f"\n[{i}/4] Training {name}...")
        
        # Train
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate on val
        y_val_pred = model.predict(X_val_tfidf)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Evaluate on test
        y_test_pred = model.predict(X_test_tfidf)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        results[name] = {
            "val_acc": val_acc,
            "test_acc": test_acc,
            "y_pred": y_test_pred
        }
        
        print(f"   Val Accuracy:  {val_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")

    # Print detailed report for best model
    best_model_name = max(results.keys(), key=lambda k: results[k]["test_acc"])
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*60}")
    print(classification_report(y_test, results[best_model_name]["y_pred"], 
                                target_names=["Real (0)", "Fake (1)"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, results[best_model_name]["y_pred"]))

    # SAVE MODELS
    print(f"\n{'='*60}")
    print("SAVING MODELS...")
    print(f"{'='*60}")
    
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models", "baseline")
    os.makedirs(models_dir, exist_ok=True)

    # Save vectorizer
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    joblib.dump(tfidf, tfidf_path)
    print(f"✅ TF-IDF: {tfidf_path}")

    # Save models
    for name, model in models.items():
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ {name}: {model_path}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print(f"{'='*60}")


# PREDICT FUNCTION

def predict_baseline(text):
    """
    Predict với baseline model
    
    Args:
        text: Text ĐÃ CLEANED (sử dụng preprocessor trước khi gọi hàm này)
    
    Returns:
        label, confidence, proba
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models", "baseline")

    model_path = os.path.join(models_dir, "logistic_regression.pkl")
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

    # Load model
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)

    # Vectorize
    text_vec = tfidf.transform([text])

    # Predict
    pred = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0]
    confidence = prob[pred]

    label = "FAKE" if pred == 1 else "REAL"

    return label, confidence, prob


if __name__ == "__main__":
    print("="*60)
    print("BASELINE MODELS TRAINING")
    print("="*60)
    train()

    # Interactive testing
    print("\n" + "="*60)
    print("INTERACTIVE TESTING")
    print("="*60)
    print("NOTE: Nhập text ĐÃ CLEANED (lowercase, no special chars)")
    print("      Hoặc sử dụng utils/preprocessor.py để clean trước")
    print("="*60)

    while True:
        user_input = input("\nNhập tin tức (gõ 'exit' để thoát): ")

        if user_input.lower() == "exit":
            print("Thoát chương trình")
            break

        label, confidence, prob = predict_baseline(user_input)

        print("\nKẾT QUẢ:")
        print(f"Dự đoán: {label}")
        print(f"Độ tin cậy: {confidence * 100:.2f}%")
        print(f"  FAKE: {prob[1] * 100:.2f}%")
        print(f"  REAL: {prob[0] * 100:.2f}%")
