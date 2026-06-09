import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


# LOAD DATA (tu data_loader)

def load_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_path = os.path.join(base_dir, "data", "processed", "train.csv")
    test_path = os.path.join(base_dir, "data", "processed", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


# TRAIN MODEL

def train():
    train_df, test_df = load_data()

    # DUNG cleaned_text
    X_train = train_df["cleaned_text"]
    y_train = train_df["label"]

    X_test = test_df["cleaned_text"]
    y_test = test_df["label"]

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # MODEL
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # EVALUATE
    y_pred = model.predict(X_test_tfidf)

    print("===== Logistic Regression =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # SAVE MODEL
    
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    tfidf_path = os.path.join(models_dir, "tfidf.pkl")
    model_path = os.path.join(models_dir, "logreg.pkl")

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(model, model_path)

    print("\nDa luu model:")
    print("TF-IDF:", tfidf_path)
    print("Model:", model_path)


# PREDICT FUNCTION

def predict_classical(text):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    model_path = os.path.join(base_dir, "models", "logreg.pkl")
    tfidf_path = os.path.join(base_dir, "models", "tfidf.pkl")

    # load model
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)

    # KHONG CAN clean (da clean o data_loader)
    text_vec = tfidf.transform([text])

    pred = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0]

    confidence = prob[pred]

    label = "Fake" if pred == 1 else "Real"

    return label, confidence, prob


if __name__ == "__main__":
    print("Bat dau training...")
    train()

    print("\n=== TEST ===")

    while True:
        user_input = input("\nNhap tin tuc (go 'exit' de thoat): ")

        if user_input.lower() == "exit":
            print("Thoat chuong trinh")
            break

        label, confidence, prob = predict_classical(user_input)

        print("\nKET QUA:")
        print("Du doan:", label)
        print(f"Do tin cay: {confidence * 100:.2f}%")
        print(f"Fake: {prob[1] * 100:.2f}% | Real: {prob[0] * 100:.2f}%")