import os
import re
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

     
# CONFIG
     
TRUE_PATH = "data/raw/True.csv"
FAKE_PATH = "data/raw/Fake.csv"
MODEL_PATH = "models/model.pth"

     
# CLEAN TEXT
     
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

     
# LOAD DATA
     
def load_data():
    df_true = pd.read_csv(TRUE_PATH)
    df_fake = pd.read_csv(FAKE_PATH)

    df_true["label"] = 0
    df_fake["label"] = 1

    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Dùng title + text
    df["content"] = df["title"] + " " + df["text"]

    texts = df["content"].apply(clean_text)
    labels = df["label"].values

    texts = texts[:20000]
    labels = labels[:20000]

    return texts, labels

     
# TRAIN
     
def train():
    texts, labels = load_data()


    vectorizer = CountVectorizer(max_features=5000)

    X = vectorizer.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Class weight (fix bias)
    fake_count = sum(y_train)
    real_count = len(y_train) - fake_count
    pos_weight = torch.tensor([real_count / fake_count])


    # Convert sau split 
    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Model
    model = nn.Sequential(
        nn.Linear(5000, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Training...")

    for epoch in range(4):
        total_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb).squeeze()

            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vectorizer": vectorizer
    }, MODEL_PATH)

    return model, vectorizer

     
# PREDICT
     
def predict(text, model, vectorizer):
    text = clean_text(text)

    X = vectorizer.transform([text]).toarray()
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = torch.sigmoid(model(X)).item()

#Dac biet ne nhe
    if output >= 0.6:
        print(f"FAKE NEWS  | {output:.4f}")
    else:
        print(f"REAL NEWS  | {output:.4f}")

     
# MAIN
     
if __name__ == "__main__":
    print("START TRAIN")

    model, vectorizer = train()


    while True:
        txt = input("Nhập văn bản: ")
        if txt == "exit":
            break

        predict(txt, model, vectorizer)