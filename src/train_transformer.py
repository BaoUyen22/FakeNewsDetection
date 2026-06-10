import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os

# 1. Cấu hình đường dẫn (Ép mọi thứ về ổ E)
CACHE_DIR = "E:/FakeNewsDetection/models/cache" 
OUTPUT_DIR = "E:/FakeNewsDetection/models/bert_output" 
MODEL_NAME = "distilbert-base-uncased" 

def train_transformer():
    # 2. Đọc dữ liệu 
    # KHAI BÁO BIẾN data_path TRƯỚC KHI DÙNG
    data_path = "data/processed/train.csv" 

    # Kiểm tra xem file có tồn tại không trước khi đọc
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file {data_path}. Hãy kiểm tra lại thư mục data/processed/")
        return

    print(f"--- Đang đọc dữ liệu từ {data_path} ---")
    df = pd.read_csv(data_path)
    
    # Để test nhanh ở máy cá nhân, thầy khuyên em nên lấy 200 dòng đầu để chạy thử code thôi
    # Khi nào lên Colab thì xóa dòng .head(200) này đi
    df = df.head(200) 
    
    # 3. Chuẩn bị Dataset cho Hugging Face
    # Sử dụng đúng tên cột 'cleaned_text' mà data_loader.py đã tạo ra
    dataset = Dataset.from_pandas(df[['cleaned_text', 'label']])
    
    # 4. Tokenizer (Bộ băm chữ)
    print(f"--- Đang tải Tokenizer {MODEL_NAME} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    def tokenize_func(examples):
        return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=128)

    print("--- Đang Tokenize dữ liệu ---")
    tokenized_dataset = dataset.map(tokenize_func, batched=True)

    # 5. Khởi tạo mô hình
    print(f"--- Đang tải mô hình {MODEL_NAME} ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2, 
        cache_dir=CACHE_DIR
    )

    # 6. Cấu hình huấn luyện (Thông số nháp để test code local)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, # Chỉnh xuống 4 cho nhẹ máy local
        num_train_epochs=1,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="no",
        use_cpu=True # Ép chạy bằng CPU để test local cho ổn định
    )

    # 7. Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # 8. Chạy thử nghiệm
    print("--- Bắt đầu chạy thử nghiệm huấn luyện (200 dòng) ---")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("--- THÀNH CÔNG! Code đã thông luồng hoàn toàn ---")

if __name__ == "__main__":
    train_transformer()