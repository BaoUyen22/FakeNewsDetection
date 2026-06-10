import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# 1. Đường dẫn tới nơi em đã lưu model (Ổ E)
MODEL_PATH = "E:/FakeNewsDetection/models/bert_output"

from langdetect import detect, DetectorFactory
# Giúp kết quả nhận diện ngôn ngữ ổn định hơn
DetectorFactory.seed = 42 

def predict_news_secure(text):
    # 1. Kiểm tra đầu vào trống
    if not text.strip():
        return "LỖI: Bạn chưa nhập nội dung!", 0

    # 2. KIỂM TRA NGÔN NGỮ
    try:
        lang = detect(text)
        if lang != 'en':
            return f"CẢNH BÁO: Mô hình này chỉ hỗ trợ Tiếng Anh (Phát hiện: {lang}). Vui lòng nhập lại bằng Tiếng Anh!", 0
    except:
        return "LỖI: Không thể nhận diện được ngôn ngữ. Vui lòng nhập đoạn văn bản rõ ràng!", 0
    # 2. Tải Tokenizer và Model đã học
    print("--- Đang tải mô hình từ ổ E... ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # 3. Tiền xử lý câu đầu vào (Giống như lúc train)
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")

    # 4. Dự đoán
    model.eval() # Chuyển sang chế độ dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 5. Tính xác suất (Softmax)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    
    # Lấy nhãn có xác suất cao nhất
    # Theo data_loader của em: 1 là FAKE, 0 là TRUE
    confidence, prediction = torch.max(probs, dim=1)
    
    label_map = {0: "TIN THẬT (TRUE)", 1: "TIN GIẢ (FAKE)"}
    
    return label_map[prediction.item()], confidence.item()

# --- CHẠY THỬ NGHIỆM ---
if __name__ == "__main__":
    while True:
        print("\n" + "="*30)
        user_input = input("Nhập tin tức tiếng Anh cần kiểm tra (hoặc gõ 'exit' để thoát): ")
        if user_input.lower() == 'exit':
            break
        
        result, score = predict_news_secure(user_input)
        print(f"\nKẾT QUẢ: {result}")
        print(f"ĐỘ TIN CẬY: {score*100:.2f}%")