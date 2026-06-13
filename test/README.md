# Testing and Benchmarking

Thư mục này chứa các script để test và so sánh tất cả models.

## 📁 Files

### 1. `compare_all_models.py` - Interactive Testing
**Chức năng:** Test một tin tức với tất cả models và so sánh kết quả real-time

**Cách chạy:**
```bash
python test/compare_all_models.py
```

**Output:**
```
┌──────────────────────────────────────────────────────────────┐
│ Model               │ Prediction │ Confidence │ Time (ms)   │
├──────────────────────────────────────────────────────────────┤
│ Logistic Regression │ FAKE       │ 95.23%     │ 0.123       │
│ LinearSVC           │ FAKE       │ 98.45%     │ 0.089       │
│ LSTM                │ FAKE       │ 92.10%     │ 15.234      │
│ BERT (DistilBERT)   │ FAKE       │ 99.12%     │ 45.678      │
└──────────────────────────────────────────────────────────────┘

Analysis:
  ✅ All models agree: FAKE NEWS
  Average confidence: 96.23%
  Fastest model: LinearSVC (0.089ms)
  Most confident: BERT (99.12%)
```

**Use case:**
- Test một tin tức cụ thể
- So sánh predictions từ tất cả models
- Kiểm tra consensus giữa các models
- Đo inference time real-time

---

### 2. `model_benchmarks.py` - Comprehensive Evaluation
**Chức năng:** Đánh giá toàn diện tất cả models trên test set với đầy đủ metrics

**Cách chạy:**
```bash
python test/model_benchmarks.py
```

**Output:**
```
┌──────────────────────────────────────────────────────────────┐
│ 📊 PERFORMANCE METRICS                                       │
├──────────────────────────────────────────────────────────────┤
│ Model                │ Accuracy   │ Precision  │ Recall     │
├──────────────────────────────────────────────────────────────┤
│ Logistic Regression  │ 0.9842     │ 0.9856     │ 0.9828     │
│ LinearSVC            │ 0.9925     │ 0.9938     │ 0.9912     │
│ LSTM                 │ 0.9756     │ 0.9780     │ 0.9732     │
│ BERT                 │ 0.9950     │ 0.9962     │ 0.9938     │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ ⚡ EFFICIENCY METRICS                                        │
├──────────────────────────────────────────────────────────────┤
│ Model                │ Inference (ms) │ Throughput (s/s)    │
├──────────────────────────────────────────────────────────────┤
│ Logistic Regression  │ 0.123          │ 8130                │
│ LinearSVC            │ 0.089          │ 11235               │
│ LSTM                 │ 15.234         │ 65                  │
│ BERT                 │ 45.678         │ 21                  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 💻 RESOURCE REQUIREMENTS                                     │
├──────────────────────────────────────────────────────────────┤
│ Model                │ Device  │ Model Size (MB) │ Memory   │
├──────────────────────────────────────────────────────────────┤
│ Logistic Regression  │ CPU     │ 2.5             │ ~500 MB  │
│ LinearSVC            │ CPU     │ 1.8             │ ~500 MB  │
│ LSTM                 │ GPU     │ 12.3            │ ~800 MB  │
│ BERT                 │ GPU     │ 255.7           │ ~1200 MB │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 🔧 OPERATIONAL CHARACTERISTICS                               │
├──────────────────────────────────────────────────────────────┤
│ Model                │ Interpretability │ Ease of Deploy    │
├──────────────────────────────────────────────────────────────┤
│ Logistic Regression  │ HIGH             │ EASY              │
│ LinearSVC            │ HIGH             │ EASY              │
│ LSTM                 │ LOW              │ MODERATE          │
│ BERT                 │ VERY LOW         │ MODERATE          │
└──────────────────────────────────────────────────────────────┘

📈 SUMMARY
   🏆 Best Accuracy:     BERT (0.9950)
   🏆 Best F1-Score:     BERT (0.9951)
   ⚡ Fastest Inference:  LinearSVC (0.089ms)
   💾 Smallest Model:    LinearSVC (1.8MB)
```

**Use case:**
- So sánh toàn diện tất cả models
- Đánh giá trên toàn bộ test set (4417 samples)
- Phân tích trade-offs giữa performance/speed/size
- Đưa ra decision cho production deployment

---

## 🎯 So Sánh 2 Approaches

| Đặc điểm | `compare_all_models.py` | `model_benchmarks.py` |
|----------|-------------------------|------------------------|
| **Input** | 1 tin tức (user nhập) | Toàn bộ test set |
| **Tốc độ** | Real-time (< 1s) | Chậm (~1-2 phút) |
| **Output** | Prediction comparison | Comprehensive metrics |
| **Use case** | Demo, testing mẫu | Production decision |
| **Interactive** | ✅ Yes | ❌ No |

---

## 📖 Giải Thích Các Tiêu Chí

### 📊 PERFORMANCE (Hiệu Suất)
- **Accuracy**: Tỷ lệ dự đoán đúng tổng thể
- **Precision**: Trong số dự đoán FAKE, bao nhiêu % thực sự là FAKE
- **Recall**: Trong số tin FAKE thực tế, bao nhiêu % được phát hiện
- **F1-Score**: Trung bình điều hòa của Precision và Recall

### ⚡ EFFICIENCY (Hiệu Năng)
- **Inference Time**: Thời gian dự đoán 1 sample (ms)
- **Throughput**: Số sample xử lý được trong 1 giây

### 💻 RESOURCES (Tài Nguyên)
- **Device**: CPU hay GPU
- **Model Size**: Kích thước file model (MB)
- **Memory**: RAM sử dụng khi chạy

### 🔧 OPERATIONS (Vận Hành)
- **Interpretability**: Mức độ hiểu được quyết định của model
  - **HIGH**: Có thể xem trọng số từng feature (Baseline)
  - **LOW**: Học hidden patterns (LSTM)
  - **VERY LOW**: Black box (BERT)
- **Ease of Deploy**: Độ dễ triển khai production
  - **EASY**: Chỉ cần CPU, dependencies ít
  - **MODERATE**: Cần GPU hoặc dependencies nhiều

---

## 🚀 Workflow Đề Xuất

### 1. Development Phase
```bash
# Test với tin tức mẫu
python test/compare_all_models.py
```

### 2. Evaluation Phase
```bash
# Benchmark toàn diện
python test/model_benchmarks.py
```

### 3. Decision Making
- Nếu cần **SPEED + INTERPRETABILITY** → Chọn **LinearSVC**
- Nếu cần **BEST ACCURACY** → Chọn **BERT**
- Nếu cân bằng → Chọn **Logistic Regression**

---

## 📝 Requirements

Tất cả models phải được train trước:
```bash
# 1. Baseline models
python src/train_baseline.py

# 2. Deep Learning (LSTM)
python src/train_deep_learning.py

# 3. BERT
# Train trên Google Colab theo hướng dẫn trong COLAB_GUIDE.md
```

---

## ⚠️ Lưu Ý

1. **Text Preprocessing**: Text được auto clean bằng `utils/preprocessor.py`
2. **GPU Support**: LSTM và BERT tự động dùng GPU nếu có
3. **Error Handling**: Nếu model chưa train, sẽ bỏ qua và tiếp tục với models khác
4. **Memory**: BERT có thể tốn ~1-2GB RAM khi load

---

## 🐛 Troubleshooting

### Lỗi: "Model not found"
```bash
# Kiểm tra models đã train chưa
ls models/baseline/
ls models/deep_learning/
ls models/bert_output/
```

### Lỗi: "CUDA out of memory"
```bash
# Giảm batch_size trong model_benchmarks.py
# Hoặc chạy trên CPU (tự động fallback)
```

### Lỗi: "Module not found"
```bash
# Cài dependencies
pip install -r requirements.txt
```
