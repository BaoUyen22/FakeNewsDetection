# Tech Stack Chi Tiết - Fake News Detection System

## 📋 Tổng Quan

Document này mô tả chi tiết **chức năng và vai trò** của từng thư viện/framework được sử dụng trong dự án.

---

## 🐍 Core Language

### **Python 3.11**
- **Vai trò**: Ngôn ngữ lập trình chính
- **Lý do chọn**:
  - Hệ sinh thái ML/NLP phong phú (sklearn, PyTorch, Transformers)
  - Syntax đơn giản, dễ maintain
  - Performance tốt với NumPy/PyTorch backend
  - Community support mạnh
- **Version**: 3.11.x (cải thiện performance 10-60% so với 3.10)

---

## 🤖 Machine Learning Frameworks

### **Scikit-learn 1.3+**

**Vai trò chính**: Baseline Models & Classical ML

#### Chức năng sử dụng:

1. **Feature Extraction**
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   ```
   - Convert text → numerical features (TF-IDF vectors)
   - Vocabulary building: 10,000 terms
   - N-gram support: unigrams + bigrams

2. **Classification Models**
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.svm import LinearSVC
   ```
   - **Logistic Regression**: Baseline probabilistic classifier
   - **Linear SVC**: High-performance linear classifier
   - Regularization: L2 penalty để tránh overfitting

3. **Model Evaluation**
   ```python
   from sklearn.metrics import accuracy_score, precision_recall_fscore_support
   ```
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix
   - Classification report

4. **Data Splitting**
   ```python
   from sklearn.model_selection import train_test_split
   ```
   - Stratified split: 80% train / 10% val / 10% test
   - Preserve class distribution (balanced REAL/FAKE ratio)

5. **Model Persistence**
   ```python
   import joblib
   joblib.dump(model, 'model.pkl')
   ```
   - Serialize trained models
   - Fast loading for inference

**Kết quả**: 
- Logistic Regression: 94.2% accuracy
- Linear SVC: 94.5% accuracy

---

### **PyTorch 2.4+**

**Vai trò chính**: Deep Learning Models (LSTM, BERT)

#### Chức năng sử dụng:

1. **Neural Network Building**
   ```python
   import torch.nn as nn
   
   class LSTMClassifier(nn.Module):
       def __init__(self):
           self.embedding = nn.Embedding(vocab_size, 100)
           self.lstm = nn.LSTM(100, 128, batch_first=True)
           self.fc = nn.Linear(128, 1)
   ```
   - Define custom architectures
   - Embedding layers cho word representations
   - LSTM layers cho sequence modeling

2. **GPU Acceleration**
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   ```
   - CUDA support: 10-50x faster training
   - Automatic mixed precision (AMP) với fp16
   - Batch processing trên GPU

3. **Training Loop**
   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.BCELoss()
   ```
   - Custom training loops
   - Backpropagation & gradient descent
   - Loss computation (BCE, CrossEntropy)

4. **Model Inference**
   ```python
   with torch.no_grad():
       outputs = model(inputs)
       predictions = torch.sigmoid(outputs)
   ```
   - No gradient computation (faster)
   - Batch predictions
   - Softmax/sigmoid cho probabilities

**Kết quả**:
- LSTM: 93.8% accuracy
- Training time: 15 mins trên T4 GPU (vs 2 hours trên CPU)

---

## 📚 NLP Libraries

### **Transformers (HuggingFace) 4.36+**

**Vai trò chính**: Pre-trained Language Models

#### Chức năng sử dụng:

1. **Model Loading**
   ```python
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   
   model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
   tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   ```
   - Load pre-trained DistilBERT (66M parameters)
   - Fine-tune cho binary classification (REAL/FAKE)

2. **Tokenization**
   ```python
   inputs = tokenizer(
       text,
       max_length=512,
       truncation=True,
       padding=True,
       return_tensors="pt"
   )
   ```
   - WordPiece tokenization
   - Handle long sequences (max 512 tokens)
   - Attention masks, special tokens ([CLS], [SEP])

3. **Training (Trainer API)**
   ```python
   from transformers import Trainer, TrainingArguments
   
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       compute_metrics=compute_metrics
   )
   trainer.train()
   ```
   - Simplified training pipeline
   - Automatic checkpointing
   - Mixed precision training (fp16)
   - Evaluation metrics integration

4. **Transfer Learning**
   - Pre-trained on 16GB+ text (Wikipedia, BookCorpus)
   - Fine-tune last layer cho fake news task
   - 3 epochs training (~30 mins trên T4 GPU)

**Kết quả**:
- BERT (DistilBERT): **98.9% accuracy** (best model)
- Inference: 50ms per article (với GPU)

---

### **NLTK 3.8+**

**Vai trò phụ**: Text Preprocessing Support

#### Chức năng sử dụng:

1. **Stopwords**
   ```python
   from nltk.corpus import stopwords
   stop_words = set(stopwords.words('english'))
   ```
   - Remove common words (a, the, is, are...)
   - 179 English stopwords
   - Backup cho custom stopword list

2. **Tokenization** (không dùng chính)
   ```python
   from nltk.tokenize import word_tokenize
   ```
   - Fallback tokenizer
   - Project chủ yếu dùng custom tokenization

**Note**: NLTK được giữ lại cho tính tương thích, nhưng preprocessing chính dùng custom logic trong `utils/preprocessor_for_model.py`

---

## 🌐 Web Frameworks

### **Streamlit 1.29+**

**Vai trò chính**: Interactive Web UI

#### Chức năng sử dụng:

1. **UI Components**
   ```python
   import streamlit as st
   
   text = st.text_area("Nhập nội dung bài báo...")
   if st.button("Phân tích ngay"):
       result = predict(text)
       st.success(f"Kết quả: {result}")
   ```
   - Text input, buttons, sliders
   - Real-time updates
   - No HTML/CSS/JS needed

2. **Data Visualization**
   ```python
   st.plotly_chart(fig)
   st.metric("Confidence", f"{confidence:.1%}")
   ```
   - Embedded charts (Plotly, Matplotlib)
   - Metrics dashboard
   - Progress bars, spinners

3. **Caching**
   ```python
   @st.cache_resource
   def load_model():
       return joblib.load('model.pkl')
   ```
   - Cache loaded models (avoid reloading)
   - Session state management
   - Performance optimization

4. **Layout**
   ```python
   col1, col2 = st.columns(2)
   with col1:
       st.write("Model A")
   with col2:
       st.write("Model B")
   ```
   - Multi-column layouts
   - Sidebar navigation
   - Tabs, expanders

**Deployment**: 
```bash
streamlit run app.py
# → http://localhost:8501
```

---

### **FastAPI 0.115+**

**Vai trò chính**: RESTful API Backend

#### Chức năng sử dụng:

1. **Route Definition**
   ```python
   from fastapi import FastAPI, HTTPException
   
   app = FastAPI()
   
   @app.post("/predict/text/{model_name}")
   async def predict_text(model_name: str, request: TextRequest):
       result = model.predict(request.text)
       return {"label": result.label, "confidence": result.confidence}
   ```
   - Type hints cho automatic validation
   - Async support (concurrent requests)
   - RESTful endpoints

2. **Request/Response Models**
   ```python
   from pydantic import BaseModel
   
   class TextPredictionResponse(BaseModel):
       model: str
       label: str
       confidence: float
       fake_probability: float
       real_probability: float
   ```
   - Data validation với Pydantic
   - Auto-generate JSON schema
   - Type safety

3. **Auto Documentation**
   ```python
   # Automatic Swagger UI
   # → http://localhost:8000/docs
   
   # Automatic ReDoc
   # → http://localhost:8000/redoc
   ```
   - Interactive API testing
   - Auto-generated from code
   - No manual doc writing

4. **Error Handling**
   ```python
   @app.exception_handler(ValueError)
   async def value_error_handler(request, exc):
       return JSONResponse(
           status_code=400,
           content={"detail": str(exc)}
       )
   ```
   - Custom error responses
   - HTTP status codes
   - Detailed error messages

**Deployment**:
```bash
uvicorn server.main:app --reload
# → http://localhost:8000
```

**Endpoints**:
- `POST /predict/text/{model}` - Predict from text
- `GET /predict/{model}/{url}` - Crawl URL & predict
- `GET /health` - Health check

---

## 🕷️ Web Crawling

### **Playwright 1.40+**

**Vai trò chính**: JavaScript-heavy Website Crawling

#### Chức năng sử dụng:

1. **Headless Browser**
   ```python
   from playwright.async_api import async_playwright
   
   async with async_playwright() as p:
       browser = await p.chromium.launch(headless=True)
       page = await browser.new_page()
       await page.goto(url, wait_until="networkidle")
   ```
   - Full browser engine (Chromium, Firefox, WebKit)
   - Execute JavaScript
   - Wait for AJAX/dynamic content

2. **Content Extraction**
   ```python
   content = await page.evaluate("""() => {
       return document.body.innerText;
   }""")
   ```
   - Run JS in browser context
   - Extract rendered content
   - Handle SPAs (React, Vue, Angular)

3. **Use Cases**:
   - Modern news sites với infinite scroll
   - Sites với lazy loading images
   - Anti-scraping protection bypass

**Installation**:
```bash
pip install playwright
playwright install chromium  # Download browser binary
```

---

### **BeautifulSoup 4.12+**

**Vai trò chính**: HTML Parsing (Static Sites)

#### Chức năng sử dụng:

1. **HTML Parsing**
   ```python
   from bs4 import BeautifulSoup
   
   soup = BeautifulSoup(html, 'html.parser')
   title = soup.find('h1').get_text()
   paragraphs = soup.find_all('p')
   ```
   - Parse HTML structure
   - CSS selectors, tag search
   - Extract text, attributes

2. **Content Cleaning**
   ```python
   # Remove scripts, styles
   for script in soup(['script', 'style']):
       script.decompose()
   
   text = soup.get_text(separator=' ', strip=True)
   ```
   - Remove unwanted tags
   - Clean whitespace
   - Join text nodes

3. **Use Cases**:
   - Static news sites
   - Faster than Playwright (no browser)
   - Fallback nếu Playwright fail

**Workflow**:
```python
# Try BeautifulSoup first (fast)
if fails or ?use_js=true:
    # Use Playwright (slower but handles JS)
```

---

## 📊 Data Processing

### **Pandas 2.1+**

**Vai trò chính**: Tabular Data Processing

#### Chức năng sử dụng:

1. **Data Loading**
   ```python
   import pandas as pd
   
   df = pd.read_csv('data/raw/Fake.csv')
   df['label'] = 1  # FAKE
   ```
   - CSV, JSON, Excel support
   - 44,898 articles loaded
   - Memory-efficient chunk loading

2. **Data Cleaning**
   ```python
   df = df.dropna(subset=['text', 'label'])
   df = df.drop_duplicates(subset=['text'])
   df['text'] = df['text'].apply(clean_text)
   ```
   - Handle missing values
   - Remove duplicates (1,247 found)
   - Apply preprocessing functions

3. **Data Transformation**
   ```python
   df['text_length'] = df['text'].str.len()
   df['has_url'] = df['text'].str.contains('http')
   ```
   - Feature engineering
   - String operations
   - Column-wise computations

4. **Data Splitting**
   ```python
   train, test = train_test_split(df, test_size=0.2, stratify=df['label'])
   ```
   - Train/val/test splits
   - Stratified sampling
   - Save processed data

**Statistics**:
- Raw data: 44,898 articles
- After cleaning: 43,651 articles
- Train: 34,921 | Val: 4,365 | Test: 4,365

---

### **NumPy 1.26+**

**Vai trò chính**: Numerical Computations

#### Chức năng sử dụng:

1. **Array Operations**
   ```python
   import numpy as np
   
   # Feature array
   X = np.array([[1, 2, 3], [4, 5, 6]])
   
   # Matrix multiplication
   result = np.dot(X, weights)
   ```
   - N-dimensional arrays
   - Vectorized operations (fast)
   - Broadcasting

2. **Statistical Functions**
   ```python
   mean = np.mean(array)
   std = np.std(array)
   normalized = (array - mean) / std
   ```
   - Mean, std, variance
   - Normalization, scaling
   - Correlation, covariance

3. **Integration**
   ```python
   # Pandas DataFrame → NumPy array
   X = df['text'].values
   
   # NumPy array → PyTorch tensor
   tensor = torch.from_numpy(X)
   ```
   - Seamless integration với Pandas, PyTorch
   - Memory-efficient conversions
   - Backend for scikit-learn

**Performance**:
- 10-100x faster than Python lists
- Optimized C/Fortran code
- BLAS/LAPACK support

---

## 📈 Visualization

### **Matplotlib 3.8+**

**Vai trò chính**: Static Plotting

#### Chức năng sử dụng:

1. **Basic Plots**
   ```python
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(10, 6))
   plt.bar(models, accuracies)
   plt.title('Model Comparison')
   plt.xlabel('Model')
   plt.ylabel('Accuracy')
   plt.savefig('comparison.png', dpi=300)
   ```
   - Bar charts, line plots
   - High-resolution exports (300 DPI)
   - Publication-quality figures

2. **Confusion Matrix**
   ```python
   from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
   
   cm = confusion_matrix(y_true, y_pred)
   disp = ConfusionMatrixDisplay(cm, display_labels=['REAL', 'FAKE'])
   disp.plot()
   ```
   - Heatmap visualization
   - True/False positives/negatives
   - Model performance analysis

3. **Customization**
   ```python
   plt.style.use('seaborn-v0_8')
   plt.rcParams['font.size'] = 12
   plt.grid(alpha=0.3)
   ```
   - Themes, styles
   - Font, colors, grid
   - Legend, annotations

---

### **Seaborn 0.13+**

**Vai trò chính**: Statistical Visualization

#### Chức năng sử dụng:

1. **Distribution Plots**
   ```python
   import seaborn as sns
   
   sns.histplot(df['text_length'], bins=50, kde=True)
   sns.boxplot(data=df, x='label', y='text_length')
   ```
   - Histograms với KDE
   - Box plots, violin plots
   - Distribution analysis

2. **Correlation Heatmap**
   ```python
   corr = df.corr()
   sns.heatmap(corr, annot=True, cmap='coolwarm')
   ```
   - Feature correlations
   - Color-coded matrix
   - Annotations

3. **Pair Plots**
   ```python
   sns.pairplot(df, hue='label')
   ```
   - Multi-variable relationships
   - Scatter matrix
   - Class separation visualization

**Built on Matplotlib**: Seaborn = Matplotlib + better defaults + statistical functions

---

## 🔧 Additional Libraries

### **Scipy 1.11+**
- **Chức năng**: Sparse matrices (TF-IDF), scientific computing
- **Sử dụng**: `scipy.sparse.hstack` để combine features

### **Joblib 1.3+**
- **Chức năng**: Model serialization
- **Sử dụng**: Save/load scikit-learn models (compressed)

### **Requests 2.31+**
- **Chức năng**: HTTP requests
- **Sử dụng**: Crawl static websites (fallback cho Playwright)

### **Uvicorn 0.25+**
- **Chức năng**: ASGI server
- **Sử dụng**: Run FastAPI application

---

## 📦 Dependencies Summary

```txt
# Core ML/DL
scikit-learn==1.3.2
torch==2.4.0
transformers==4.36.0

# Web
streamlit==1.29.0
fastapi==0.115.0
uvicorn==0.25.0

# Crawling
playwright==1.40.0
beautifulsoup4==4.12.2

# Data
pandas==2.1.4
numpy==1.26.2

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# NLP
nltk==3.8.1

# Utilities
joblib==1.3.2
scipy==1.11.4
requests==2.31.0
```

---

## 🎯 Tech Stack Decision Matrix

| Requirement | Solution | Why? |
|-------------|----------|------|
| **Baseline ML** | Scikit-learn | Industry standard, easy to use, fast training |
| **Deep Learning** | PyTorch | Flexibility, research-friendly, strong community |
| **Pre-trained Models** | HuggingFace | Best model hub, easy fine-tuning, 98.9% accuracy |
| **Web UI** | Streamlit | Rapid prototyping, no frontend code, beautiful UI |
| **API** | FastAPI | Fast, async, auto docs, type-safe |
| **Dynamic Crawling** | Playwright | Handles JS, modern sites, reliable |
| **Static Crawling** | BeautifulSoup | Fast, lightweight, simple |
| **Data Processing** | Pandas | Standard for tabular data, rich functionality |

---

## 🚀 Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| **Baseline Training** | Time | 2 minutes (CPU) |
| **LSTM Training** | Time | 15 minutes (T4 GPU) |
| **BERT Fine-tuning** | Time | 30 minutes (T4 GPU) |
| **Inference (Baseline)** | Latency | 1-2ms per article |
| **Inference (LSTM)** | Latency | 10-20ms per article |
| **Inference (BERT)** | Latency | 50ms per article (GPU) |
| **Web Crawling** | Speed | 2-5s per URL |
| **Streamlit** | Load time | <1s (với cached models) |
| **API** | Throughput | 100+ requests/second |

---

## 📚 Further Reading

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [HuggingFace Course](https://huggingface.co/course)
- [Streamlit Docs](https://docs.streamlit.io/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)

---

**Last Updated**: 2026-06-22  
**Maintained By**: Development Team
