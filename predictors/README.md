# Model Predictors Package

Thư mục này chứa các predictor class cho tất cả models trong project.

## Structure

```
models/predictors/
├── __init__.py          # Package exports
├── baseline.py          # Baseline models (Logistic Regression, LinearSVC)
├── deep_learning.py     # LSTM model
├── bert.py              # BERT (DistilBERT) model
└── comparison.py        # Comparison utilities
```

## Usage

### Basic Usage

```python
from models.predictors import BaselinePredictor, DeepLearningPredictor, BERTPredictor
from models.predictors.comparison import display_comparison
from utils.preprocessor import clean_text

# Load models
baseline = BaselinePredictor()
deep_learning = DeepLearningPredictor()
bert = BERTPredictor()

# Prepare text
text = "Your news article here"
cleaned_text = clean_text(text)

# Get predictions
baseline_results = baseline.predict(cleaned_text)
dl_results = deep_learning.predict(cleaned_text)
bert_results = bert.predict(cleaned_text)

# Combine all results
all_results = {}
all_results.update(baseline_results)
all_results.update(dl_results)
all_results.update(bert_results)

# Display comparison
display_comparison(cleaned_text, all_results)
```

### Individual Model Usage

#### Baseline Models

```python
from models.predictors import BaselinePredictor

# Initialize with custom threshold
predictor = BaselinePredictor(fake_threshold=0.70)

# Predict
results = predictor.predict(cleaned_text)

# Results format:
# {
#     "Logistic Regression": {
#         "label": "FAKE" or "REAL",
#         "confidence": 0.85,
#         "fake_prob": 0.85,
#         "real_prob": 0.15,
#         "inference_time": 5.2  # ms
#     },
#     "LinearSVC": { ... }
# }
```

#### Deep Learning (LSTM)

```python
from models.predictors import DeepLearningPredictor

predictor = DeepLearningPredictor()
results = predictor.predict(cleaned_text)

# Results format:
# {
#     "LSTM": {
#         "label": "FAKE" or "REAL",
#         "confidence": 0.92,
#         "fake_prob": 0.92,
#         "real_prob": 0.08,
#         "inference_time": 15.3  # ms
#     }
# }
```

#### BERT

```python
from models.predictors import BERTPredictor

predictor = BERTPredictor()
results = predictor.predict(cleaned_text)

# Results format:
# {
#     "BERT (DistilBERT)": {
#         "label": "FAKE" or "REAL",
#         "confidence": 0.95,
#         "fake_prob": 0.95,
#         "real_prob": 0.05,
#         "inference_time": 120.5  # ms
#     }
# }
```

### Comparison Utilities

```python
from models.predictors.comparison import (
    display_comparison,
    get_consensus,
    get_fastest_model,
    get_most_confident,
    analyze_results
)

# Get consensus from all models
consensus = get_consensus(all_results)
print(f"Consensus: {consensus['label']}")
print(f"Agreement: {consensus['agreement_pct']:.1f}%")

# Find fastest model
fastest_name, fastest_time = get_fastest_model(all_results)
print(f"Fastest: {fastest_name} ({fastest_time:.2f}ms)")

# Find most confident model
conf_name, conf_value, conf_label = get_most_confident(all_results)
print(f"Most confident: {conf_name} ({conf_value:.2f})")

# Comprehensive analysis
analysis = analyze_results(all_results)
print(f"Consensus: {analysis['consensus']}")
print(f"Statistics: {analysis['statistics']}")
```

## Model Details

### BaselinePredictor

**Models:**
- Logistic Regression
- LinearSVC

**Features:**
- TF-IDF vectorization
- Custom fake threshold (default 0.70)
- Probability calibration for SVC

**Size:** ~10 MB (models + vectorizer)
**Device:** CPU only
**Speed:** ~5-10 ms per prediction

### DeepLearningPredictor

**Model:**
- LSTM with Word Embeddings

**Features:**
- Custom vocabulary
- 200-token sequence length
- Dropout for regularization

**Size:** ~15 MB
**Device:** CPU/GPU (auto-detect)
**Speed:** ~10-20 ms per prediction

### BERTPredictor

**Model:**
- DistilBERT (distilbert-base-uncased)

**Features:**
- 512 token max length
- Transformer architecture
- Pre-trained language understanding

**Size:** ~256 MB
**Device:** CPU/GPU (auto-detect)
**Speed:** ~100-200 ms per prediction (CPU), ~20-50 ms (GPU)

## Comparison Features

### display_comparison()

Hiển thị bảng so sánh đẹp với:
- Predictions từ tất cả models
- Confidence scores
- FAKE/REAL probabilities
- Inference times
- Consensus analysis
- Statistics

### get_consensus()

Trả về kết quả consensus từ multiple models:
- Majority vote
- Agreement percentage
- Average confidence

### analyze_results()

Phân tích toàn diện:
- Consensus
- Fastest model
- Most confident model
- Statistics (avg confidence, avg time, etc.)

## Integration

### With Streamlit App

```python
# In app.py or components/model_loader.py
import streamlit as st
from models.predictors import BaselinePredictor, DeepLearningPredictor, BERTPredictor

@st.cache_resource
def load_all_predictors():
    baseline = BaselinePredictor(fake_threshold=0.60)
    deep_learning = DeepLearningPredictor()
    bert = BERTPredictor()
    return baseline, deep_learning, bert
```

### With Interactive Testing

```python
# In test/compare_all_models.py
from models.predictors import BaselinePredictor, DeepLearningPredictor, BERTPredictor
from models.predictors.comparison import display_comparison
from utils.preprocessor import clean_text

# Main loop
while True:
    user_input = input("Enter text: ")
    cleaned = clean_text(user_input)
    
    # Predict with all models
    all_results = {}
    all_results.update(baseline.predict(cleaned))
    all_results.update(deep_learning.predict(cleaned))
    all_results.update(bert.predict(cleaned))
    
    # Display
    display_comparison(cleaned, all_results)
```

## Error Handling

Tất cả predictors đều handle errors gracefully:

```python
# If model fails to load
predictor = BaselinePredictor()
# Prints: "❌ Error loading baseline models: ..."
# predictor.models will be empty dict

# If prediction fails
results = predictor.predict(text)
# Returns empty dict {} instead of crashing
```

## Testing

```bash
# Test interactive comparison
python test/compare_all_models.py

# Test quick demo
python test/quick_demo.py

# Test in Streamlit app
streamlit run app.py
```

## Benefits of This Architecture

1. **Separation of Concerns** - Each model in its own file
2. **Reusability** - Can import any predictor independently
3. **Testability** - Easy to test each component
4. **Maintainability** - Clear structure, easy to modify
5. **Extensibility** - Easy to add new models

## Adding a New Model

1. Create `models/predictors/new_model.py`
2. Implement predictor class with `predict()` method
3. Export in `__init__.py`
4. Use in comparison scripts

Example:

```python
# models/predictors/new_model.py
class NewModelPredictor:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        # Load your model
        pass
    
    def predict(self, text):
        # Return same format as other predictors
        return {
            "New Model": {
                "label": "FAKE" or "REAL",
                "confidence": 0.0 to 1.0,
                "fake_prob": 0.0 to 1.0,
                "real_prob": 0.0 to 1.0,
                "inference_time": float  # ms
            }
        }

# models/predictors/__init__.py
from .new_model import NewModelPredictor
```

## Notes

- All models expect **cleaned text** from `utils.preprocessor.clean_text()`
- Prediction format is standardized across all models
- Inference times are in milliseconds
- Confidences and probabilities are floats between 0 and 1
- Labels are strings: "FAKE" or "REAL"
