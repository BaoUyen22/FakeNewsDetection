"""
=============================================================================
COMPARE ALL MODELS - INTERACTIVE TESTING
=============================================================================
Test a single news article with all available models and compare results
- Baseline: Logistic Regression, LinearSVC
- Deep Learning: LSTM + Word Embedding
- Transformer: BERT (DistilBERT)

Compare based on:
- Prediction result (REAL/FAKE)
- Confidence score
- Inference time
=============================================================================
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from predictors import BaselinePredictor, DeepLearningPredictor, BERTPredictor
from predictors.comparison import display_comparison

# =============================================================================
# MAIN INTERACTIVE TESTING
# =============================================================================

def main():
    """Main interactive testing loop"""
    print("="*80)
    print("FAKE NEWS DETECTION - MODEL COMPARISON")
    print("="*80)
    print("\nLoading models...")
    
    # Load all models
    baseline = BaselinePredictor()
    deep_learning = DeepLearningPredictor()
    bert = BERTPredictor()
    
    print("\n" + "="*80)
    print("READY FOR TESTING")
    print("="*80)
    print("\n📝 Instructions:")
    print("   1. Enter a news article (title + text)")
    print("   2. Each model will apply its own preprocessing")
    print("   3. All models will predict and results will be compared")
    print("   4. Type 'exit' to quit")
    print("\n" + "="*80)
    
    while True:
        print("\n" + "-"*80)
        user_input = input("\n📰 Enter news article (or 'exit' to quit):\n> ")
        
        if user_input.lower().strip() == "exit":
            print("\n👋 Goodbye!")
            break
        
        if not user_input.strip():
            print("⚠️  Please enter some text!")
            continue
        
        # Use raw text - each predictor will apply its own preprocessing
        raw_text = user_input
        
        # Collect predictions from all models
        all_results = {}
        
        # Baseline models (will apply MLTextPreprocessor internally)
        baseline_results = baseline.predict(raw_text)
        all_results.update(baseline_results)
        
        # Deep Learning (will apply LSTMTextPreprocessor internally)
        dl_results = deep_learning.predict(raw_text)
        all_results.update(dl_results)
        
        # BERT (will apply BERTTextPreprocessor internally)
        bert_results = bert.predict(raw_text)
        all_results.update(bert_results)
        
        # Display comparison
        if all_results:
            display_comparison(raw_text, all_results)
        else:
            print("❌ No models available for prediction!")


if __name__ == "__main__":
    main()
