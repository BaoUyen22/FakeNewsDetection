"""
=============================================================================
QUICK DEMO - Test with Sample News
=============================================================================
Quick demo with pre-defined sample news articles
=============================================================================
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from compare_all_models import (
    BaselinePredictor,
    DeepLearningPredictor,
    BERTPredictor,
    display_comparison
)
from utils.preprocessor import clean_text


# Sample news articles
SAMPLE_NEWS = {
    "fake_1": {
        "title": "BREAKING: Scientists Discover Chocolate Cures Cancer 100%",
        "text": "Amazing breakthrough! Researchers at fake university claim eating chocolate daily completely eliminates all cancer cells. This miracle cure will make pharmaceutical companies obsolete overnight!"
    },
    "fake_2": {
        "title": "Shocking: Moon Landing Was Filmed in Hollywood Studio",
        "text": "Leaked documents prove NASA faked the 1969 moon landing. Stanley Kubrick directed the entire thing. Wake up people! The government has been lying to us for decades!"
    },
    "real_1": {
        "title": "Federal Reserve Announces Interest Rate Decision",
        "text": "The Federal Reserve announced today that it will maintain current interest rates at 5.25-5.50 percent. The decision follows recent economic data showing inflation remains above the Fed's 2 percent target. Chair Jerome Powell emphasized the committee's commitment to returning inflation to target levels."
    },
    "real_2": {
        "title": "New Study Shows Benefits of Regular Exercise",
        "text": "A comprehensive study published in the Journal of Medicine found that individuals who engage in 150 minutes of moderate exercise per week showed improved cardiovascular health. The research, conducted over five years with 10,000 participants, provides additional evidence supporting current exercise guidelines."
    }
}


def test_sample(name, sample, predictors):
    """Test a sample news article"""
    full_text = f"{sample['title']} {sample['text']}"
    cleaned_text = clean_text(full_text)
    
    print(f"\n{'='*80}")
    print(f"TESTING: {name.upper()}")
    print(f"{'='*80}")
    print(f"\n📰 Title: {sample['title']}")
    print(f"\n📄 Text: {sample['text'][:150]}...")
    
    # Collect predictions
    all_results = {}
    
    baseline, deep_learning, bert = predictors
    
    # Baseline
    baseline_results = baseline.predict(cleaned_text)
    all_results.update(baseline_results)
    
    # Deep Learning
    dl_results = deep_learning.predict(cleaned_text)
    all_results.update(dl_results)
    
    # BERT
    bert_results = bert.predict(cleaned_text)
    all_results.update(bert_results)
    
    # Display results
    if all_results:
        display_comparison(cleaned_text, all_results)
    else:
        print("❌ No predictions available!")


def main():
    """Run quick demo"""
    print("="*80)
    print("QUICK DEMO - FAKE NEWS DETECTION")
    print("="*80)
    print("\nLoading models...")
    
    # Load all models
    baseline = BaselinePredictor()
    deep_learning = DeepLearningPredictor()
    bert = BERTPredictor()
    
    predictors = (baseline, deep_learning, bert)
    
    print("\n" + "="*80)
    print("TESTING SAMPLE NEWS ARTICLES")
    print("="*80)
    
    # Test each sample
    for name, sample in SAMPLE_NEWS.items():
        test_sample(name, sample, predictors)
        input("\n⏸️  Press Enter to continue to next sample...")
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETED!")
    print("="*80)
    print("\n💡 Tips:")
    print("   - All samples tested with all available models")
    print("   - Compare predictions to see model agreement")
    print("   - Notice speed differences between models")
    print("\n📝 Next steps:")
    print("   - Run 'python test/compare_all_models.py' for interactive testing")
    print("   - Run 'python test/model_benchmarks.py' for full evaluation")


if __name__ == "__main__":
    main()
