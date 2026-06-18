"""
Model Comparison Utilities
Provides functions to compare predictions from multiple models
"""

import numpy as np


def display_comparison(text, all_results):
    """
    Display comparison table for all model predictions
    
    Args:
        text: The input text that was predicted
        all_results: Dict of model_name -> prediction_result
    """
    print("\n" + "="*80)
    print("PREDICTION RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n📰 Input Text (cleaned):")
    print(f"   {text[:200]}..." if len(text) > 200 else f"   {text}")
    
    print("\n" + "="*80)
    print("📊 PREDICTIONS:")
    print("="*80)
    print(f"{'Model':<25} {'Prediction':<12} {'Confidence':<12} {'FAKE %':<12} {'REAL %':<12} {'Time (ms)':<12}")
    print("-"*80)
    
    for model_name, result in all_results.items():
        print(f"{model_name:<25} "
              f"{result['label']:<12} "
              f"{result['confidence']*100:>10.2f}% "
              f"{result['fake_prob']*100:>10.2f}% "
              f"{result['real_prob']*100:>10.2f}% "
              f"{result['inference_time']:>10.3f}")
    
    print("\n" + "="*80)
    print("📈 ANALYSIS:")
    print("="*80)
    
    # Consensus
    predictions = [r["label"] for r in all_results.values()]
    fake_count = predictions.count("FAKE")
    real_count = predictions.count("REAL")
    
    print(f"   Consensus: {fake_count} FAKE, {real_count} REAL")
    
    if fake_count == len(predictions):
        print("   ✅ All models agree: FAKE NEWS")
    elif real_count == len(predictions):
        print("   ✅ All models agree: REAL NEWS")
    else:
        print("   ⚠️  Models disagree!")
    
    # Average confidence
    avg_confidence = np.mean([r["confidence"] for r in all_results.values()])
    print(f"   Average confidence: {avg_confidence*100:.2f}%")
    
    # Fastest model
    fastest = min(all_results.items(), key=lambda x: x[1]["inference_time"])
    print(f"   Fastest model: {fastest[0]} ({fastest[1]['inference_time']:.3f}ms)")
    
    # Most confident
    most_confident = max(all_results.items(), key=lambda x: x[1]["confidence"])
    print(f"   Most confident: {most_confident[0]} ({most_confident[1]['confidence']*100:.2f}%)")


def get_consensus(all_results):
    """
    Get consensus prediction from multiple models
    
    Args:
        all_results: Dict of model_name -> prediction_result
    
    Returns:
        dict: Consensus analysis
    """
    predictions = [r["label"] for r in all_results.values()]
    fake_count = predictions.count("FAKE")
    real_count = predictions.count("REAL")
    
    # Majority vote
    if fake_count > real_count:
        consensus_label = "FAKE"
        agreement_pct = (fake_count / len(predictions)) * 100
    elif real_count > fake_count:
        consensus_label = "REAL"
        agreement_pct = (real_count / len(predictions)) * 100
    else:
        consensus_label = "UNCLEAR"
        agreement_pct = 50.0
    
    # Average confidence of models that agree with consensus
    agreeing_confidences = [
        r["confidence"] for r in all_results.values() 
        if r["label"] == consensus_label
    ]
    avg_confidence = np.mean(agreeing_confidences) if agreeing_confidences else 0.5
    
    return {
        "label": consensus_label,
        "confidence": avg_confidence,
        "agreement_pct": agreement_pct,
        "votes": {
            "fake": fake_count,
            "real": real_count
        }
    }


def get_fastest_model(all_results):
    """
    Find the fastest model
    
    Args:
        all_results: Dict of model_name -> prediction_result
    
    Returns:
        tuple: (model_name, inference_time)
    """
    fastest = min(all_results.items(), key=lambda x: x[1]["inference_time"])
    return fastest[0], fastest[1]["inference_time"]


def get_most_confident(all_results):
    """
    Find the most confident model
    
    Args:
        all_results: Dict of model_name -> prediction_result
    
    Returns:
        tuple: (model_name, confidence, label)
    """
    most_confident = max(all_results.items(), key=lambda x: x[1]["confidence"])
    return (
        most_confident[0],
        most_confident[1]["confidence"],
        most_confident[1]["label"]
    )


def analyze_results(all_results):
    """
    Comprehensive analysis of all model predictions
    
    Args:
        all_results: Dict of model_name -> prediction_result
    
    Returns:
        dict: Complete analysis including consensus, stats, etc.
    """
    consensus = get_consensus(all_results)
    fastest_model, fastest_time = get_fastest_model(all_results)
    most_conf_model, most_conf_val, most_conf_label = get_most_confident(all_results)
    
    # Calculate statistics
    confidences = [r["confidence"] for r in all_results.values()]
    fake_probs = [r["fake_prob"] for r in all_results.values()]
    real_probs = [r["real_prob"] for r in all_results.values()]
    times = [r["inference_time"] for r in all_results.values()]
    
    return {
        "consensus": consensus,
        "fastest": {
            "model": fastest_model,
            "time": fastest_time
        },
        "most_confident": {
            "model": most_conf_model,
            "confidence": most_conf_val,
            "label": most_conf_label
        },
        "statistics": {
            "avg_confidence": np.mean(confidences),
            "avg_fake_prob": np.mean(fake_probs),
            "avg_real_prob": np.mean(real_probs),
            "total_time": np.sum(times),
            "avg_time": np.mean(times)
        }
    }
