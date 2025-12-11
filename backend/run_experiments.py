#!/usr/bin/env python
"""
Run thesis experiments separately
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_detector import SQLInjectionMLDetector
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json

print("="*70)
print("SQL INJECTION DETECTION - THESIS EXPERIMENTS")
print("="*70)

def main():
    # Load detector
    detector = SQLInjectionMLDetector()
    
    # Load model
    model_path = "models/sql_injection_model.pkl"
    if os.path.exists(model_path):
        detector.model = joblib.load(model_path)
        detector.model_trained = True
        print(f"✅ Loaded trained model from {model_path}")
    else:
        print("⚠️  Model not found, training new one...")
        detector.train_model()
    
    print("\n1️⃣  DATASET ANALYSIS")
    print("-"*40)
    
    # Load dataset
    df = pd.read_csv("datasets/combined_sql_dataset.csv")
    print(f"📊 Dataset: {len(df)} total queries")
    print(f"   Benign: {len(df[df['label'] == 'benign'])}")
    print(f"   Malicious: {len(df[df['label'] == 'malicious'])}")
    
    print("\n2️⃣  FEATURE IMPORTANCE")
    print("-"*40)
    
    if hasattr(detector.model, 'feature_importances_'):
        importances = detector.model.feature_importances_
        feature_names = [
            'length', 'spaces', 'special_chars', 'keywords',
            'has_union', 'has_select', 'has_or', 'has_comment',
            'has_semicolon', 'has_equals', 'has_quotes', 'has_always_true',
            'pattern_tautology', 'pattern_union', 'pattern_comment',
            'pattern_stacked', 'pattern_time',
            'single_quotes', 'double_quotes', 'has_admin', 'has_password',
            'has_info_schema'
        ]
        
        print("Top 10 Most Important Features:")
        print("-"*30)
        
        feature_imp = list(zip(feature_names, importances))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_imp[:10], 1):
            print(f"{i:2d}. {feature:20} {importance:.4f}")
    
    print("\n3️⃣  QUERY DETECTION TEST")
    print("-"*40)
    
    test_cases = [
        ("SELECT * FROM users WHERE id = 1", "benign"),
        ("admin' OR '1'='1'", "malicious"),
        ("1 UNION SELECT username, password FROM users", "malicious"),
        ("UPDATE products SET price = 100 WHERE id = 1", "benign"),
        ("1; DROP TABLE users --", "malicious"),
        ("admin'/**/OR/**/'1'='1'", "malicious"),
        ("SELECT name, email FROM customers WHERE country = 'USA'", "benign"),
        ("' OR EXISTS(SELECT * FROM users) --", "malicious")
    ]
    
    results = []
    correct = 0
    
    for query, expected in test_cases:
        is_malicious, confidence, patterns = detector.predict(query)
        detected = "malicious" if is_malicious else "benign"
        
        is_correct = detected == expected
        if is_correct:
            correct += 1
            indicator = "✅"
        else:
            indicator = "❌"
        
        results.append({
            'query': query[:50] + "..." if len(query) > 50 else query,
            'expected': expected,
            'detected': detected,
            'confidence': float(confidence),
            'correct': is_correct,
            'patterns': patterns[:2] if patterns else []
        })
        
        print(f"{indicator} {query[:50]}...")
        print(f"   Expected: {expected}, Detected: {detected} ({confidence:.1f}%)")
    
    accuracy = correct / len(test_cases)
    print(f"\n📊 Test Accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})")
    
    print("\n4️⃣  PERFORMANCE METRICS")
    print("-"*40)
    
    # Sample performance test
    import time
    
    test_queries = [
        "SELECT * FROM users",
        "admin' OR '1'='1'",
        "INSERT INTO logs (message) VALUES ('test')",
        "1 UNION SELECT 1,2,3"
    ]
    
    times = []
    for query in test_queries:
        start = time.time()
        detector.predict(query)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    print(f"⚡ Average detection time: {avg_time:.2f} ms")
    print(f"📈 Max time: {max(times):.2f} ms, Min time: {min(times):.2f} ms")
    
    # Save results
    os.makedirs("experiments", exist_ok=True)
    
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_stats': {
            'total_queries': len(df),
            'benign': len(df[df['label'] == 'benign']),
            'malicious': len(df[df['label'] == 'malicious'])
        },
        'feature_importance': [
            {'feature': f, 'importance': float(i)} 
            for f, i in feature_imp[:10]
        ],
        'detection_test': {
            'accuracy': accuracy,
            'total_tests': len(test_cases),
            'correct': correct,
            'details': results
        },
        'performance': {
            'avg_detection_time_ms': avg_time,
            'min_time_ms': min(times),
            'max_time_ms': max(times)
        }
    }
    
    with open('experiments/thesis_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to: experiments/thesis_results.json")
    
    print("\n" + "="*70)
    print("✅ EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()