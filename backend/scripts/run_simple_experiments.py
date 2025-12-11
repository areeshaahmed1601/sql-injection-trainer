#!/usr/bin/env python3
"""
Simple experiment runner that works with your existing project
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*70)
print("SQL INJECTION DETECTION - THESIS EXPERIMENTS")
print("="*70)

def run_simple_experiments():
    """Run simple experiments that don't require complex dependencies"""
    
    try:
        # Import your existing modules
        from ml_detector import SQLInjectionMLDetector
        import pandas as pd
        import numpy as np
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        print("\n📊 EXPERIMENT 1: BASIC MODEL PERFORMANCE")
        print("-"*40)
        
        # Load your model
        model_path = "models/sql_injection_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✅ Loaded trained model")
        else:
            print("❌ Model not found, training new one...")
            detector = SQLInjectionMLDetector()
            detector.train_model()
            model = detector.model
        
        # Load dataset
        df = pd.read_csv("datasets/combined_sql_dataset.csv")
        print(f"✅ Loaded dataset: {len(df)} queries")
        
        # Extract features
        detector = SQLInjectionMLDetector()
        X = []
        y = []
        
        print("🔍 Extracting features...")
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 1000 == 0 and i > 0:
                print(f"   Processed {i}/{len(df)}...")
            
            features = detector.extract_features(str(row['query']))
            X.append(features)
            y.append(1 if str(row['label']).lower() in ['malicious', '1', 'true'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train/test
        if not hasattr(model, 'predict'):
            print("❌ Model not properly trained")
            return
        
        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 RESULTS:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Training samples: {len(X_train)}")
        
        # Save basic results
        results = {
            'accuracy': float(accuracy),
            'test_samples': len(X_test),
            'training_samples': len(X_train),
            'model_type': 'Random Forest',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        import json
        os.makedirs('experiments', exist_ok=True)
        with open('experiments/basic_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n📊 EXPERIMENT 2: FEATURE IMPORTANCE ANALYSIS")
        print("-"*40)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Feature names (simplified)
            feature_names = [
                'length', 'spaces', 'special_chars', 'keywords',
                'has_union', 'has_select', 'has_or', 'has_comment',
                'has_semicolon', 'has_equals', 'has_quotes', 'has_always_true',
                'pattern_tautology', 'pattern_union', 'pattern_comment',
                'pattern_stacked', 'pattern_time',
                'single_quotes', 'double_quotes', 'has_admin', 'has_password',
                'has_info_schema'
            ]
            
            print("\nTop 10 Most Important Features:")
            print("-"*40)
            
            feature_imp = list(zip(feature_names, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_imp[:10], 1):
                print(f"{i:2d}. {feature:20} {importance:.4f}")
            
            # Save feature importance
            feature_results = {
                'top_features': [
                    {'feature': f, 'importance': float(imp)}
                    for f, imp in feature_imp[:10]
                ],
                'all_features': dict(zip(feature_names, [float(x) for x in importances]))
            }
            
            with open('experiments/feature_importance.json', 'w') as f:
                json.dump(feature_results, f, indent=2)
            
            print("💾 Feature importance saved to experiments/feature_importance.json")
        
        print("\n📊 EXPERIMENT 3: CROSS-VALIDATION (SIMPLIFIED)")
        print("-"*40)
        
        # Simple cross-validation
        from sklearn.model_selection import cross_val_score
        
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        print(f"\nCross-Validation Results (5-fold):")
        print(f"   Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"   Mean accuracy: {cv_scores.mean():.4f}")
        print(f"   Std deviation: {cv_scores.std():.4f}")
        
        cv_results = {
            'fold_scores': [float(s) for s in cv_scores],
            'mean_accuracy': float(cv_scores.mean()),
            'std_deviation': float(cv_scores.std())
        }
        
        with open('experiments/cross_validation.json', 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        print("\n✅ ALL EXPERIMENTS COMPLETED!")
        print("="*70)
        
        # Generate summary
        print("\n📋 EXPERIMENT SUMMARY:")
        print("-"*40)
        print(f"1. Model Accuracy: {accuracy:.4f}")
        print(f"2. Cross-Validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"3. Top Feature: {feature_imp[0][0]} ({feature_imp[0][1]:.4f})")
        
        # Save final summary
        summary = {
            'experiment_summary': {
                'final_accuracy': float(accuracy),
                'cross_val_mean': float(cv_scores.mean()),
                'cross_val_std': float(cv_scores.std()),
                'top_feature': feature_imp[0][0],
                'top_feature_importance': float(feature_imp[0][1])
            },
            'files_generated': [
                'experiments/basic_results.json',
                'experiments/feature_importance.json',
                'experiments/cross_validation.json'
            ],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open('experiments/experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n💾 All results saved to experiments/ directory")
        
    except Exception as e:
        print(f"\n❌ Error running experiments: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_simple_experiments()