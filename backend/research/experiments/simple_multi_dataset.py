"""
Simple multi-dataset testing that works with your project
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SimpleMultiDatasetTest:
    def __init__(self, detector):
        self.detector = detector
        self.results_dir = Path("experiments/multi_dataset")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_test(self):
        """Run simple multi-dataset test"""
        print("\n📊 RUNNING SIMPLE MULTI-DATASET TEST")
        print("-"*40)
        
        # Load your main dataset
        main_df = pd.read_csv("datasets/combined_sql_dataset.csv")
        print(f"✅ Main dataset: {len(main_df)} queries")
        
        # Create test sets
        test_sets = {}
        
        # Test Set 1: Random split (80-20)
        test_size = int(len(main_df) * 0.2)
        test_set_1 = main_df.sample(n=test_size, random_state=42)
        test_sets['random_split'] = test_set_1
        
        # Test Set 2: Only malicious queries with variations
        malicious_queries = main_df[main_df['label'] == 'malicious'].copy()
        test_set_2 = malicious_queries.sample(min(100, len(malicious_queries)), random_state=42)
        
        # Add some benign
        benign_queries = main_df[main_df['label'] == 'benign'].sample(100, random_state=42)
        test_set_2 = pd.concat([test_set_2, benign_queries])
        test_set_2 = test_set_2.sample(frac=1, random_state=42).reset_index(drop=True)
        test_sets['malicious_focused'] = test_set_2
        
        # Test Set 3: Generated queries
        test_set_3 = self._generate_test_queries()
        test_sets['generated'] = test_set_3
        
        # Evaluate on each test set
        results = {}
        for name, test_df in test_sets.items():
            print(f"\n📈 Testing on {name} ({len(test_df)} queries):")
            
            # Extract features
            X_test, y_test = self._extract_features(test_df)
            
            # Get predictions from your detector
            correct = 0
            total = len(test_df)
            
            for i, (_, row) in enumerate(test_df.iterrows()):
                query = str(row['query'])
                expected_label = 1 if str(row['label']).lower() in ['malicious', '1', 'true'] else 0
                
                # Use your detector's predict method
                is_malicious, confidence, _ = self.detector.predict(query)
                predicted_label = 1 if is_malicious else 0
                
                if predicted_label == expected_label:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            results[name] = {
                'accuracy': accuracy,
                'samples': total,
                'benign': len(test_df[test_df['label'] == 'benign']),
                'malicious': len(test_df[test_df['label'] == 'malicious'])
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Samples: {total} ({results[name]['benign']} benign, {results[name]['malicious']} malicious)")
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _extract_features(self, df):
        """Extract features using your detector"""
        X = []
        y = []
        
        for _, row in df.iterrows():
            features = self.detector.extract_features(str(row['query']))
            X.append(features)
            y.append(1 if str(row['label']).lower() in ['malicious', '1', 'true'] else 0)
        
        return np.array(X), np.array(y)
    
    def _generate_test_queries(self):
        """Generate additional test queries"""
        from dataset_generator import SQLInjectionDatasetGenerator
        
        generator = SQLInjectionDatasetGenerator()
        generator.generate_thesis1_queries()
        
        data = []
        for query in generator.benign_queries[:50]:
            data.append({'query': query, 'label': 'benign'})
        for query in generator.malicious_queries[:50]:
            data.append({'query': query, 'label': 'malicious'})
        
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def _save_results(self, results):
        """Save results to JSON"""
        import json
        
        results_file = self.results_dir / "multi_dataset_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")