"""
Adversarial Robustness Testing Experiment
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import re

logger = logging.getLogger(__name__)

class AdversarialRobustnessTester:
    """
    Test model robustness against adversarial SQL injection attacks.
    Implements various obfuscation and evasion techniques.
    """
    
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
        self.results_dir = Path("research/results/adversarial")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define adversarial techniques
        self.techniques = {
            'case_obfuscation': self._apply_case_obfuscation,
            'comment_injection': self._apply_comment_injection,
            'whitespace_manipulation': self._apply_whitespace_manip,
            'keyword_substitution': self._apply_keyword_substitution,
            'encoding_obfuscation': self._apply_encoding_obfuscation,
            'string_concatenation': self._apply_string_concatenation,
            'null_byte_injection': self._apply_null_byte_injection,
            'multi_technique': self._apply_multi_technique
        }
    
    def generate_adversarial_samples(self, original_queries: List[str], technique: str = 'all') -> Dict[str, List[str]]:
        """Generate adversarial samples using specified techniques"""
        adversarial_samples = {}
        
        if technique == 'all':
            techniques_to_apply = self.techniques.keys()
        else:
            techniques_to_apply = [technique]
        
        for tech_name in techniques_to_apply:
            if tech_name in self.techniques:
                adversarial_samples[tech_name] = []
                for query in original_queries:
                    adversarial_query = self.techniques[tech_name](query)
                    if adversarial_query != query:  # Only add if changed
                        adversarial_samples[tech_name].append(adversarial_query)
        
        return adversarial_samples
    
    def test_robustness(self, original_queries: List[str], original_labels: List[int]) -> Dict:
        """Test model robustness against adversarial attacks"""
        logger.info("🔬 Testing adversarial robustness...")
        
        # Test on original samples
        original_accuracy = self._evaluate_accuracy(original_queries, original_labels)
        
        # Generate and test adversarial samples
        results = {'original_accuracy': original_accuracy, 'techniques': {}}
        
        for tech_name, tech_func in self.techniques.items():
            logger.info(f"  Testing {tech_name}...")
            
            adversarial_queries = []
            expected_labels = []
            
            for query, label in zip(original_queries, original_labels):
                adv_query = tech_func(query)
                adversarial_queries.append(adv_query)
                expected_labels.append(label)
            
            # Evaluate
            tech_accuracy = self._evaluate_accuracy(adversarial_queries, expected_labels)
            accuracy_drop = original_accuracy - tech_accuracy
            
            results['techniques'][tech_name] = {
                'accuracy': tech_accuracy,
                'accuracy_drop': accuracy_drop,
                'relative_robustness': tech_accuracy / original_accuracy if original_accuracy > 0 else 0,
                'samples_tested': len(adversarial_queries),
                'success_rate': self._calculate_success_rate(adversarial_queries, expected_labels)
            }
        
        # Calculate overall robustness
        results['overall_robustness'] = self._calculate_overall_robustness(results['techniques'])
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _evaluate_accuracy(self, queries: List[str], labels: List[int]) -> float:
        """Evaluate accuracy on given queries"""
        correct = 0
        total = len(queries)
        
        for query, true_label in zip(queries, labels):
            try:
                features = self.feature_extractor.extract_features(query)
                pred = self.model.predict([features])[0]
                if pred == true_label:
                    correct += 1
            except:
                continue  # Skip queries that cause errors
        
        return correct / total if total > 0 else 0
    
    # Implementation of adversarial techniques...
    def _apply_case_obfuscation(self, query: str) -> str:
        """Apply case variation obfuscation"""
        import random
        
        words = query.split()
        obfuscated = []
        
        for word in words:
            if random.random() < 0.3:  # 30% chance to obfuscate each word
                # Randomly mix case
                obfuscated_word = ''.join(
                    random.choice([c.upper(), c.lower()]) for c in word
                )
                obfuscated.append(obfuscated_word)
            else:
                obfuscated.append(word)
        
        return ' '.join(obfuscated)
    
    def _apply_comment_injection(self, query: str) -> str:
        """Inject SQL comments to break detection"""
        import random
        
        comment_types = ['--', '/* comment */', '#', '-- ']
        
        if ' ' in query:
            words = query.split()
            insert_pos = random.randint(1, len(words) - 1)
            words.insert(insert_pos, random.choice(comment_types))
            return ' '.join(words)
        
        return query
    
    # ... Add more techniques as shown in previous code
    
    def run_complete_experiment(self) -> Dict:
        """Run complete adversarial robustness experiment"""
        logger.info("="*60)
        logger.info("ADVERSARIAL ROBUSTNESS EXPERIMENT")
        logger.info("="*60)
        
        # Load test data
        test_data = self._load_test_data()
        
        # Run robustness tests
        results = self.test_robustness(
            test_data['queries'],
            test_data['labels']
        )
        
        # Generate report
        report = self._generate_report(results)
        
        logger.info("✅ Adversarial robustness experiment completed!")
        
        return results