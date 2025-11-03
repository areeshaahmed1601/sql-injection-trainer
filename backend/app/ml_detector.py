import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import os
from typing import Dict, List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

class SQLInjectionMLDetector:
    def __init__(self):
        self.model = None
        self.rule_patterns = self._initialize_patterns()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.model_trained = False
        # Try combined dataset first, fall back to current dataset
        self.dataset_path = "datasets/combined_sql_dataset.csv"
        if not os.path.exists(self.dataset_path):
            self.dataset_path = "datasets/complete_sql_dataset.csv"
        
    def _initialize_patterns(self) -> Dict:
        return {
            'tautology': [
                r"OR\s*['\"\d]\s*=\s*['\"\d]",
                r"1\s*=\s*1",
                r"['\"\d]\s*=\s*['\"\d]",
                r"OR\s+'1'='1'"
            ],
            'union_attack': [
                r"UNION\s+ALL\s+SELECT",
                r"UNION\s+SELECT",
                r"UNION.*SELECT"
            ],
            'comment_attack': [
                r"--",
                r"#",
                r"\/\*"
            ],
            'stacked_queries': [
                r";\s*(DROP|DELETE|UPDATE|INSERT)",
                r";\s*EXEC"
            ],
            'time_delay': [
                r"SLEEP\s*\(",
                r"WAITFOR\s+DELAY"
            ]
        }
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset"""
        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Loaded dataset: {len(df)} queries")
            return df
        else:
            return self.create_basic_dataset()
    
    def create_basic_dataset(self) -> pd.DataFrame:
        """Create a basic dataset for training"""
        print("📝 Creating training dataset...")
        
        # Simple dataset with clear patterns
        data = {
            'query': [
                # Clear benign queries
                "SELECT * FROM users WHERE username = 'john'",
                "INSERT INTO products (name, price) VALUES ('laptop', 1000)",
                "UPDATE users SET last_login = NOW() WHERE id = 1",
                "DELETE FROM sessions WHERE expires_at < NOW()",
                "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
                
                # Clear malicious queries
                "admin' OR '1'='1'",
                "1 UNION SELECT username, password FROM users", 
                "1; DROP TABLE users --",
                "1 AND SLEEP(5) --",
                "admin' OR 1=1 --",
            ],
            'label': ['benign'] * 5 + ['malicious'] * 5
        }
        
        df = pd.DataFrame(data)
        print(f"✅ Created basic dataset with {len(df)} samples")
        return df
    
    def extract_features(self, query: str) -> List[float]:
        """Extract comprehensive features from SQL query"""
        features = []
        query_lower = query.lower()
        
        # 1. Basic statistical features
        features.append(len(query))  # length
        features.append(query.count(' '))  # num_spaces
        features.append(len(re.findall(r'[^\w\s]', query)))  # num_special_chars
        features.append(len(re.findall(r'\b(select|insert|update|delete|drop|union|or|and|where)\b', query_lower)))  # num_keywords
        
        # 2. Structural features
        features.append(1 if 'union' in query_lower else 0)
        features.append(1 if 'select' in query_lower else 0)
        features.append(1 if ' or ' in query_lower else 0)
        features.append(1 if '--' in query or '/*' in query or '#' in query else 0)
        features.append(1 if ';' in query else 0)
        features.append(1 if '=' in query else 0)
        features.append(1 if "'" in query or '"' in query else 0)
        features.append(1 if "1=1" in query or "'1'='1'" in query else 0)
        
        # 3. Pattern-based features
        for category, patterns in self.rule_patterns.items():
            pattern_found = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    pattern_found = 1
                    break
            features.append(pattern_found)
        
        # 4. Additional heuristic features
        features.append(query.count("'"))  # single quote count
        features.append(query.count('"'))  # double quote count
        features.append(1 if 'admin' in query_lower else 0)  # mentions admin
        features.append(1 if 'password' in query_lower else 0)  # mentions password
        features.append(1 if 'information_schema' in query_lower else 0)  # system tables
        
        return features
    
    async def train_model(self):
        """Train the ML model with better parameters"""
        print("🔄 Training ML model...")
        
        try:
            # Load dataset
            df = self.load_dataset()
            
            # For very large datasets, we might want to sample for faster training
            if len(df) > 10000:
                print(f"📊 Large dataset detected ({len(df)} queries), sampling for training...")
                # Sample while maintaining class balance
                benign_df = df[df['label'] == 'benign']
                malicious_df = df[df['label'] == 'malicious']
                
                # Sample up to 5000 from each class
                sample_size = min(5000, len(benign_df), len(malicious_df))
                benign_sample = benign_df.sample(n=sample_size, random_state=42)
                malicious_sample = malicious_df.sample(n=sample_size, random_state=42)
                
                df = pd.concat([benign_sample, malicious_sample])
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                print(f"📊 Using sampled dataset: {len(df)} queries")
            
            # Extract features
            X = []
            y = []
            
            print("🔍 Extracting features from queries...")
            for idx, (_, row) in enumerate(df.iterrows()):
                if idx % 1000 == 0 and idx > 0:
                    print(f"   Processed {idx}/{len(df)} queries...")
                    
                features = self.extract_features(str(row['query']))
                X.append(features)
                y.append(1 if str(row['label']).lower() in ['malicious', '1', 'true', 'attack'] else 0)
            
            if len(X) == 0:
                print("❌ No data for training")
                return False
            
            # Ensure we have both classes
            if len(set(y)) < 2:
                print("❌ Need both benign and malicious samples for training")
                return False
            
            # Convert to numpy arrays
            X_array = np.array(X, dtype=np.float64)
            y_array = np.array(y, dtype=np.int32)
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
            )
            
            print(f"🎯 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
            print(f"📊 Class distribution - Benign: {sum(y_train == 0)}, Malicious: {sum(y_train == 1)}")
            
            # Train model with better parameters for large dataset
            self.model = RandomForestClassifier(
                n_estimators=150,  # More trees for better performance
                max_depth=20,      # Deeper trees for complex patterns
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,        # Use all CPU cores
                verbose=1         # Show training progress
            )
            
            print("🏋️ Training Random Forest model...")
            # Train in thread pool
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.model.fit,
                X_train,
                y_train
            )
            
            # Calculate accuracy
            accuracy = self.model.score(X_test, y_test)
            print(f"✅ Model trained successfully! Accuracy: {accuracy:.3f}")
            
            # Show feature importance if we have a good model
            if accuracy > 0.6 and hasattr(self.model, 'feature_importances_'):
                self._show_feature_importance()
            
            # Save model
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/sql_injection_model.pkl')
            self.model_trained = True
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _show_feature_importance(self):
        """Show feature importance for analysis"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            print("🔍 Top Feature Importances:")
            feature_names = [
                'length', 'spaces', 'special_chars', 'keywords',
                'has_union', 'has_select', 'has_or', 'has_comment', 
                'has_semicolon', 'has_equals', 'has_quotes', 'has_always_true',
                'pattern_tautology', 'pattern_union', 'pattern_comment',
                'pattern_stacked', 'pattern_time',
                'single_quotes', 'double_quotes', 'has_admin', 'has_password', 'has_info_schema'
            ]
            
            importances = self.model.feature_importances_
            feature_imp = list(zip(feature_names, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            
            print("   Rank | Feature | Importance")
            print("   ---- | ------- | ----------")
            for i, (feature, importance) in enumerate(feature_imp[:10], 1):  # Show top 10
                if importance > 0.01:  # Only show meaningful features
                    print(f"   {i:2d}    | {feature:12} | {importance:.4f}")
    
    async def predict(self, query: str) -> Tuple[bool, float, List[str]]:
        """Predict if a query is malicious"""
        # Rule-based detection
        rule_score = 0.0
        detected_patterns = []
        
        for category, patterns in self.rule_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    rule_score += 0.2
                    detected_patterns.append(f"{category}: {pattern}")
                    break
        
        # ML-based detection
        ml_confidence = 0.0
        if self.model_trained and self.model:
            try:
                features = self.extract_features(query)
                features_array = np.array([features], dtype=np.float64)
                
                ml_proba = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.model.predict_proba,
                    features_array
                )
                ml_confidence = ml_proba[0][1] * 100
            except Exception as e:
                print(f"⚠️ ML prediction error: {e}")
        
        # Combine scores (favor ML when available)
        if self.model_trained:
            final_confidence = (ml_confidence * 0.8) + (rule_score * 100 * 0.2)
        else:
            final_confidence = rule_score * 100
        
        is_malicious = final_confidence > 40
        
        return is_malicious, final_confidence, detected_patterns
    
    async def get_model_info(self):
        """Get model information"""
        if not self.model_trained:
            return {"status": "Model not trained", "detection_method": "rule_based"}
        
        dataset_size = "Unknown"
        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            dataset_size = f"{len(df):,} queries"
        
        return {
            "status": "Model trained and ready",
            "model_type": "Random Forest",
            "dataset_size": dataset_size,
            "detection_method": "ML + Rule-based hybrid",
            "accuracy": "High (varies by dataset)"
        }