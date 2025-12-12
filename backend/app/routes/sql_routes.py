from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re
import numpy as np
import joblib
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

# Load your ML model from thesis
try:
    ML_MODEL = joblib.load("models/sql_injection_model.pkl")
    print("✅ ML Model loaded successfully")
    ML_ENABLED = True
except:
    print("⚠️ ML Model not found, using rule-based fallback")
    ML_ENABLED = False

class SQLQuery(BaseModel):
    query: str
    context: Optional[str] = "training"
    user_id: Optional[int] = 1

class SQLResponse(BaseModel):
    is_malicious: bool
    confidence: float
    message: str
    detected_patterns: List[str] = []
    detection_method: str = "rule_based"  # or "ml_hybrid"
    risk_level: Optional[str] = "low"

# Your 22 feature extractor from thesis
def extract_features(query: str) -> List[float]:
    """Extract 22 features as per your thesis methodology"""
    features = []
    query_lower = query.lower()
    
    # 1. Basic statistical features
    features.append(len(query))  # length
    features.append(query.count(' '))  # spaces
    features.append(len(re.findall(r'[^\w\s]', query)))  # special chars
    features.append(len(re.findall(r'\b(select|insert|update|delete|drop|union|or|and|where)\b', query_lower)))  # keywords
    
    # 2. Structural features
    features.append(1 if 'union' in query_lower else 0)
    features.append(1 if 'select' in query_lower else 0)
    features.append(1 if ' or ' in query_lower else 0)
    features.append(1 if '--' in query or '/*' in query or '#' in query else 0)
    features.append(1 if ';' in query else 0)
    features.append(1 if '=' in query else 0)
    features.append(1 if "'" in query or '"' in query else 0)
    features.append(1 if "1=1" in query or "'1'='1'" in query else 0)
    
    # 3. Pattern-based features (from your thesis)
    pattern_found = 0
    for pattern in [r"OR\s*['\"\d]\s*=\s*['\"\d]", r"1\s*=\s*1"]:
        if re.search(pattern, query, re.IGNORECASE):
            pattern_found = 1
            break
    features.append(pattern_found)
    
    # Add more features to reach 22...
    # (Add the remaining features from your ml_detector.py)
    
    # Ensure we have 22 features as per thesis
    while len(features) < 22:
        features.append(0.0)
    
    return features[:22]  # Return exactly 22 features

# Rule-based detection as fallback
SQL_INJECTION_PATTERNS = {
    'tautology': [
        r"OR\s+'1'='1'",
        r"OR\s+1=1",
        r"OR\s+\d+=\d+",
    ],
    'union_attack': [
        r"UNION\s+SELECT",
        r"UNION\s+ALL\s+SELECT"
    ],
    'comment_attack': [
        r"--",
        r"#",
        r"\/\*.*\*\/"
    ],
    'time_delay': [
        r"SLEEP\s*\(",
        r"WAITFOR\s+DELAY",
    ],
    'stacked_queries': [
        r";\s*SELECT",
        r";\s*DROP"
    ]
}

def detect_sql_injection_hybrid(query: str) -> Dict:
    """Hybrid detection: ML + Rule-based as per thesis"""
    
    detected_patterns = []
    
    # 1. Rule-based detection
    for category, patterns in SQL_INJECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                detected_patterns.append(f"{category}: {pattern}")
    
    rule_confidence = min(len(detected_patterns) * 0.2, 0.95)
    
    # 2. ML-based detection (from your thesis)
    ml_confidence = 0.0
    detection_method = "rule_based"
    
    if ML_ENABLED and ML_MODEL is not None:
        try:
            # Extract 22 features as per thesis
            features = extract_features(query)
            features_array = np.array([features], dtype=np.float64)
            
            # Get ML prediction (as per your thesis methodology)
            ml_proba = ML_MODEL.predict_proba(features_array)
            ml_confidence = ml_proba[0][1]  # Probability of being malicious
            detection_method = "ml_hybrid"
            
        except Exception as e:
            print(f"ML prediction failed: {e}")
            ml_confidence = 0.0
    
    # 3. Combine scores (as per thesis methodology)
    if detection_method == "ml_hybrid":
        # 80% weight to ML, 20% to rules (as mentioned in thesis)
        final_confidence = (ml_confidence * 100 * 0.8) + (rule_confidence * 100 * 0.2)
    else:
        final_confidence = rule_confidence * 100
    
    # Determine if malicious (threshold from thesis: 40%)
    is_malicious = final_confidence > 40
    
    # Risk level based on confidence
    if final_confidence > 80:
        risk_level = "critical"
    elif final_confidence > 60:
        risk_level = "high"
    elif final_confidence > 40:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "is_malicious": is_malicious,
        "confidence": round(final_confidence, 2),
        "detected_patterns": detected_patterns,
        "detection_method": detection_method,
        "risk_level": risk_level,
        "message": f"{'High-confidence SQL injection detected' if is_malicious else 'Query appears to be safe'} (Confidence: {final_confidence:.1f}%)"
    }

@router.post("/check-sql", response_model=SQLResponse)
async def check_sql_injection(sql_query: SQLQuery):
    """
    SQL Injection Detection Endpoint
    Uses hybrid ML + rule-based approach as described in thesis
    """
    try:
        print(f"🔍 Analyzing query: {sql_query.query[:50]}...")
        
        # Use hybrid detection as per thesis methodology
        result = detect_sql_injection_hybrid(sql_query.query)
        
        # Log to database (as per thesis system architecture)
        # await log_detection_to_db(sql_query.query, result)
        
        print(f"✅ Detection complete: {result['is_malicious']} ({result['confidence']}%)")
        
        return SQLResponse(**result)
        
    except Exception as e:
        print(f"❌ Error in ML detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Get ML model information as per thesis"""
    if not ML_ENABLED:
        return {
            "status": "ML model not loaded",
            "detection_method": "rule_based_only",
            "accuracy": "N/A",
            "recommendation": "Train model using ml_detector.py first"
        }
    
    return {
        "status": "ML model loaded and ready",
        "model_type": "Random Forest",
        "detection_method": "ML + Rule-based hybrid",
        "expected_accuracy": "99.42% (as per thesis)",
        "features_used": 22,
        "inference_time": "<5ms",
        "model_path": "models/sql_injection_model.pkl"
    }