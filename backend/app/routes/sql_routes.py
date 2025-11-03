from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re
from typing import List, Dict

router = APIRouter()

class SQLQuery(BaseModel):
    query: str

class SQLResponse(BaseModel):
    is_malicious: bool
    confidence: float
    message: str
    detected_patterns: List[str] = []

# SQL Injection patterns for detection
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

def detect_sql_injection(query: str) -> Dict:
    detected_patterns = []
    confidence = 0.0
    
    normalized_query = query.upper().replace(' ', '')
    
    for category, patterns in SQL_INJECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                detected_patterns.append(f"{category}: {pattern}")
                confidence += 0.15
    
    if "OR" in normalized_query and "=" in normalized_query:
        if "'1'='1'" in normalized_query or "1=1" in normalized_query:
            confidence += 0.3
    
    if "UNION" in normalized_query and "SELECT" in normalized_query:
        confidence += 0.25
    
    if "--" in query or "/*" in query:
        confidence += 0.2
    
    confidence = min(confidence, 0.95)
    is_malicious = len(detected_patterns) > 0
    
    return {
        "is_malicious": is_malicious,
        "confidence": round(confidence * 100, 2),
        "detected_patterns": detected_patterns,
        "message": "Malicious SQL injection detected" if is_malicious else "Query appears to be safe"
    }

@router.post("/check-sql", response_model=SQLResponse)
async def check_sql_injection(sql_query: SQLQuery):
    try:
        result = detect_sql_injection(sql_query.query)
        return SQLResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")