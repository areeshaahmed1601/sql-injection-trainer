from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from app.config import ml_detector, db_manager

router = APIRouter()

class SQLQuery(BaseModel):
    query: str
    context: Optional[str] = "general"
    user_id: Optional[int] = 1

class DetectionResponse(BaseModel):
    is_malicious: bool
    confidence: float
    message: str
    detected_patterns: List[str]
    detection_method: str
    risk_level: str

@router.post("/detect", response_model=DetectionResponse)
async def detect_sql_injection(query: SQLQuery, background_tasks: BackgroundTasks):
    """Detect SQL injection in a query using ML"""
    try:
        is_malicious, confidence, patterns = await ml_detector.predict(query.query)
        
        # Log detection in background
        background_tasks.add_task(
            db_manager.log_detection,
            query.query,
            is_malicious,
            confidence,
            patterns,
            query.context,
            query.user_id
        )
        
        # Determine risk level
        if confidence >= 80:
            risk_level = "critical"
            message = "High-confidence SQL injection detected"
        elif confidence >= 60:
            risk_level = "high" 
            message = "SQL injection likely detected"
        elif confidence >= 40:
            risk_level = "medium"
            message = "Suspicious patterns found"
        else:
            risk_level = "low"
            message = "Query appears safe"
        
        detection_method = "ml_hybrid"
        
        return DetectionResponse(
            is_malicious=is_malicious,
            confidence=round(confidence, 2),
            message=message,
            detected_patterns=patterns,
            detection_method=detection_method,
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@router.get("/stats")
async def get_detection_stats():
    """Get comprehensive detection statistics"""
    return await db_manager.get_detection_stats()

@router.get("/user-stats/{user_id}")
async def get_user_stats(user_id: int):
    """Get user-specific statistics"""
    return await db_manager.get_user_stats(user_id)