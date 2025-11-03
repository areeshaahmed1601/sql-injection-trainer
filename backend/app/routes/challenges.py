from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from app.config import ml_detector, db_manager

router = APIRouter()

class ChallengeSubmission(BaseModel):
    challenge_id: int
    query: str
    user_id: Optional[int] = 1

@router.post("/submit-challenge")
async def submit_challenge(submission: ChallengeSubmission, background_tasks: BackgroundTasks):
    """Submit a challenge solution"""
    try:
        is_malicious, confidence, patterns = await ml_detector.predict(submission.query)
        
        # Challenge-specific validation
        challenge_solutions = {
            1: ["or '1'='1'", "or 1=1"],  # Login Bypass
            2: ["union select"],           # Data Extraction
            3: ["information_schema"],     # Database Schema
            4: ["sleep(", "waitfor"]       # Blind Injection
        }
        
        required_patterns = challenge_solutions.get(submission.challenge_id, [])
        solution_found = any(pattern in submission.query.lower() for pattern in required_patterns)
        completed = solution_found and is_malicious
        
        # Calculate score based on challenge difficulty
        base_scores = {1: 50, 2: 75, 3: 100, 4: 150}
        score = base_scores.get(submission.challenge_id, 50) if completed else 0
        
        # Log submission in background
        background_tasks.add_task(
            db_manager.log_challenge_submission,
            submission.user_id,
            submission.challenge_id,
            submission.query,
            completed,
            score,
            patterns
        )
        
        return {
            "challenge_id": submission.challenge_id,
            "is_malicious": is_malicious,
            "completed": completed,
            "score": score,
            "confidence": confidence,
            "message": "🎉 Challenge completed successfully!" if completed else "❌ Keep trying! Your solution needs to match the challenge requirements.",
            "detected_patterns": patterns,
            "required_patterns": required_patterns
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Challenge error: {str(e)}")

@router.get("/challenge-info")
async def get_challenge_info():
    """Get information about available challenges"""
    challenges = {
        1: {
            "title": "Login Bypass",
            "description": "Bypass authentication using SQL injection",
            "difficulty": "Easy",
            "points": 50,
            "hint": "Use OR conditions to make the WHERE clause always true"
        },
        2: {
            "title": "Data Extraction", 
            "description": "Extract data using UNION attacks",
            "difficulty": "Medium",
            "points": 75,
            "hint": "Use UNION to combine queries"
        },
        3: {
            "title": "Database Schema",
            "description": "Discover database structure",
            "difficulty": "Hard", 
            "points": 100,
            "hint": "Query information_schema tables"
        },
        4: {
            "title": "Blind Injection",
            "description": "Use time-based blind SQL injection",
            "difficulty": "Expert",
            "points": 150,
            "hint": "Use time delays to detect table existence"
        }
    }
    return challenges