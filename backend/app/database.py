import sqlite3
import json
from typing import List, Dict, Optional
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path: str = "database/detection_logs.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with comprehensive tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                level INTEGER DEFAULT 1,
                total_score INTEGER DEFAULT 0,
                challenges_completed INTEGER DEFAULT 0
            )
        ''')
        
        # Detection logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER DEFAULT 1,
                query TEXT NOT NULL,
                is_malicious BOOLEAN NOT NULL,
                confidence REAL NOT NULL,
                detected_patterns TEXT,
                context TEXT,
                detection_method TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Challenge submissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS challenge_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER DEFAULT 1,
                challenge_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                is_correct BOOLEAN NOT NULL,
                score_earned INTEGER DEFAULT 0,
                detected_patterns TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert default user
        cursor.execute('''
            INSERT OR IGNORE INTO users (id, username, email, level, total_score) 
            VALUES (1, 'demo_user', 'demo@example.com', 1, 0)
        ''')
        
        conn.commit()
        conn.close()
        
        print("✅ Database initialized successfully!")
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    async def log_detection(self, query: str, is_malicious: bool, confidence: float, patterns: List[str], context: str = "general", user_id: int = 1):
        """Log detection results"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detection_logs 
            (user_id, query, is_malicious, confidence, detected_patterns, context, detection_method)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, query, is_malicious, confidence, json.dumps(patterns), context, 'ml_hybrid'))
        
        conn.commit()
        conn.close()
    
    async def log_challenge_submission(self, user_id: int, challenge_id: int, query: str, is_correct: bool, score_earned: int, patterns: List[str]):
        """Log challenge submissions"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            print(f"📝 Logging challenge submission: User {user_id}, Challenge {challenge_id}, Correct: {is_correct}")
            
            cursor.execute('''
                INSERT INTO challenge_submissions 
                (user_id, challenge_id, query, is_correct, score_earned, detected_patterns)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, challenge_id, query, is_correct, score_earned, json.dumps(patterns)))
            
            # Update user score if correct
            if is_correct:
                cursor.execute('''
                    UPDATE users 
                    SET total_score = total_score + ?,
                        challenges_completed = challenges_completed + 1
                    WHERE id = ?
                ''', (score_earned, user_id))
                print(f"✅ Updated user score: +{score_earned} points")
            
            conn.commit()
            conn.close()
            print(f"✅ Successfully logged challenge submission")
            
        except Exception as e:
            print(f"❌ Error logging challenge submission: {e}")
            # Don't re-raise to avoid breaking the main challenge flow
    
    async def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Basic stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN is_malicious THEN 1 ELSE 0 END) as malicious_count,
                    AVG(confidence) as avg_confidence
                FROM detection_logs
            ''')
            
            result = cursor.fetchone()
            if result:
                total, malicious, avg_conf = result
            else:
                total, malicious, avg_conf = 0, 0, 0
            
            # Daily activity
            cursor.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN is_malicious THEN 1 ELSE 0 END) as malicious_queries
                FROM detection_logs 
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 7
            ''')
            daily_activity = cursor.fetchall()
            
            # Common patterns
            cursor.execute('''
                SELECT detected_patterns, COUNT(*) as count
                FROM detection_logs 
                WHERE detected_patterns IS NOT NULL AND detected_patterns != '[]' AND detected_patterns != 'null'
                GROUP BY detected_patterns
                ORDER BY count DESC
                LIMIT 5
            ''')
            common_patterns = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_queries": total,
                "malicious_count": malicious,
                "benign_count": total - malicious,
                "average_confidence": round(avg_conf, 2) if avg_conf else 0,
                "daily_activity": [
                    {
                        "date": activity[0],
                        "total_queries": activity[1],
                        "malicious_queries": activity[2]
                    }
                    for activity in daily_activity
                ],
                "common_patterns": [
                    pattern[0] for pattern in common_patterns
                ] if common_patterns else []
            }
            
        except Exception as e:
            print(f"Error in get_detection_stats: {e}")
            conn.close()
            return {
                "total_queries": 0,
                "malicious_count": 0,
                "benign_count": 0,
                "average_confidence": 0,
                "daily_activity": [],
                "common_patterns": []
            }
    
    async def get_user_stats(self, user_id: int = 1) -> Dict:
        """Get user statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # User info
            cursor.execute('SELECT username, level, total_score, challenges_completed FROM users WHERE id = ?', (user_id,))
            user_info = cursor.fetchone()
            
            conn.close()
            
            return {
                "user_info": {
                    "username": user_info[0] if user_info else "Unknown",
                    "level": user_info[1] if user_info else 1,
                    "total_score": user_info[2] if user_info else 0,
                    "challenges_completed": user_info[3] if user_info else 0
                }
            }
        except Exception as e:
            print(f"Error in get_user_stats: {e}")
            conn.close()
            return {
                "user_info": {
                    "username": "Unknown",
                    "level": 1,
                    "total_score": 0,
                    "challenges_completed": 0
                }
            }
    
    async def get_user_progress(self, user_id: int = 1) -> Dict:
        """Get user progress and achievements"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # User info
            cursor.execute('''
                SELECT username, level, total_score, challenges_completed 
                FROM users WHERE id = ?
            ''', (user_id,))
            user_info = cursor.fetchone()
            
            # Challenge progress
            cursor.execute('''
                SELECT 
                    challenge_id,
                    COUNT(*) as attempts,
                    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as successes,
                    MAX(score_earned) as best_score
                FROM challenge_submissions 
                WHERE user_id = ?
                GROUP BY challenge_id
            ''', (user_id,))
            challenge_progress = cursor.fetchall()
            
            # Recent activity
            cursor.execute('''
                SELECT 
                    dl.timestamp,
                    dl.query,
                    dl.is_malicious,
                    dl.confidence
                FROM detection_logs dl
                WHERE dl.user_id = ?
                ORDER BY dl.timestamp DESC
                LIMIT 5
            ''', (user_id,))
            recent_detections = cursor.fetchall()
            
            conn.close()
            
            return {
                "user_info": {
                    "username": user_info[0] if user_info else "Unknown",
                    "level": user_info[1] if user_info else 1,
                    "total_score": user_info[2] if user_info else 0,
                    "challenges_completed": user_info[3] if user_info else 0
                },
                "challenge_progress": [
                    {
                        "challenge_id": progress[0],
                        "attempts": progress[1],
                        "successes": progress[2],
                        "success_rate": round((progress[2] / progress[1]) * 100, 2) if progress[1] > 0 else 0,
                        "best_score": progress[3] or 0
                    }
                    for progress in challenge_progress
                ],
                "recent_detections": [
                    {
                        "timestamp": detection[0],
                        "query": detection[1],
                        "is_malicious": bool(detection[2]),
                        "confidence": detection[3]
                    }
                    for detection in recent_detections
                ]
            }
            
        except Exception as e:
            print(f"Error in get_user_progress: {e}")
            conn.close()
            return {
                "user_info": {
                    "username": "demo_user",
                    "level": 1,
                    "total_score": 0,
                    "challenges_completed": 0
                },
                "challenge_progress": [],
                "recent_detections": []
            }