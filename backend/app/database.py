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
        conn = self.get_connection()
        cursor = conn.cursor()
        
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
        
        conn.commit()
        conn.close()
    
    async def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total_queries,
                SUM(CASE WHEN is_malicious THEN 1 ELSE 0 END) as malicious_count,
                AVG(confidence) as avg_confidence
            FROM detection_logs
        ''')
        
        total, malicious, avg_conf = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_queries": total or 0,
            "malicious_count": malicious or 0,
            "benign_count": (total or 0) - (malicious or 0),
            "average_confidence": round(avg_conf or 0, 2)
        }
    
    async def get_user_stats(self, user_id: int = 1) -> Dict:
        """Get user statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
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