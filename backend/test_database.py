# test_database.py
import sqlite3
import pandas as pd
from app.config import db_manager

def test_database():
    """Test database connectivity and data"""
    print("🧪 Testing Database Connection...")
    
    try:
        # Test basic connection
        conn = sqlite3.connect('database/detection_logs.db')
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("📊 Database Tables:")
        for table in tables:
            print(f"   - {table[0]}")
        
        # Check detection logs
        cursor.execute("SELECT COUNT(*) FROM detection_logs")
        detection_count = cursor.fetchone()[0]
        print(f"📈 Detection Logs: {detection_count} records")
        
        # Check challenge submissions
        cursor.execute("SELECT COUNT(*) FROM challenge_submissions")
        challenge_count = cursor.fetchone()[0]
        print(f"🎯 Challenge Submissions: {challenge_count} records")
        
        # Check users
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        print(f"👥 Users: {len(users)} records")
        
        # Show recent detections
        cursor.execute("""
            SELECT query, is_malicious, confidence, timestamp 
            FROM detection_logs 
            ORDER BY timestamp DESC 
            LIMIT 5
        """)
        recent_detections = cursor.fetchall()
        print("\n🔍 Recent Detections:")
        for detection in recent_detections:
            status = "🔴 MALICIOUS" if detection[1] else "🟢 SAFE"
            print(f"   {status} ({detection[2]}%): {detection[0][:50]}...")
        
        conn.close()
        print("✅ Database test completed successfully!")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")

if __name__ == "__main__":
    test_database()