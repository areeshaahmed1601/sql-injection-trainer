from flask import Flask, request, jsonify
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

@app.route('/')
def root():
    return jsonify({"message": "SQL Injection Trainer API is running!"})

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/api/check-sql', methods=['POST'])
def check_sql():
    data = request.json
    query = data.get('query', '').lower()
    
    malicious_patterns = [
        ("or '1'='1'", "tautology"),
        ("or 1=1", "tautology"),
        ("union select", "union_attack"),
        ("--", "comment_attack"),
        ("/*", "comment_attack"),
        ("; drop", "stacked_queries"),
        ("; delete", "stacked_queries"),
        ("sleep(", "time_delay"),
        ("waitfor delay", "time_delay"),
        ("insert into", "data_manipulation"),
        ("update set", "data_manipulation"),
        ("drop table", "destructive"),
        ("create table", "schema_manipulation")
    ]
    
    detected_patterns = []
    for pattern, category in malicious_patterns:
        if pattern in query:
            detected_patterns.append(f"{category}: {pattern}")
    
    is_malicious = len(detected_patterns) > 0
    confidence = min(len(detected_patterns) * 15, 95)
    
    return jsonify({
        "is_malicious": is_malicious,
        "confidence": confidence,
        "message": "Malicious SQL injection patterns detected" if is_malicious else "Query appears to be safe",
        "detected_patterns": detected_patterns
    })

@app.route('/api/submit-challenge', methods=['POST'])
def submit_challenge():
    data = request.json
    challenge_id = data.get('challenge_id', 1)
    query = data.get('query', '').lower()
    
    # Challenge-specific solutions
    challenge_solutions = {
        1: ["or '1'='1'", "or 1=1"],  # Login Bypass
        2: ["union select"],           # Data Extraction
        3: ["information_schema"],     # Database Schema
        4: ["sleep(", "waitfor"]       # Blind Injection
    }
    
    # Check if solution contains required patterns
    required_patterns = challenge_solutions.get(challenge_id, [])
    solution_found = any(pattern in query for pattern in required_patterns)
    
    # Additional basic malicious pattern check
    basic_patterns = ["or '1'='1'", "or 1=1", "union", "--", "/*", ";", "sleep(", "waitfor"]
    is_malicious = any(pattern in query for pattern in basic_patterns)
    
    # Calculate score based on challenge difficulty
    base_scores = {1: 50, 2: 75, 3: 100, 4: 150}
    score = base_scores.get(challenge_id, 50) if solution_found and is_malicious else 0
    
    detected_patterns = []
    for pattern in basic_patterns:
        if pattern in query:
            detected_patterns.append(pattern)
    
    return jsonify({
        "challenge_id": challenge_id,
        "is_malicious": is_malicious,
        "completed": solution_found and is_malicious,
        "score": score,
        "confidence": min(len(detected_patterns) * 20, 95),
        "message": "Challenge completed successfully!" if solution_found and is_malicious else "Keep trying! Your solution needs to match the challenge requirements.",
        "detected_patterns": detected_patterns
    })

if __name__ == '__main__':
    print("🚀 SQL Injection Trainer API starting...")
    print("📚 Available endpoints:")
    print("   GET  / - API status")
    print("   GET  /health - Health check")
    print("   POST /api/check-sql - Check SQL query")
    print("   POST /api/submit-challenge - Submit challenge solution")
    print("🔗 Frontend: http://localhost:3000")
    print("🔗 Backend:  http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)