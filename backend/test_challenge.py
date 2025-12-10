# test_challenge.py
import requests
import json

def test_challenge_endpoint():
    base_url = "http://localhost:8000/api"
    
    # Test data for challenge 1
    test_data = {
        "challenge_id": 1,
        "query": "admin' OR '1'='1' --",
        "user_id": 1
    }
    
    print("🧪 Testing Challenge Endpoint...")
    
    try:
        response = requests.post(
            f"{base_url}/submit-challenge",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Challenge submitted successfully!")
            print(f"Result: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_challenge_endpoint()