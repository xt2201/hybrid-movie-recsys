import requests
import json
import time
import subprocess
import sys

def test_api():
    base_url = "http://localhost:8000"
    
    # Wait for API to be ready
    print("Waiting for API to be ready...")
    for _ in range(30):
        try:
            resp = requests.get(f"{base_url}/health")
            if resp.status_code == 200:
                print("API is ready.")
                break
        except requests.ConnectionError:
            time.sleep(2)
    else:
        print("API failed to start.")
        return

    # Test Recommendation
    print("\nTesting /recommend endpoint...")
    payload = {
        "user_id": 1,
        "query": "I want a funny action movie",
        "num_items": 5,
        "use_llm": True
    }
    
    try:
        resp = requests.post(f"{base_url}/recommend", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            print("Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    # Start API in background
    print("Starting API...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**dict(sys.modules['os'].environ), "PYTHONPATH": "."}
    )
    
    try:
        test_api()
    finally:
        print("Stopping API...")
        proc.terminate()
        proc.wait()
