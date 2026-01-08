import requests
import numpy as np
import json
import random

def test_classify():
    url = "http://localhost:8000/classify"
    
    # Create a dummy image (random noise or zero)
    # 28x28 image with values between 0 and 1
    dummy_image = np.random.rand(28 * 28).tolist()
    
    payload = {
        "image": dummy_image
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("Success!")
            print("Response:", json.dumps(result, indent=2))
            
            if "prediction" in result and "encrypted" in result:
                print("Verification PASSED: Received valid prediction.")
            else:
                print("Verification FAILED: Missing fields in response.")
        else:
            print(f"Failed with status code: {response.status_code}")
            print("Response:", response.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_classify()
