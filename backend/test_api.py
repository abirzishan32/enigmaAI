#!/usr/bin/env python3
"""
API Test Script
Tests the FHE MNIST API endpoints
"""

import requests
import numpy as np
import json
import sys

API_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("Testing GET / ...")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Root endpoint: {data.get('message')}")
            print(f"  FHE Enabled: {data.get('fhe_enabled')}")
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_health():
    """Test health endpoint"""
    print("\nTesting GET /health ...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed")
            print(f"  Model loaded: {data.get('model_loaded')}")
            print(f"  Context loaded: {data.get('context_loaded')}")
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting GET /api/model/info ...")
    try:
        response = requests.get(f"{API_URL}/api/model/info")
        if response.status_code == 200:
            data = response.json()
            info = data.get('model_info', {})
            print(f"✓ Model info retrieved")
            print(f"  Model type: {info.get('model_type')}")
            print(f"  Framework: {info.get('framework')}")
            print(f"  Test accuracy: {info.get('test_accuracy'):.2f}%")
            print(f"  FHE scheme: {info.get('fhe_scheme')}")
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_clear_inference():
    """Test clear inference endpoint"""
    print("\nTesting POST /api/inference/clear ...")
    
    # Create a simple test image: vertical line (digit 1)
    pixels = np.zeros(784)
    for i in range(28):
        pixels[i * 28 + 13] = 1.0  # Vertical line in middle
        pixels[i * 28 + 14] = 1.0
    
    try:
        response = requests.post(
            f"{API_URL}/api/inference/clear",
            json={"pixels": pixels.tolist()}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Inference completed")
            print(f"  Prediction: {data.get('prediction')}")
            print(f"  Status: {data.get('status')}")
            
            # Show top 3 probabilities
            probs = data.get('probabilities', [])
            sorted_indices = np.argsort(probs)[::-1][:3]
            print(f"  Top 3 predictions:")
            for idx in sorted_indices:
                print(f"    Digit {idx}: {probs[idx]*100:.2f}%")
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            print(f"  {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_public_context():
    """Test public context download"""
    print("\nTesting GET /api/model/public-context ...")
    try:
        response = requests.get(f"{API_URL}/api/model/public-context")
        if response.status_code == 200:
            size_kb = len(response.content) / 1024
            print(f"✓ Public context downloaded: {size_kb:.2f} KB")
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("FHE MNIST API Test Suite")
    print("=" * 60)
    print(f"\nTesting API at: {API_URL}")
    print("\nMake sure the server is running:")
    print("  cd backend && uvicorn app:app --reload")
    print()
    
    input("Press Enter to start tests...")
    print()
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Public Context", test_public_context),
        ("Clear Inference", test_clear_inference),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        print("\nAPI is ready for use.")
        print("Start the frontend: cd frontend && npm run dev")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nCheck the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
