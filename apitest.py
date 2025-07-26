import requests
import json
from PIL import Image
import io
import os

def test_api():
    """Test the FastAPI endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Waste Classification API")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        print("   ✅ Health check passed\n")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}\n")
        return
    
    # Test 2: Get classes
    print("2. Testing classes endpoint...")
    try:
        response = requests.get(f"{base_url}/classes")
        data = response.json()
        print(f"   Classes: {data.get('classes', [])}")
        print("   ✅ Classes endpoint passed\n")
    except Exception as e:
        print(f"   ❌ Classes endpoint failed: {e}\n")
    
    # Test 3: Root endpoint
    print("3. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Response: {response.json()}")
        print("   ✅ Root endpoint passed\n")
    except Exception as e:
        print(f"   ❌ Root endpoint failed: {e}\n")
    
    # Test 4: Image classification (if test image exists)
    print("4. Testing image classification...")
    
    # Create a test image if none exists
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print("   Creating test image...")
        # Create a simple colored image for testing
        img = Image.new('RGB', (224, 224), color='green')
        img.save(test_image_path)
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post(f"{base_url}/classify", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Prediction: {result['prediction']['class']}")
            print(f"   Confidence: {result['prediction']['confidence']:.4f}")
            print(f"   Recyclable: {result['recycling_info']['recyclable']}")
            print("   ✅ Classification test passed\n")
        else:
            print(f"   ❌ Classification failed with status: {response.status_code}")
            print(f"   Error: {response.text}\n")
    
    except Exception as e:
        print(f"   ❌ Classification test failed: {e}\n")
    
    print("🎉 API testing completed!")

if __name__ == "__main__":
    test_api()