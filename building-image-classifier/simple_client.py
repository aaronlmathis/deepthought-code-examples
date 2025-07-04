import requests

# Test health endpoint
response = requests.get("http://127.0.0.1:8000/health/")
print("Health check:", response.json())

# Test prediction with an image file
with open("/home/amathis/Downloads/frog.png", "rb") as f:
    files = {"file": f}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    
if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
else:
    print(f"Error: {response.status_code} - {response.text}")