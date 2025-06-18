import requests
import json

with open("example_request.json", "r", encoding="utf-8") as f:
    data = json.load(f)

url = "http://127.0.0.1:8000/prediction/multimodal"

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Response Text:", response.text)
