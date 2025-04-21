import requests
import json

# 1. JSON 파일 읽기
with open("example_request.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. API 엔드포인트 URL
url = "http://127.0.0.1:8000/prediction/multimodal"

# 3. POST 요청 보내기
response = requests.post(url, json=data)

# 4. 결과 출력
print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Response Text:", response.text)
