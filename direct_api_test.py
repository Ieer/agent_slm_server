import requests
import json
import sys

def check_health(base_url):
    """檢查API健康狀態"""
    try:
        r = requests.get(base_url.replace("/v1", "") + "/health", timeout=1.5)
        return r.ok
    except Exception:
        return False

def call_chat_api(prompt, base_url="http://127.0.0.1:8001/v1"):
    """直接調用API進行聊天"""
    # 檢查API是否可用
    if not check_health(base_url):
        print("API伺服器未啟動或無法訪問")
        return None
    
    # 構建請求
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer local"  # 本地伺服器只需任何token
    }
    data = {
        "model": "qwen2.5-0.5b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False  # 不使用流式傳輸
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API調用錯誤: {e}")
        return None

def test_add_numbers():
    """測試加法功能"""
    prompt = "5和7的和是多少？請使用add_numbers工具幫我計算。"
    result = call_chat_api(prompt)
    if result:
        print("===== 加法測試 =====")
        print(f"問題: {prompt}")
        choices = result["choices"]
        message = choices[0]["message"]
        content = message["content"]
        print(f"回答: {content}")
        print(f"使用量: {result[\"usage\"]}")

def test_bmi_calculation():
    """測試BMI計算功能"""
    prompt = "我的名字是小明，今年10歲，體重30公斤，身高1.4米，請計算我的BMI。"
    result = call_chat_api(prompt)
    if result:
        print("\n===== BMI測試 =====")
        print(f"問題: {prompt}")
        choices = result["choices"]
        message = choices[0]["message"]
        content = message["content"]
        print(f"回答: {content}")
        print(f"使用量: {result[\"usage\"]}")

if __name__ == "__main__":
    test_add_numbers()
    test_bmi_calculation()

