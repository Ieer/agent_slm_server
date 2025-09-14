import sys
from typing import Any

import requests
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# 配置本地 OpenAI 兼容端點
provider = OpenAIProvider(
    base_url="http://127.0.0.1:8001/v1",
    api_key="local"  # 只作佔位
)

# 簡單健康檢查，避免 APIConnectionError 時才發現服務未啟動
def _check_api_health(base_url: str) -> bool:
    try:
        r = requests.get(base_url.replace("/v1", "") + "/health", timeout=1.5)
        return r.ok
    except Exception:
        return False

model = OpenAIModel("qwen2.5-0.5b-instruct", provider=provider)
lm_agent = Agent(
    model,
    system_prompt="請你調用合適的工具函數完成任務，並用中文回答結果，如果沒有適合調用的函數就回復\"我無法處理本次任務\"。",
)

# 测试调用
@lm_agent.tool
def add_numbers(ctx: RunContext[Any], a: int, b: int) -> int:
    """返回两个数字的和"""
    return (a + b)*2  # 故意乘以2來檢查結果是否被模型使用

# 测试调用, 我的名字是小明, 今年10岁, 体重30公斤, 身高1.4米, 你能帮我算算我的BMI吗？
@lm_agent.tool
def calculate_bmi(ctx: RunContext[Any], weight_kg: float, height_m: float) -> float:
    """计算BMI指数"""
    if height_m <= 0:
        return 0.0
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)

if __name__ == "__main__":
    api_ok = _check_api_health("http://127.0.0.1:8001/v1")
    if not api_ok:
        print("本地 API 尚未啟動，請先執行: python -m uvicorn qwen_api_server:app --host 127.0.0.1 --port 8001")
        sys.exit(1)

    import time
    print("===== 測試1: 計算5和7的和 =====")
    # 启动时间，格式为：yyy-mm-dd hh:mm:ss
    time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"啟動時間: {time1}")
    res3 = lm_agent.run_sync("5 和 7 的和是多少？")
    print(f"原始輸出: {res3.output}")
    print(f"使用情況: {res3.usage()}")
    time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"結束時間: {time2}")
    time_diff = time.mktime(time.strptime(time2, "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(time1, "%Y-%m-%d %H:%M:%S"))
    print(f"總共耗時: {time_diff} 秒")

    print("\n===== 測試2: 計算BMI =====")
    time3 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"啟動時間: {time3}")
    res4 = lm_agent.run_sync("我的名字是小明, 今年10岁, 体重30公斤,身高1.4米, 你能帮我算算我的BMI吗？")
    print(f"原始輸出: {res4.output}")
    print(f"使用情況: {res4.usage()}")
    time4 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"結束時間: {time4}")
    time_diff2 = time.mktime(time.strptime(time4, "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(time3, "%Y-%m-%d %H:%M:%S"))
    print(f"總共耗時: {time_diff2} 秒")
