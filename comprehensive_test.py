"""
项目功能和性能综合测试脚本
测试各个API端点的功能性和性能表现
"""

import requests
import time
import sys
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# API 配置
API_BASE = "http://127.0.0.1:8001"
API_TIMEOUT = 30

def test_api_availability() -> bool:
    """测试API是否可用"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"API不可用: {e}")
        return False

def test_health_endpoints():
    """测试健康检查端点"""
    print("\n=== 健康检查测试 ===")
    
    endpoints = [
        "/health",
        "/health/memory", 
        "/health/detailed",
        "/metrics"
    ]
    
    for endpoint in endpoints:
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE}{endpoint}", timeout=10)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✓ {endpoint}: {response.status_code} ({elapsed:.2f}s)")
                if endpoint == "/health/detailed":
                    data = response.json()
                    print(f"  状态: {data.get('status')}")
                    if 'memory' in data:
                        memory = data['memory']
                        print(f"  内存使用: {memory.get('current_usage_percent', 0):.1%}")
                        print(f"  可用内存: {memory.get('available_gb', 0):.1f}GB")
            else:
                print(f"✗ {endpoint}: {response.status_code}")
                
        except Exception as e:
            print(f"✗ {endpoint}: 错误 - {e}")

def test_model_list():
    """测试模型列表API"""
    print("\n=== 模型列表测试 ===")
    try:
        response = requests.get(f"{API_BASE}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            print(f"✓ 找到 {len(models)} 个模型:")
            for model in models:
                print(f"  - {model.get('id')}")
            return True
        else:
            print(f"✗ 模型列表获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 模型列表测试失败: {e}")
        return False

def test_chat_completion():
    """测试聊天完成API"""
    print("\n=== 聊天完成测试 ===")
    
    test_cases = [
        {
            "name": "简单问答",
            "messages": [{"role": "user", "content": "1+1等于多少?"}],
            "max_tokens": 50
        },
        {
            "name": "中文对话",
            "messages": [
                {"role": "system", "content": "你是一个友善的助手"},
                {"role": "user", "content": "请介绍一下人工智能"}
            ],
            "max_tokens": 100
        },
        {
            "name": "工具调用测试",
            "messages": [{"role": "user", "content": "请帮我计算5+3的结果"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "计算两个数的和",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"}
                            },
                            "required": ["a", "b"]
                        }
                    }
                }
            ],
            "max_tokens": 50
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        try:
            start_time = time.time()
            
            payload = {
                "model": "qwen2.5-0.5b-instruct",
                "messages": test_case["messages"],
                "max_tokens": test_case["max_tokens"],
                "temperature": 0.7
            }
            
            if "tools" in test_case:
                payload["tools"] = test_case["tools"]
            
            response = requests.post(
                f"{API_BASE}/v1/chat/completions",
                json=payload,
                timeout=API_TIMEOUT
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                
                print(f"✓ 响应时间: {elapsed:.2f}s")
                print(f"  完成原因: {choice.get('finish_reason')}")
                
                if message.get("content"):
                    content = message["content"][:100]
                    print(f"  内容: {content}{'...' if len(message.get('content', '')) > 100 else ''}")
                
                if message.get("tool_calls"):
                    print(f"  工具调用: {len(message['tool_calls'])} 个")
                    for tool_call in message["tool_calls"]:
                        func = tool_call["function"]
                        print(f"    - {func['name']}: {func['arguments']}")
                
                # 显示token使用情况
                usage = data.get("usage", {})
                print(f"  Token使用: 输入{usage.get('prompt_tokens', 0)} + "
                      f"输出{usage.get('completion_tokens', 0)} = "
                      f"总计{usage.get('total_tokens', 0)}")
                
            else:
                print(f"✗ 失败: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  错误: {error_data.get('detail', '未知错误')}")
                except:
                    print(f"  响应: {response.text[:200]}")
                    
        except Exception as e:
            print(f"✗ 异常: {e}")

def test_embeddings():
    """测试嵌入API"""
    print("\n=== 嵌入测试 ===")
    
    test_cases = [
        {
            "name": "单个文本",
            "input": "人工智能是计算机科学的一个分支"
        },
        {
            "name": "批量文本",
            "input": [
                "机器学习是人工智能的重要技术",
                "深度学习基于神经网络",
                "自然语言处理让机器理解人类语言"
            ]
        },
        {
            "name": "长文本",
            "input": "人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，它致力于理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。人工智能研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。自1956年提出以来，人工智能经历了几次兴衰，现在正处于新的发展高峰期。" * 3
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        try:
            start_time = time.time()
            
            payload = {
                "model": "paraphrase-MiniLM-L6-v2",
                "input": test_case["input"],
                "encoding_format": "float"
            }
            
            response = requests.post(
                f"{API_BASE}/v1/embeddings",
                json=payload,
                timeout=API_TIMEOUT
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                embeddings = data["data"]
                
                print(f"✓ 响应时间: {elapsed:.2f}s")
                print(f"  嵌入数量: {len(embeddings)}")
                
                if embeddings:
                    first_embedding = embeddings[0]["embedding"]
                    print(f"  向量维度: {len(first_embedding)}")
                    print(f"  向量样例: [{first_embedding[0]:.4f}, {first_embedding[1]:.4f}, ...]")
                
                usage = data.get("usage", {})
                print(f"  Token使用: {usage.get('prompt_tokens', 0)} 个")
                
            else:
                print(f"✗ 失败: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  错误: {error_data.get('detail', '未知错误')}")
                except:
                    print(f"  响应: {response.text[:200]}")
                    
        except Exception as e:
            print(f"✗ 异常: {e}")

def test_concurrent_requests():
    """测试并发请求性能"""
    print("\n=== 并发性能测试 ===")
    
    def make_request(request_id: int) -> Dict[str, Any]:
        """发送单个请求"""
        start_time = time.time()
        try:
            response = requests.post(
                f"{API_BASE}/v1/chat/completions",
                json={
                    "model": "qwen2.5-0.5b-instruct",
                    "messages": [{"role": "user", "content": f"请回答问题 #{request_id}: 什么是编程?"}],
                    "max_tokens": 50,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": response.status_code == 200,
                "elapsed": elapsed,
                "status_code": response.status_code
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "request_id": request_id,
                "success": False,
                "elapsed": elapsed,
                "error": str(e)
            }
    
    # 测试不同并发级别
    concurrency_levels = [1, 3, 5]
    
    for concurrency in concurrency_levels:
        print(f"\n并发级别: {concurrency}")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(make_request, i) for i in range(concurrency)]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        total_elapsed = time.time() - start_time
        
        # 统计结果
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if successful_requests:
            avg_response_time = sum(r["elapsed"] for r in successful_requests) / len(successful_requests)
            min_response_time = min(r["elapsed"] for r in successful_requests)
            max_response_time = max(r["elapsed"] for r in successful_requests)
            
            print(f"✓ 成功: {len(successful_requests)}/{len(results)}")
            print(f"  总耗时: {total_elapsed:.2f}s")
            print(f"  平均响应时间: {avg_response_time:.2f}s")
            print(f"  响应时间范围: {min_response_time:.2f}s - {max_response_time:.2f}s")
            print(f"  吞吐量: {len(successful_requests)/total_elapsed:.2f} 请求/秒")
        else:
            print("全部失败")
        
        if failed_requests:
            print(f"✗ 失败: {len(failed_requests)} 个")
            for failed in failed_requests[:3]:  # 只显示前3个错误
                error_msg = failed.get("error", f"HTTP {failed.get('status_code')}")
                print(f"  - 请求 #{failed['request_id']}: {error_msg}")

def test_memory_monitoring():
    """测试内存监控功能"""
    print("\n=== 内存监控测试 ===")
    
    try:
        # 获取初始内存状态
        response = requests.get(f"{API_BASE}/health/memory", timeout=10)
        if response.status_code == 200:
            initial_memory = response.json()
            print(f"初始内存使用: {initial_memory.get('current_usage_percent', 0):.1%}")
            print(f"可用内存: {initial_memory.get('available_gb', 0):.1f}GB")
            
            # 执行一些操作来观察内存变化
            print("\n执行内存密集型操作...")
            
            # 大批量嵌入请求
            large_texts = ["这是一个测试文本。" * 100] * 10
            requests.post(
                f"{API_BASE}/v1/embeddings",
                json={
                    "model": "paraphrase-MiniLM-L6-v2",
                    "input": large_texts
                },
                timeout=30
            )
            
            # 再次检查内存
            response = requests.get(f"{API_BASE}/health/memory", timeout=10)
            if response.status_code == 200:
                final_memory = response.json()
                print(f"操作后内存使用: {final_memory.get('current_usage_percent', 0):.1%}")
                
                usage_change = final_memory.get('current_usage_percent', 0) - initial_memory.get('current_usage_percent', 0)
                print(f"内存使用变化: {usage_change:+.1%}")
                
                if final_memory.get('leak_detected'):
                    print("⚠️ 检测到可能的内存泄漏")
                else:
                    print("✓ 未检测到内存泄漏")
            
        else:
            print(f"✗ 内存监控API不可用: {response.status_code}")
            
    except Exception as e:
        print(f"✗ 内存监控测试失败: {e}")

def main():
    """主测试函数"""
    print("=== 项目功能与性能综合测试 ===")
    print(f"API地址: {API_BASE}")
    
    # 检查API可用性
    if not test_api_availability():
        print("❌ API服务不可用，请确保服务已启动")
        print("启动命令: python -m uvicorn slm_server:app --host 127.0.0.1 --port 8001")
        sys.exit(1)
    
    print("✅ API服务可用")
    
    # 执行各项测试
    try:
        test_health_endpoints()
        test_model_list()
        test_chat_completion()
        test_embeddings()
        test_concurrent_requests()
        test_memory_monitoring()
        
        print("\n=== 测试总结 ===")
        print("✅ 功能测试完成")
        print("\n建议的性能优化措施:")
        print("1. 监控内存使用情况，必要时调整模型卸载策略")
        print("2. 根据并发测试结果调整MAX_CONCURRENT_REQUESTS")
        print("3. 对于高负载场景，考虑启用模型量化")
        print("4. 定期检查/health/detailed端点的内存趋势")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()