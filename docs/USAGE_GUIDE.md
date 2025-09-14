<!-- 此檔為根目錄 USAGE_GUIDE.md 完整複製版本（避免 MkDocs 404）。更新時建議只維護本檔，根目錄可放 symlink 或簡化版。 -->

# SLM 伺服器使用指南（進階）

> 總覽請見 `README.md`；記憶體與量化細節：`memory-optimization-guide.md`。

## 目錄

- [快速開始](#快速開始)
- [llama.cpp / GGUF 後端](#llamacpp--gguf-後端)
- [配置優化](#配置優化)
- [監控與維護](#監控與維護)
- [故障排除](#故障排除)
- [性能調優指南](#性能調優指南)
- [測試與驗證](#測試與驗證)
- [API 使用示例](#api-使用示例)
- [生產部署建議](#生產部署建議)
- [技術支持](#技術支持)
- [延伸閱讀：記憶體優化](memory-optimization-guide.md)

## 快速開始

### 1. 檢查系統需求

```powershell
# 检查可用内存
python -c "import psutil; print(f'可用内存: {psutil.virtual_memory().available/1024**3:.1f}GB')"

# 检查Python版本 (推荐3.8+)
python --version
```text

### 2. 安裝依賴

```powershell
# 基础依赖
pip install fastapi uvicorn transformers torch pydantic psutil

# 可选依赖 (用于量化和嵌入)
pip install bitsandbytes sentence-transformers tiktoken
```text

### 3. 啟動服務

#### 方式一: 使用啟動腳本 (推薦)

```powershell
python start_server.py
```text

#### 方式二: 直接啟動

```powershell
# 完整版 (需要所有依赖)
python -m uvicorn slm_server:app --host 127.0.0.1 --port 8001

# 简化版 (减少依赖)
python -m uvicorn slm_server_simple:app --host 127.0.0.1 --port 8001
```text

### 4. 驗證服務

```powershell
# 基本测试
python test_simple.py

# 完整测试
python comprehensive_test.py
```

## llama.cpp / GGUF 後端

當 `transformers` 主後端載入 safetensors 受到記憶體 / Pagefile 限制（Windows 1455）或希望更低佔用時，可改用 `MODEL_BACKEND=llama.cpp` + GGUF 量化檔：

### 安裝

```powershell
# 方式一：使用 extras
pip install -e .[llama]
# 方式二：僅安裝核心套件
pip install llama-cpp-python
```

若安裝失敗（需要編譯）:
```powershell
pip install --upgrade pip
pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels
```

### 啟動（基礎）
```powershell
$env:MODEL_BACKEND="llama.cpp"
$env:MODEL_PATH="models/qwen/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
$env:COMPILE_MODEL="0"
$env:MAX_INPUT_TOKENS="1024"
uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

### 使用腳本
```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_llama_cpp.ps1 -Ctx 1024 -Threads 6
```
或 Linux / macOS：
```bash
bash scripts/start_llama_cpp.sh
```

### 重要環境變數

| 變數 | 說明 | 建議 |
|------|------|------|
| MODEL_BACKEND | 後端選擇 | 設為 `llama.cpp` |
| MODEL_PATH | GGUF 模型檔案路徑 | 指向 .gguf |
| MAX_INPUT_TOKENS / LLAMA_CTX | 上下文長度 | 512~2048 視記憶體 |
| LLAMA_THREADS | CPU 執行緒 | 實體核心數或核心數-1 |
| COMPILE_MODEL | Torch compile（不適用） | 固定 0 |

### 記憶體策略

- GGUF Q4_K_M 量化顯著降低常駐內存
- 若仍遇 `context full` 可降低 `MAX_INPUT_TOKENS`
- 減少同時請求數（可於外部限流）

### 回退 transformers

```powershell
Remove-Item Env:MODEL_BACKEND
# 或
$env:MODEL_BACKEND="transformers"
```

### 常見錯誤

| 錯誤 | 原因 | 解法 |
|------|------|------|
| llama_cpp not installed | 缺套件 | 安裝 `llama-cpp-python` |
| model path not found | 路徑錯 | 確認 `MODEL_PATH` 指向 .gguf |
| context too large | ctx 超過模型配置 | 降低 `LLAMA_CTX` / `MAX_INPUT_TOKENS` |
| high latency | 執行緒不足 | 調整 `LLAMA_THREADS` |

## 配置優化

### 環境變數配置

```powershell
# 内存优化 (必须)
set LOW_CPU_MEM_USAGE=1
set MODEL_UNLOAD_ENABLED=1
set MODEL_IDLE_TIMEOUT=1800

# 性能调优 (推荐)
set TORCH_NUM_THREADS=4
set MAX_INPUT_TOKENS=1024
set MAX_CONCURRENT_REQUESTS=5

# 内存受限环境 (可选)
set MODEL_QUANTIZATION=4bit

# 监控和调试 (开发环境)
set LOG_LEVEL=INFO
set ENABLE_DETAILED_METRICS=1
```

### 批次檔設定 (Windows)

建立 `start_optimized.bat`:

```batch
@echo off
set LOW_CPU_MEM_USAGE=1
set MODEL_UNLOAD_ENABLED=1
set MODEL_IDLE_TIMEOUT=1800
set TORCH_NUM_THREADS=4
set MAX_INPUT_TOKENS=1024
set LOG_LEVEL=INFO

echo 启动优化版SLM服务器...
python -m uvicorn slm_server:app --host 127.0.0.1 --port 8001
pause
```

## 監控與維護

### 健康检查端点

| 端点 | 功能 | 用途 |
|------|------|------|
| `/health` | 基本状态检查 | 负载均衡器健康检查 |
| `/health/memory` | 内存使用情况 | 内存监控 |
| `/health/detailed` | 详细系统状态 | 运维监控 |
| `/metrics` | 性能指标 | Prometheus集成 |

### 監控腳本示例

```python
import requests
import time

def monitor_service():
    while True:
        try:
            response = requests.get("http://127.0.0.1:8001/health/detailed")
            data = response.json()

            memory_usage = data['memory']['current_usage_percent']
            status = data['status']

            print(f"[{time.strftime('%H:%M:%S')}] 状态: {status}, 内存: {memory_usage:.1%}")

            if memory_usage > 0.9:
                print("⚠️ 内存使用率过高!")

        except Exception as e:
            print(f"❌ 监控失败: {e}")

        time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    monitor_service()
```

## 故障排除

### 常见问题及解决方案

#### 1. 記憶體不足錯誤

```
错误: 系统虚擬記憶體不足，無法載入大型模型
解决方案:
- 启用量化: set MODEL_QUANTIZATION=4bit
- 增加虚拟内存 (页面文件)
- 减少并发: set MAX_CONCURRENT_REQUESTS=2
```

#### 2. 模型載入失敗

```
错误: 模型載入失敗
解决方案:
- 检查模型路径: set MODEL_PATH=./models/qwen/Qwen2.5-0.5B-Instruct
- 检查模型文件完整性
- 确认磁盘空间充足
```

#### 3. 回應速度慢

```
问题: 推理速度慢
解决方案:
- 调整线程数: set TORCH_NUM_THREADS=4
- 启用模型编译: set COMPILE_MODEL=1 (需要PyTorch 2.0+)
- 减少输入长度: set MAX_INPUT_TOKENS=512
```

#### 4. 服務啟動失敗

```
问题: uvicorn启动失败
解决方案:
- 检查端口占用: netstat -an | findstr :8001
- 使用简化版: python -m uvicorn slm_server_simple:app
- 检查依赖安装: pip install -r requirements-optimized.txt
```

## 性能調優指南

### 依硬體配置調優

#### 低記憶體環境 (< 8GB)

```powershell
set MODEL_QUANTIZATION=4bit
set MAX_CONCURRENT_REQUESTS=2
set MAX_INPUT_TOKENS=512
set MODEL_IDLE_TIMEOUT=300
```

#### 中等記憶體環境 (8-16GB)

```powershell
set MODEL_QUANTIZATION=8bit
set MAX_CONCURRENT_REQUESTS=5
set MAX_INPUT_TOKENS=1024
set MODEL_IDLE_TIMEOUT=1800
```

#### 高記憶體環境 (> 16GB)

```powershell
# 不使用量化
set MAX_CONCURRENT_REQUESTS=10
set MAX_INPUT_TOKENS=2048
set MODEL_IDLE_TIMEOUT=3600
```

### CPU 最佳化

```powershell
# 根据CPU核心数设置线程
# 4核心系统
set TORCH_NUM_THREADS=4

# 8核心系统
set TORCH_NUM_THREADS=6

# 16核心系统
set TORCH_NUM_THREADS=12
```

## 測試與驗證

### 基本功能測試

```powershell
# 快速测试
curl http://127.0.0.1:8001/health

# 聊天测试
curl -X POST http://127.0.0.1:8001/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"qwen2.5-0.5b-instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":50}"
```

### 壓力測試

```python
import requests
import threading
import time

def stress_test(num_requests=10):
    def make_request():
        try:
            response = requests.post(
                "http://127.0.0.1:8001/v1/chat/completions",
                json={
                    "model": "qwen2.5-0.5b-instruct",
                    "messages": [{"role": "user", "content": "测试"}],
                    "max_tokens": 50
                },
                timeout=30
            )
            return response.status_code == 200
        except:
            return False

    threads = []
    results = []

    start_time = time.time()

    for i in range(num_requests):
        thread = threading.Thread(target=lambda: results.append(make_request()))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    duration = time.time() - start_time
    success_rate = sum(results) / len(results)

    print(f"压力测试结果: {num_requests}个请求, {duration:.2f}s, 成功率: {success_rate:.1%}")

if __name__ == "__main__":
    stress_test()
```

## API 使用示例

### Chat API

```python
import requests

response = requests.post(
    "http://127.0.0.1:8001/v1/chat/completions",
    json={
        "model": "qwen2.5-0.5b-instruct",
        "messages": [
            {"role": "system", "content": "你是一个helpful的助手"},
            {"role": "user", "content": "解释什么是人工智能"}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
)

data = response.json()
print(data["choices"][0]["message"]["content"])
```

### Embeddings API

```python
import requests

response = requests.post(
    "http://127.0.0.1:8001/v1/embeddings",
    json={
        "model": "paraphrase-MiniLM-L6-v2",
        "input": ["人工智能", "机器学习", "深度学习"],
        "encoding_format": "float"
    }
)

data = response.json()
for item in data["data"]:
    print(f"向量维度: {len(item['embedding'])}")
```

## 生產部署建議

### Docker 部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements-optimized.txt .
RUN pip install -r requirements-optimized.txt

COPY . .

# 设置环境变量
ENV LOW_CPU_MEM_USAGE=1
ENV MODEL_UNLOAD_ENABLED=1
ENV MAX_CONCURRENT_REQUESTS=5

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "slm_server:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 反向代理配置 (Nginx)

```nginx
upstream slm_backend {
    server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://slm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location /health {
        proxy_pass http://slm_backend/health;
        access_log off;
    }
}
```

## 技術支持

如遇到问题，请按以下顺序排查:

1. 检查服务状态: `GET /health/detailed`
2. 查看错误日志
3. 验证环境变量配置
4. 运行测试脚本
5. 检查系统资源使用情况

---

**注意**: 本指南基於 Windows 環境撰寫，Linux / macOS 使用者請調整命令語法。
