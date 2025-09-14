# 記憶體優化和模型量化指南

本文檔提供了如何使用新增的記憶體優化和模型量化功能，以提高系統在低規格硬體上的穩定性和效能。

## 1. 安裝優化依賴

為了啟用模型量化和記憶體優化，請安裝新增的依賴項：

```bash
pip install -r requirements-optimized.txt
```

這將安裝 bitsandbytes、accelerate 等必要的庫，使模型量化和其他優化功能正常工作。

## 2. 環境變數配置

系統現在支援多種環境變數來控制優化行為：

### 2.1 模型量化

```bash
# 啟用 4-bit 量化（記憶體使用最少，約減少75%）
export MODEL_QUANTIZATION=4bit

# 或啟用 8-bit 量化（平衡記憶體和精度，約減少50%）
export MODEL_QUANTIZATION=8bit

# 不使用量化（預設）
export MODEL_QUANTIZATION=none
```

### 2.2 模型卸載

```bash
# 啟用模型卸載（預設）
export MODEL_UNLOAD_ENABLED=1

# 模型閒置多長時間後卸載（秒）
export MODEL_IDLE_TIMEOUT=3600  # 預設1小時

# 檢查閒置模型的頻率（秒）
export MODEL_UNLOAD_CHECK_INTERVAL=600  # 預設10分鐘
```

### 2.3 其他記憶體優化

```bash
# 啟用低記憶體使用模式（預設）
export LOW_CPU_MEM_USAGE=1

# 控制PyTorch執行緒數量（0=自動）
export TORCH_NUM_THREADS=4  # 建議設置為CPU核心數或更少
```

### 2.4 模型配置

```bash
# 自訂模型路徑
export MODEL_PATH=./models/qwen/Qwen2.5-0.5B-Instruct

# 輸入截斷控制
export MAX_INPUT_TOKENS=2048
export MAX_INPUT_CHARS=12000
```

## 3. 啟動優化的服務

配置環境變數後，正常啟動服務：

```bash
python -m uvicorn slm_server:app --host 127.0.0.1 --port 8001
```

啟動日誌會顯示已啟用的優化功能及其配置。

## 4. 監控與診斷

服務現在提供更詳細的日誌信息，包括：
- 模型載入和卸載事件
- 記憶體使用警告和錯誤
- 模型量化狀態

可以通過以下指標了解系統狀態：
- `/metrics` 端點查看請求和錯誤統計
- 伺服器日誌中的記憶體相關警告

## 5. 配置建議

### 5.1 低記憶體設備（<8GB RAM）

```bash
export MODEL_QUANTIZATION=4bit
export MAX_INPUT_TOKENS=1024
export MAX_INPUT_CHARS=6000
export MODEL_IDLE_TIMEOUT=1800  # 30分鐘後卸載
export TORCH_NUM_THREADS=2
```

### 5.2 中等記憶體設備（8-16GB RAM）

```bash
export MODEL_QUANTIZATION=8bit
export MAX_INPUT_TOKENS=2048
export MAX_INPUT_CHARS=12000
export MODEL_IDLE_TIMEOUT=3600  # 1小時後卸載
export TORCH_NUM_THREADS=4
```

### 5.3 高記憶體設備（>16GB RAM）

```bash
export MODEL_QUANTIZATION=none
export MAX_INPUT_TOKENS=4096
export MAX_INPUT_CHARS=24000
export MODEL_IDLE_TIMEOUT=7200  # 2小時後卸載
export TORCH_NUM_THREADS=0  # 自動
```

## 6. 故障排除

如果遇到「系統虛擬記憶體不足」錯誤：

1. 嘗試啟用更積極的量化（4bit）
2. 減小 MAX_INPUT_TOKENS 和 MAX_INPUT_CHARS 值
3. 確保沒有其他大型應用程序佔用記憶體
4. 在 Windows 上，增加頁面文件大小
5. 檢查是否已正確安裝 bitsandbytes 套件