# Agent SLM Server (原 y26hkx_server) 本地 LLM / OpenAI 相容服務

輕量、可本機離線運行的 Qwen 0.5B + 多 embedding 模型 OpenAI 相容 API 服務。

核心能力：

- Chat / Completion (`/v1/chat/completions`, `/v1/completions`)
- Embeddings (`/v1/embeddings` 自動掃描 `models/embedding/*`)
- 模型列出與重新載入 (`/v1/models`, `/v1/reload-embeddings`)
- 健康檢查與指標 (`/health`, `/metrics`)
- 工具呼叫解析 (多種標記格式)
- 基礎串流生成能力（可擴充 SSE）
- 多後端：`transformers` 與 `llama.cpp` (GGUF) 可切換

完整使用、環境變數、性能調校、故障排除請見：
👉 `USAGE_GUIDE.md`（進階指南）
👉 `docs/memory-optimization-guide.md`（記憶體/量化策略）

## 新增：llama.cpp (GGUF) 後端快速指南

當 `transformers` + 原始 safetensors 內存不足或遇到 Windows 1455（頁面檔過小）時，可切換至 GGUF 量化模型：

```powershell
# 安裝（可選 extras）
pip install -e .[llama]
# 或只裝核心套件
pip install llama-cpp-python

# 啟動（使用提供的 Q4_K_M GGUF 範例）
$env:MODEL_BACKEND="llama.cpp"
$env:MODEL_PATH="models/qwen/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
$env:COMPILE_MODEL="0"
$env:MAX_INPUT_TOKENS="1024"
uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

更簡：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_llama_cpp.ps1
```
或（Linux / macOS / WSL）：

```bash
bash scripts/start_llama_cpp.sh
```
調參（可選）

```powershell
$env:LLAMA_THREADS="6"   # CPU 核心數
$env:LLAMA_CTX="1024"    # 上下文長度 (與 MAX_INPUT_TOKENS 協調)
```

回退 transformers：清除/修改 `MODEL_BACKEND` 或設為 `transformers`。

---

## 目錄結構（核心）

```text
models/
  qwen/Qwen2.5-0.5B-Instruct/   # 主要對話模型
  embedding/                    # 多個 embedding 子資料夾
scripts/                        # 啟動與運維腳本
  start_server.py               # 依賴/模型/環境檢查 + 啟動簡化服務
  start_optimized.(sh|ps1)      # 記憶體偵測 + 量化/限制自動設定
  start_llama_cpp.(sh|ps1)      # GGUF / llama.cpp 後端啟動
slm_server.py                   # 主服務 (若後續遷移可放入 src/)
slm_server_simple.py            # 簡化/低依賴版本
qwenchat.py                     # QwenChatAPI 封裝
memory_monitor.py               # 記憶體監控工具
performance_config.py           # 性能/資源配置
requirements*.txt               # 正逐步被 pyproject.extras 取代
pyproject.toml                  # 套件與 extras 管理
```

 
 
## 安裝方式（建議 Python 3.11）

最小安裝：

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -e .
```

開發環境（lint / type / test）：

```powershell
pip install -e .[dev]
```

量化/優化：

```powershell
pip install -e .[optimized]
```

僅量化（保留 quant）：

```powershell
pip install -e .[quant]
```

組合安裝：

```powershell
pip install -e .[optimized,dev]
```

GPU 版 Torch（CUDA 12.1 範例）：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> 更多依賴與優化說明請見 `USAGE_GUIDE.md`。

 
 
## 啟動

最簡（完整主服務）:

```powershell
uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

簡化版（低依賴）:

```powershell
uvicorn agent_slm_server.slm_server_simple:app --host 127.0.0.1 --port 8001
```

啟動腳本（依賴/模型/環境檢查）：

```powershell
python scripts/start_server.py
```

低記憶體 / 自動量化：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_optimized.ps1
```
或（Linux / macOS / WSL）

```bash
bash scripts/start_optimized.sh
```

健康檢查：`GET http://127.0.0.1:8001/health`

（可選）未來可新增 console entry point：

```toml
[project.scripts]
agent-slm-server = "scripts.start_server:main"
```
安裝後可直接執行：`agent-slm-server`

 
 
## 快速 API 測試（Chat）

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8001/v1/chat/completions -Method Post -Body (@{
  model = 'qwen2.5-0.5b-instruct'
  messages = @(@{role='user'; content='說一句早安'})
} | ConvertTo-Json -Depth 5) -ContentType 'application/json'
```
> 更多完整 JSON 範例與工具呼叫 / embeddings / metrics：見 `USAGE_GUIDE.md`

 
## 目錄 & 文檔導覽

- 進階指南：`USAGE_GUIDE.md`
- 記憶體 / 量化：`docs/memory-optimization-guide.md`
- 測試：`tests/`（可執行 `pytest -q`）
- 啟動腳本：`scripts/`
- 模型資源：`models/`

## 測試

```powershell
pytest -q            # 單元/整合測試
```

> 可擴充更多 API 覆蓋與性能基準測試，詳見 `USAGE_GUIDE.md`。

 
## Metrics & 監控

支持 Prometheus `/metrics`；詳細指標名稱與解讀：見 `USAGE_GUIDE.md`。

 
## 環境變數

常用：MODEL_QUANTIZATION / LOW_CPU_MEM_USAGE / MODEL_UNLOAD_ENABLED / MAX_INPUT_TOKENS ...

完整列表與建議值：`USAGE_GUIDE.md` + `docs/memory-optimization-guide.md`。

 
## 進階主題
輸入截斷策略 / 串流 SSE / Stop Tokens / 性能調優：見 `USAGE_GUIDE.md`。

## 開發快速測試

```powershell
python qwenchat.py             # 對話測試
python scripts/start_server.py # 啟動（含檢查）
pytest -q                      # 測試
```

## 遷移說明 (Rename Migration)

專案已從 `y26hkx-llm-server` / `y26hkx_server` 重命名為 `agent_slm_server`。

### 變更重點

| 項目 | 舊 | 新 |
|------|----|----|
| 套件名稱 | y26hkx_server | agent_slm_server |
| 發佈 distribution | y26hkx-llm-server | agent_slm_server |
| uvicorn 啟動 | uvicorn y26hkx_server.slm_server:app | uvicorn agent_slm_server.slm_server:app |
| Metrics 前綴 | y26hkx_ | agent_slm_ |

### 升級步驟

```powershell
pip uninstall -y y26hkx-llm-server  # 若存在
pip install -e .[dev]
uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port 8001 --reload
```

### 自訂腳本更新

請搜尋 `y26hkx_server` 並替換為 `agent_slm_server`。

### 兼容性提示

不再保留舊模組別名；若仍有歷史程式依賴請同步更新。

## Roadmap

- [ ] SSE 串流輸出（OpenAI 事件格式）
- [ ] 更精準 token 累積截斷策略（對話滑動視窗）
- [ ] 環境變數集中管理（集中 config 模組）
- [ ] Embeddings：normalize 選項、批次大小控制
- [ ] 工具呼叫在串流模式下的逐步輸出
- [ ] console entry point (`agent-slm-server`)
- [ ] 完整 API 文件自動生成 (mkdocstrings)

## 授權

僅供本地研究測試，請遵循各模型原始授權（Qwen / sentence-transformers）。

---
若需新增功能或優化，請在議題中提出或直接修改後提交 PR。
