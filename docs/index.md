# Y26HKX 本地 LLM 服務文件站

> 輕量、可本機離線運行的 Qwen 0.5B + 多 embedding 模型，OpenAI 相容 API。這裡匯總快速開始、環境設定、性能 / 記憶體優化與品質檢查內容。

## 功能總覽

- Chat / Completion: `/v1/chat/completions`, `/v1/completions`
- Embeddings: `/v1/embeddings`（自動掃描 `models/embedding/*`）
- 模型列表 / 重新載入：`/v1/models`, `/v1/reload-embeddings`
- 健康檢查 / 指標：`/health`, `/metrics`
- 多後端：`transformers` 與 `llama.cpp` (GGUF)
- 記憶體優化：量化 / 模型卸載 / 低記憶體模式

## 快速開始 (節選)

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -e .[optimized]
python -m uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

聊天測試：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8001/v1/chat/completions -Method Post -Body (@{
  model = 'qwen2.5-0.5b-instruct'
  messages = @(@{role='user'; content='說一句早安'})
} | ConvertTo-Json -Depth 5) -ContentType 'application/json'
```

## llama.cpp (GGUF) 後端

```powershell
pip install -e .[llama]
$env:MODEL_BACKEND="llama.cpp"
$env:MODEL_PATH="models/qwen/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

## 文檔導覽

| 分類 | 內容 |
|------|------|
| 快速開始 | `README.md` |
| 進階使用 / API | `USAGE_GUIDE.md` |
| 記憶體 / 量化 | `docs/memory-optimization-guide.md` |
| 性能優化完整報告 | `docs/OPTIMIZATION_REPORT.md` |
| 優化摘要 | `docs/optimization_summary.md` |
| 品質檢查 / 改進建議 | `docs/quality_report.md` |

## 常用環境變數（摘要）

| 變數 | 用途 | 範例 |
|------|------|------|
| MODEL_BACKEND | 選擇後端 | transformers / llama.cpp |
| MODEL_PATH | 模型路徑 | models/qwen/... |
| MODEL_QUANTIZATION | 量化模式 | 4bit / 8bit / none |
| LOW_CPU_MEM_USAGE | 低記憶體載入 | 1 |
| MODEL_UNLOAD_ENABLED | 啟用模型閒置卸載 | 1 |
| MAX_INPUT_TOKENS | 輸入 token 限制 | 1024 |
| TORCH_NUM_THREADS | CPU 執行緒 | 4 |

更多細節請進入各章節。

## 品質 / Roadmap 摘要

- 待辦：SSE 串流、截斷策略改良、統一錯誤格式、embedding cache 管理
- 指標建議：模型載入次數、生成耗時、token 數、記憶體使用率

## 參與貢獻

欢迎提交 Issue / PR：改進錯誤處理、指標、文件與性能。

---

文件站使用 MkDocs Material 生成；如需離線瀏覽，可執行：

```powershell
pip install -e .[docs]
mkdocs serve
```
