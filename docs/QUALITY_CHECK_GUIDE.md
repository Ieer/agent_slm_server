# Y26HKX 服務品質檢查與說明文檔

> 版本: 2025-09-14
> 適用範圍: `slm_server.py` (完整版) 與 `slm_server_simple.py`

---
## 1. 架構概覽

組件 | 角色 | 重點
---|---|---
`slm_server.py` | 完整 OpenAI 相容 API（健康/模型/聊天/嵌入/metrics/自動卸載） | 帶有記憶體監控與閒置卸載
`slm_server_simple.py` | 極簡版（主要聊天/健康） | 用於在受限環境快速啟動
`chat_backends.py` | 後端抽象 (Transformers / llama.cpp / 可擴充) | 統一 `generate()` 介面
`qwenchat.py` | Transformers 版 Qwen 封裝 | 專注對話生成與 token 統計
`memory_monitor.py` | 記憶體監控與洩漏啟發式 | `/health/memory` 與 `/health/detailed` 使用
`performance_config.py` | 自動化效能/資源策略參數 | 透過 ENV 與程式常量輸出

---
## 2. 目前品質狀態摘要

領域 | 狀態 | 備註
---|---|---
型別註解 | 主要模組已現代化 (PEP 585 / `| None`) | `slm_server.py` 已移除 `List/Dict/Optional`
錯誤處理 | OSError 1455 預防性處理 (返回 503 + 建議) | 尚未統一所有錯誤 envelope 結構
記憶體管理 | 支援主模型與 embedding 閒置卸載 | 閾值與策略為靜態參數；可改動態調整
多後端 | 支援 `MODEL_BACKEND=transformers/llama.cpp` | llama.cpp 流程為最小骨架，未做 streaming
嵌入 | 動態掃描 + lazy load + 閒置清理 | 無向量快取（視用例可增加 LRU）
Metrics | 基礎計數 + latency (sum/max) | 尚缺 token 聚合、加權平均、錯誤分型更多維度
測試 | 健康與 embeddings 冒煙測試通過 | Chat 端點在低記憶體 Windows 環境需手動調整 Pagefile
文件 | 已有 `docs/quality_report.md` + 本指引 | 可補 FAQ & 環境最佳化指南 (Pagefile/量化)
安全/資源防護 | 有基礎輸入截斷 (token heuristic) | 未加速率限制 / 防 prompt injection / 指令黑名單

---
## 3. 型別與靜態分析建議

推薦設定 (可於 `pyproject.toml` 或 `mypy.ini`)：
```ini
[mypy]
python_version = 3.11
strict = True
warn_unused_ignores = True
warn_redundant_casts = True
warn_unreachable = True
disable_error_code = attr-defined

[tool.ruff]
line-length = 120
select = ["E","F","W","I","N","B","UP","DTZ","TID","RUF"]
ignore = ["E203","E501"]

[tool.ruff.format]
quote-style = "preserve"
```

額外改善點：
- 為 `chat_backends.py` 的 `generate` 回傳建立 TypedDict 以替代裸 dict。
- 引入 `Protocol` 擴展：加入 `supports_stream: bool` 屬性，用於未來 streaming 協商。
- 建立 `ErrorEnvelope` Pydantic 模型，集中錯誤格式。

---
## 4. 錯誤處理統一化建議

現況：
- 初始化後端失敗（記憶體 / Pagefile） => 503 + 建議清單。
- 其他例外 => 500 + 純文字訊息。

建議標準結構：
```json
{
  "error": {
    "code": "BACKEND_LOAD_FAILED",
    "type": "resource",
    "backend": "transformers",
    "message": "模型/後端載入失敗 (內存或分頁檔不足)",
    "suggestions": ["增大 Pagefile 至 >=16GB"],
    "request_id": "..."  
  }
}
```
實作步驟：
1. 新增 `error_models.py` (或保留單檔：放於 `slm_server.py` 頂部) 定義 `ErrorInfo` Pydantic。
2. 包裝 `HTTPException` 與統一建構函式 `raise_error(code, type_, message, suggestions=None, status=500)`。
3. chat 與 embeddings 都使用統一方法。

---
## 5. Metrics 進階方案

指標 | 類型 | 說明
---|---|---
`y26hkx_requests_total` | counter | 全部請求數
`y26hkx_chat_requests_total` | counter | 聊天端點請求數
`y26hkx_embeddings_requests_total` | counter | 嵌入端點請求數
`y26hkx_chat_latency_sum_seconds` | counter | 聊天延遲總和
`y26hkx_chat_latency_max_seconds` | gauge | 歷史最大延遲
`y26hkx_model_loaded_timestamp` | gauge | 模型最近載入時間

建議新增：
- `prompt_tokens_total`, `completion_tokens_total`, `total_tokens_total`
- `backend_load_failures_total`
- `embedding_model_load_seconds` (Histogram/Summary 模式；簡化可紀錄 sum + count)
- `active_embedding_models` (gauge)

實作範例（chat 回傳後）：
```python
_metrics_inc("prompt_tokens_total", out.get("prompt_tokens", 0))
_metrics_inc("completion_tokens_total", out.get("completion_tokens", 0))
_metrics_inc("total_tokens_total", out.get("total_tokens", 0))
```

---
## 6. 記憶體與資源管理檢查清單

項目 | 問題徵兆 | 檢查方式 | 建議行動
---|---|---|---
Pagefile 不足 | OSError 1455 | Windows 事件檢視 / 503 回應 detail | 增加虛擬記憶體 (>= 16GB)
CUDA OOM | torch 報錯 | `nvidia-smi` | 使用 4bit 量化 / 減少 batch
嵌入模型過多 | 記憶體緩慢增加 | `/health/detailed` embedding_models_count | 降低 TTL 或設定最大載入數
主模型閒置未卸載 | 常駐記憶體不回收 | 觀察 unload log 是否缺席 | 確認 `MODEL_UNLOAD_ENABLED=1`
未截斷輸入 | 延遲、OOM | 請求 payload size | 調整 `MAX_INPUT_TOKENS` 或實作角色感知截斷

---
## 7. Streaming 與擴展預留建議

目標：支援 `stream=True` 以 SSE 或 chunk 傳遞 tokens。
最低步驟：
1. 在 `Protocol` 新增可選 `stream_generate(messages, **kwargs) -> Iterator[str]`。
2. Transformers backend 實作使用 `TextIteratorStreamer`。
3. FastAPI 端點改為：
```python
return StreamingResponse(streamer_iter, media_type="text/event-stream")
```
4. 封裝事件格式符合 OpenAI `data: {"id":...,"choices":[{"delta":{"content":"..."}}]}`。

---
## 8. 安全 / 穩定性建議（短期可行）

層面 | 建議
---|---
輸入驗證 | 限制單訊息長度，過長直接截斷並標示 trimmed flag
速率限制 | 加入簡單 IP + 時窗計數 (in-memory dict)
日誌追蹤 | 生成 `request_id` 並於錯誤/metrics 帶上
配置檢測 | 啟動時輸出環境變數摘要與風險（例如 Pagefile 建議值）
備援 | 若主 backend 載入失敗可自動 fallback 到更小 GGUF

---
## 9. 推薦改動優先序 (Roadmap)

階段 | 項目 | 說明
---|---|---
P0 | 錯誤格式統一、token usage metrics | 低風險，高觀測價值
P0 | TypedDict for backend generate | 消除 dict magic keys
P1 | Streaming 支援 | 增強互動性
P1 | Rate limiting + request_id | 基礎穩定與可追蹤性
P2 | Embedding LRU + 快取統計 | 控制記憶體占用
P2 | mypy/ruff CI | 自動化品質門檻
P3 | Fallback backend 策略 | 提升可用性
P3 | 更細粒度的記憶體預估 | 預防載入前崩潰

---
## 10. 開發者操作指南 (Quick Ops)

啟動完整服務：
```bash
uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```
啟動極簡版：
```bash
uvicorn agent_slm_server.slm_server_simple:app --port 8002
```
切換 backend：
```bash
set MODEL_BACKEND=llama.cpp
set MODEL_PATH=./models/qwen/Qwen2.5-0.5B-Instruct-GGUF
```
檢測健康：
```bash
curl http://127.0.0.1:8001/health/detailed
```
查看 metrics：
```bash
curl http://127.0.0.1:8001/metrics
```

---
## 11. 核心稽核檢查腳本範例

```python
from fastapi.testclient import TestClient
import agent_slm_server.slm_server as srv
c = TestClient(srv.app)
assert c.get('/health').status_code == 200
assert 'models' in c.get('/health/detailed').json()
assert any(d['id'].startswith('qwen') for d in c.get('/v1/models').json()['data'])
```

---
## 12. 立即可執行的下一步建議
1. 新增 `metrics`：token usage counters。
2. 建立 `backend_types.py`：`GenerateResult` TypedDict。
3. 實作統一 `error_response()` helper 與 `ErrorInfo`。
4. 撰寫 `STREAMING_PLAN.md` 初稿（列協議事件格式）。

---
## 13. 變更追蹤摘要 (本迭代)
- 導入後端抽象與 multi-backend 支援。
- 將全域模型狀態封裝 (`MODEL_STATE`)。
- Modern typing 全面應用於 server 主要檔。
- 新增 OSError 1455 預防性錯誤建議回傳。
- 嵌入模型重新整理端點與 lazy load 正常運作。

---
## 14. 風險與緩解
風險 | 描述 | 緩解
---|---|---
Windows Pagefile 不足 | 模型初載入失敗 | 啟動前檢查加預警；文件明示建議值
Embedding 過多占用 RAM | 無上限載入 | 加入 LRU 與最大載入數參數
未統一錯誤格式 | 前端處理複雜 | 引入標準 envelope
缺乏 streaming | 長回應體驗差 | 分階段導入 SSE
缺少限流 | 遭濫用導致 OOM | 加 IP/time window 計數

---
## 15. 結論
目前系統已具備穩定的非聊天核心功能（健康檢查、嵌入、模型列舉、基礎 metrics）。建議首先補足觀測與錯誤結構，再推進 streaming 與記憶體使用優化。此文檔可作為後續 PR / CI 對齊基準。

---
*如需我直接幫你落實上述任一改進（例如新增 token metrics 或錯誤封裝），請告訴我優先順序。*
