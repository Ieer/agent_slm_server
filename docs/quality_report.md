# 專案品質檢查與改進建議報告

生成時間: 2025-09-14

---
## 1. 整體狀態概覽

| 面向 | 現況 | 風險等級 | 建議優先順序 |
|------|------|----------|--------------|
| 目錄佈局 | 已改為 `src/` 套件化，最小侵入 | 低 | 已完成 |
| 主服務完整性 | `slm_server.py` 可能未完全恢復原全部端點 | 中 | 視需求回填 |
| 簡化服務 | `slm_server_simple.py` 可用；聊天受模型載入失敗阻擋 | 中 | 量化 / Pagefile |
| 模型載入 | Windows Page File 導致 OSError 1455 | 高 | 已加 503 防護；需環境調整 |
| 型別現代化 | 部分（`qwenchat` 等）完成；尚未一致 | 中 | 漸進推進 |
| 測試 | 有腳本型測試；缺少斷言式自動化 | 中 | 引入 pytest 斷言 |
| 錯誤回應一致性 | 部分 500 / 503；格式不統一 | 中 | 統一錯誤結構 |
| 可觀測性 | `/metrics` 基礎；缺少耗時/載入統計 | 中 | 增加自訂指標 |
| 資源管理 | 嵌入快取無過期 / LRU | 低 | 增加回收策略 |

---
## 2. Lint 熱點彙總

類型 | 具體熱點 | 說明 | 建議
-----|---------|------|-----
Magic Numbers | 16 (截斷閾值)、2 (tensor.dim) | 可讀性差 | 抽常數 `MIN_TRUNCATE_REMAIN`, `TENSOR_2D`
過長參數列表 | `QwenChatAPI.__init__`, `chat_messages` | 維護成本高 | 導入 dataclass 聚合
分支過多 | `_truncate_messages_role_aware` | 複雜度提升 | 拆子函式：token 計算 / 截斷
動態匯入 | 內部 import BitsAndBytesConfig | Lint 警告 | 提前到頂層 + try/except
環境變數散落 | 多處 `os.getenv` | 重複樣板 | 集中 config 模組
異常鏈缺失 | 部分 raise 無 `from e` | 追蹤不便 | 漸進補全
`dict()` 用法 | mix 字面量 / dict() | 風格不一 | 全用 `{}` 字面量
未標註全域型別 | `qwen`, `embedding_model_cache` | 可讀性 | 明確標註型別

---
## 3. 型別飽和（Typing Saturation）策略

階段 | 目標 | 範圍 | 操作建議
------|------|------|----------
Phase 1 | 移除舊 `List/Dict/Optional` | 已在核心部分 | 維持風格一致
Phase 2 | Public API datamodel 嚴格 | FastAPI Pydantic Models | `Config: extra = 'forbid'`
Phase 3 | 內部函式返回值明確 | `_truncate_*`, encode | 加回傳型別註解
Phase 4 | 導入 mypy 嚴格模式 | 全專案 | 增 `mypy.ini`
Phase 5 | 型別驅動重構 | config / model manager | Protocol + dataclass 抽象

建議 `mypy.ini` 範例：
```
[mypy]
python_version = 3.11
warn_unused_ignores = True
warn_redundant_casts = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
no_implicit_optional = True
strict_optional = True
```

---
## 4. 結構優化建議（不拆檔前提）

主題 | 現況 | 不拆檔方案 | 若允許新增模組
-----|------|-----------|--------------
模型載入 | Handler 直接處理 | 內部新增 `_LazyModelHolder` 類 | `model_manager.py`
嵌入快取 | dict + 無回收 | 加 idle 時間掃描 + 上限 | `embedding_cache.py`
錯誤格式 | 多樣 | 寫 `api_error()` 工具函式 | `errors.py`
可觀測性 | 指標少 | `/metrics` 延伸 | `metrics.py`
配置 | 分散 ENV | 建 `ENV_KEYS` + loop | Pydantic BaseSettings
後端替換 | 僅 HF 模型 | `backend` 參數預留 | `backends/gguf_backend.py`

---
## 5. OSError 1455（Page File）處理策略

層次 | 說明 | 代價 | 收益
-----|------|------|-----
環境調整 | Windows Pagefile 扩大 | 無代碼改動 | 立即解決載入中斷
量化 | 安裝 `bitsandbytes` + `MODEL_QUANTIZATION=4bit` | 額外依賴 | 降內存峰值
替換權重 | GGUF + llama.cpp | 推理路徑改寫 | 內存更低
延遲/分段載入 | partial streaming | 複雜 | 適用極限環境
失敗回退 | 503 + hints (已加) | 已完成 | 使用者體驗改善

---
## 6. 建議新增 Prometheus 指標

指標 | 類型 | 說明
-----|------|-----
`model_load_attempts_total` | counter | 帶 status 標籤
`model_load_duration_seconds` | histogram | 載入耗時
`embedding_encode_duration_seconds` | histogram | encode 耗時
`chat_generation_tokens` | counter | 生成 token 數
`chat_request_duration_seconds` | histogram | 請求耗時
`memory_available_gb` | gauge | 可用記憶體
`embedding_cache_entries` | gauge | cache 大小
`model_loaded` | gauge | 1/0

---
## 7. 統一錯誤回應格式建議

標準格式：
```json
{
  "error_code": "MODEL_LOAD_FAILED",
  "message": "模型載入失敗 (可能內存/分頁檔不足)",
  "original_error": "...",
  "suggestions": ["..."],
  "timestamp": 1730000000.123,
  "request_id": "..."
}
```
工具函式範例：
```python
def api_error(code: str, message: str, *, status: int = 500, **extra):
    body = {"error_code": code, "message": message, "timestamp": time.time()}
    if extra: body.update(extra)
    raise HTTPException(status_code=status, detail=body)
```

---
## 8. 可立即執行增量任務建議

優先 | 任務 | 難度 | 說明
-----|------|------|-----
高 | Pagefile / 量化啟用說明補入 README | 低 | 降低再現 1455
高 | `slm_server_simple.py` 型別現代化 | 低 | 與現有風格統一
中 | 統一錯誤格式 | 中 | 增可預測性
中 | Embedding cache 清理機制 | 低 | 長時運行穩定性
中 | Dataclass 聚合生成設定 | 中 | 降參數長度
低 | Prometheus 指標 | 中 | 觀測改善
低 | mypy 導入 | 中 | 長期收益

---
## 9. 實用 Snippet 彙整

(1) 生成設定 dataclass：
```python
@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
```

(2) 統一錯誤：
```python
def api_error(code: str, message: str, *, status: int = 500, **extra):
    body = {"error_code": code, "message": message, "timestamp": time.time()}
    if extra: body.update(extra)
    raise HTTPException(status_code=status, detail=body)
```

(3) 簡易嵌入快取清理：
```python
MAX_EMBEDDING_CACHE = 3
EMBEDDING_IDLE_SECONDS = 1800

def prune_embedding_cache():
    now = time.time()
    for mid, last_used in list(_embedding_last_used_times.items()):
        if now - last_used > EMBEDDING_IDLE_SECONDS:
            embedding_model_cache.pop(mid, None)
            _embedding_last_used_times.pop(mid, None)
    if len(embedding_model_cache) > MAX_EMBEDDING_CACHE:
        ordered = sorted(_embedding_last_used_times.items(), key=lambda x: x[1])
        for mid, _ in ordered[:-MAX_EMBEDDING_CACHE]:
            embedding_model_cache.pop(mid, None)
            _embedding_last_used_times.pop(mid, None)
```

---
## 10. 建議執行順序（Roadmap）
1. 調整系統 Pagefile 或啟用量化（解除功能阻擋）
2. `slm_server_simple.py` 型別現代化 + cache 清理
3. 統一錯誤格式 + 新增基礎指標（模型載入、請求耗時）
4. 視需求回填 `slm_server.py` 全端點（若仍要完整版）
5. 導入 mypy + 嚴格 lint
6. 規劃 GGUF / 多後端抽象

---
## 11. 總結
基礎結構已穩定遷移到 `src` 佈局；主要功能瓶頸集中在模型載入內存限制。建議先解決環境層面問題，再進行型別/錯誤格式/觀測性加固，最後視業務需求擴展後端彈性與高階優化。

若需要，我可以接著：
- (A) 直接套用 embedding cache 清理
- (B) 開始 `slm_server_simple.py` 型別現代化
- (C) 寫統一錯誤工具並替換現有 raise

回覆 A / B / C 或自訂需求即可。
