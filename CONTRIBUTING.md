# 貢獻指南

感謝你對本地 LLM 服務專案的興趣！以下說明協助你快速進行開發、提交修改與保持品質一致。

## 開發環境設定

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements-dev.txt
```

或使用 `pyproject.toml`:

```powershell
pip install -e .[dev,quant]
```

## 主要資料夾與檔案

| 路徑 | 說明 |
|------|------|
| `slm_server.py` | FastAPI 主應用（OpenAI 相容端點） |
| `qwenchat.py` | Qwen 模型封裝、訊息截斷與生成邏輯 |
| `memory_monitor.py` | 記憶體監控與分析 |
| `performance_config.py` | 性能參數與自動調整方法 |
| `models/` | 本地模型與 embedding 權重（不應提交大型權重） |
| `docs/` | 深入指南與報告 |

## 分支策略 (建議)

- `main`：穩定、可執行版本
- `develop`：整合測試中版本
- Feature 分支：`feat/<short-desc>`
- Bug 修正：`fix/<issue-id>-<short>`
- 文件：`docs/<topic>`

## Commit 規範 (Conventional Commits)

```text
feat: 新增 SSE 串流支援
fix(memory): 修正模型卸載後記憶體未回收問題
docs: 補充 embeddings 端點說明
refactor(qwen): 抽離訊息截斷函式
perf: 降低初始化延遲
chore: 更新依賴與工具設定
```

可選範本：

```text
<type>(<scope>): <subject>

<body - What & Why>

<footer - Breaking change / Closes #Issue>
```

## 程式風格與工具

```powershell
ruff check .
black .
isort .
mypy .
pytest
```

或一次性檢查：

```powershell
ruff check . ; black . ; isort . ; mypy . ; pytest
```

## 測試 (建議新增)

- 單元測試放置於 `tests/`
- 命名：`test_*.py`
- 可使用 `pytest-asyncio` 測試 async 路由

## 新增端點時

1. 在 `slm_server.py` 實作路由與 Pydantic 模型
2. 更新 `README.md` 中的 API 章節
3. 若涉及環境變數：更新文件與 `.env.example`
4. 增加對應測試（若可能）

## 文件更新

- 重大功能：更新 `README.md`
- 深度性能或記憶體分析：放入 `docs/`

## 發佈版本 (建議流程)

1. `bumpversion` (可新增工具) 或手動更新 `pyproject.toml` 版本
2. 建立 Tag：`git tag -a v0.1.0 -m "First release"`
3. 推送：`git push origin main --tags`

## 問題回報 (Issues)

- 描述環境：OS / Python 版本 / CPU or GPU
- 重現步驟（必要）
- 預期 vs. 實際行為
- 日誌或錯誤截圖

## 安全

本專案無遠端推送機密資訊，請勿提交任何私密憑證或專有數據。

---

歡迎提出建議與 PR，一起讓本地 LLM 部署更高效！
