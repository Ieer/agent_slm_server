# 快速開始 (節選自根目錄 README)

> 此頁為根目錄 `README.md` 摘要，確保在 MkDocs 站點內可正常瀏覽。完整內容請查看原始檔案或專案首頁。

## 安裝

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -e .
```

## 開發環境

```powershell
pip install -e .[dev]
```

## 啟動服務

```powershell
python -m uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

## llama.cpp 後端

```powershell
pip install -e .[llama]
$env:MODEL_BACKEND="llama.cpp"
$env:MODEL_PATH="models/qwen/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port 8001
```

## 測試

```powershell
pytest -q
```

---

此頁僅作為文件站引用。
