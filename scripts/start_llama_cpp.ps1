<#
 start_llama_cpp.ps1
 啟動使用 llama.cpp (GGUF) 後端的本地 API 服務。
 需求：pip install -e .[llama]  或  pip install llama-cpp-python
#>
param(
    [string]$ModelPath = "models/qwen/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    [int]$Port = 8001,
    [int]$Ctx = 1024,
    [int]$Threads = 4,
    [switch]$Verbose
)

if (-not (Test-Path $ModelPath)) {
    Write-Host "[警告] 指定的 GGUF 模型檔不存在: $ModelPath" -ForegroundColor Yellow
}

$env:MODEL_BACKEND = "llama.cpp"
$env:MODEL_PATH    = $ModelPath
$env:COMPILE_MODEL = "0"              # llama.cpp 不使用 torch.compile
$env:MAX_INPUT_TOKENS = "$Ctx"        # 與 llama-cpp n_ctx 對齊
$env:LLAMA_CTX = "$Ctx"
$env:LLAMA_THREADS = "$Threads"
$env:LOW_CPU_MEM_USAGE = "1"
$env:LOG_LEVEL = if ($Verbose) { "INFO" } else { $env:LOG_LEVEL -or "INFO" }

Write-Host "=== 啟動 llama.cpp 後端 ===" -ForegroundColor Cyan
Write-Host "MODEL_PATH       = $env:MODEL_PATH"
Write-Host "LLAMA_CTX        = $env:LLAMA_CTX"
Write-Host "LLAMA_THREADS    = $env:LLAMA_THREADS"
Write-Host "MAX_INPUT_TOKENS = $env:MAX_INPUT_TOKENS"

# 簡單依賴檢查
try {
    python - <<'PY'
import importlib, sys
for m in ["llama_cpp"]:
    try:
        importlib.import_module(m)
    except Exception as e:
        print(f"[缺少依賴] {m}: {e}")
        sys.exit(1)
print("[依賴檢查] OK")
PY
} catch {
    Write-Host "依賴檢查失敗，請先安裝 llama-cpp-python" -ForegroundColor Red
    exit 1
}

Write-Host "啟動 uvicorn 中..." -ForegroundColor Green
python -m uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port $Port
