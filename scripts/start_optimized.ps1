# 已移至 scripts/：啟動腳本：啟用模型量化和記憶體優化功能 (PowerShell)

Write-Host "Y26HKX 優化啟動腳本" -ForegroundColor Blue
Write-Host "啟用記憶體優化和模型量化..." -ForegroundColor Yellow

$totalMem = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property capacity -Sum).sum/1MB
Write-Host "系統總記憶體: $totalMem MB" -ForegroundColor Blue

if ($totalMem -lt 8192) {
    Write-Host "低記憶體 (<8GB) 4-bit" -ForegroundColor Yellow
    $env:MODEL_QUANTIZATION = "4bit"
    $env:MAX_INPUT_TOKENS = 1024
    $env:MAX_INPUT_CHARS = 6000
    $env:MODEL_IDLE_TIMEOUT = 1800
    $env:TORCH_NUM_THREADS = 2
    $env:LOW_CPU_MEM_USAGE = 1
}
elseif ($totalMem -lt 16384) {
    Write-Host "中等記憶體 (8-16GB) 8-bit" -ForegroundColor Yellow
    $env:MODEL_QUANTIZATION = "8bit"
    $env:MAX_INPUT_TOKENS = 2048
    $env:MAX_INPUT_CHARS = 12000
    $env:MODEL_IDLE_TIMEOUT = 3600
    $env:TORCH_NUM_THREADS = 4
    $env:LOW_CPU_MEM_USAGE = 1
}
else {
    Write-Host "高記憶體 (>16GB) 預設" -ForegroundColor Green
    $env:MODEL_QUANTIZATION = "none"
    $env:MAX_INPUT_TOKENS = 4096
    $env:MAX_INPUT_CHARS = 24000
    $env:MODEL_IDLE_TIMEOUT = 7200
    $env:TORCH_NUM_THREADS = 0
    $env:LOW_CPU_MEM_USAGE = 0
}

$env:MODEL_UNLOAD_ENABLED = 1

Write-Host "設定:" -ForegroundColor Blue
Write-Host "  量化: $($env:MODEL_QUANTIZATION)" -ForegroundColor Green
Write-Host "  Tokens: $($env:MAX_INPUT_TOKENS)" -ForegroundColor Green
Write-Host "  Chars: $($env:MAX_INPUT_CHARS)" -ForegroundColor Green
Write-Host "  Idle: $($env:MODEL_IDLE_TIMEOUT)s" -ForegroundColor Green
Write-Host "  Threads: $($env:TORCH_NUM_THREADS)" -ForegroundColor Green
Write-Host "  LowMem: $($env:LOW_CPU_MEM_USAGE)" -ForegroundColor Green

if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
    Write-Host "已啟用虛擬環境" -ForegroundColor Green
}

Write-Host "啟動服務..." -ForegroundColor Yellow
python -m uvicorn agent_slm_server.slm_server_simple:app --host 127.0.0.1 --port 8001
