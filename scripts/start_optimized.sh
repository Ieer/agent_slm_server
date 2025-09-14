#!/bin/bash
# 已移至 scripts/ ：啟動腳本：啟用模型量化和記憶體優化功能

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Y26HKX 優化啟動腳本${NC}"
echo -e "${YELLOW}啟用記憶體優化和模型量化...${NC}"

if [ "$(uname)" == "Darwin" ]; then
    TOTAL_MEM=$(sysctl hw.memsize | awk '{print $2/1024/1024}')
elif [ "$(uname)" == "Linux" ]; then
    TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
else
    TOTAL_MEM=8192
    if command -v powershell.exe &> /dev/null; then
        TOTAL_MEM=$(powershell.exe -command "(Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property capacity -Sum).sum/1MB" | tr -d '\r')
    fi
fi

echo -e "${BLUE}系統總記憶體: ${TOTAL_MEM} MB${NC}"

if [ "$TOTAL_MEM" -lt 8192 ]; then
    echo -e "${YELLOW}低記憶體 (<8GB) 啟用 4-bit${NC}"
    export MODEL_QUANTIZATION="4bit"
    export MAX_INPUT_TOKENS=1024
    export MAX_INPUT_CHARS=6000
    export MODEL_IDLE_TIMEOUT=1800
    export TORCH_NUM_THREADS=2
    export LOW_CPU_MEM_USAGE=1
elif [ "$TOTAL_MEM" -lt 16384 ]; then
    echo -e "${YELLOW}中等記憶體 (8-16GB) 啟用 8-bit${NC}"
    export MODEL_QUANTIZATION="8bit"
    export MAX_INPUT_TOKENS=2048
    export MAX_INPUT_CHARS=12000
    export MODEL_IDLE_TIMEOUT=3600
    export TORCH_NUM_THREADS=4
    export LOW_CPU_MEM_USAGE=1
else
    echo -e "${GREEN}高記憶體 (>16GB) 使用預設${NC}"
    export MODEL_QUANTIZATION="none"
    export MAX_INPUT_TOKENS=4096
    export MAX_INPUT_CHARS=24000
    export MODEL_IDLE_TIMEOUT=7200
    export TORCH_NUM_THREADS=0
    export LOW_CPU_MEM_USAGE=0
fi

export MODEL_UNLOAD_ENABLED=1

echo -e "${BLUE}設定:${NC}"
echo -e "  量化: ${GREEN}${MODEL_QUANTIZATION}${NC}"
echo -e "  Tokens: ${GREEN}${MAX_INPUT_TOKENS}${NC}"
echo -e "  Chars: ${GREEN}${MAX_INPUT_CHARS}${NC}"
echo -e "  Idle: ${GREEN}${MODEL_IDLE_TIMEOUT}s${NC}"
echo -e "  Threads: ${GREEN}${TORCH_NUM_THREADS}${NC}"
echo -e "  LowMem: ${GREEN}${LOW_CPU_MEM_USAGE}${NC}"

echo -e "${YELLOW}啟動服務...${NC}"
python -m uvicorn agent_slm_server.slm_server_simple:app --host 127.0.0.1 --port 8001
