#!/usr/bin/env bash
# start_llama_cpp.sh - 啟動使用 llama.cpp (GGUF) 後端的本地 API 服務。
# 需求: pip install -e .[llama]  或  pip install llama-cpp-python

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-models/qwen/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}
PORT=${PORT:-8001}
CTX=${LLAMA_CTX:-${MAX_INPUT_TOKENS:-1024}}
THREADS=${LLAMA_THREADS:-4}

if [ ! -f "$MODEL_PATH" ]; then
  echo "[警告] 模型檔不存在: $MODEL_PATH" >&2
fi

export MODEL_BACKEND=llama.cpp
export MODEL_PATH
export COMPILE_MODEL=0
export MAX_INPUT_TOKENS=$CTX
export LLAMA_CTX=$CTX
export LLAMA_THREADS=$THREADS
export LOW_CPU_MEM_USAGE=1
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "=== 啟動 llama.cpp 後端 ==="
echo "MODEL_PATH       = $MODEL_PATH"
echo "LLAMA_CTX        = $LLAMA_CTX"
echo "LLAMA_THREADS    = $LLAMA_THREADS"
echo "MAX_INPUT_TOKENS = $MAX_INPUT_TOKENS"

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

echo "啟動 uvicorn 中..."
python -m uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port "$PORT"
