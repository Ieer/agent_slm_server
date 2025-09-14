"""
優化後的 SLM 伺服器啟動腳本 (已移至 scripts/)
提供: 依賴檢查 / 模型檢查 / 推薦環境變數 / 啟動簡化版伺服器
原始位置: 專案根目錄
"""

import os
import subprocess
import sys
import time

import requests

REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "transformers",
    "torch",
    "pydantic",
    "psutil",
]

MODEL_PATHS = [
    "./models/qwen/Qwen2.5-0.5B-Instruct",
    "./models/embedding/paraphrase-MiniLM-L6-v2",
]

RECOMMENDED_ENV = {
    "LOW_CPU_MEM_USAGE": "1",
    "MODEL_UNLOAD_ENABLED": "1",
    "MODEL_IDLE_TIMEOUT": "1800",
    "MAX_INPUT_TOKENS": "1024",
    "LOG_LEVEL": "INFO",
}

def check_dependencies() -> bool:
    """檢查必要套件是否存在。"""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ 缺少依賴: {', '.join(missing)}")
        print("請執行: pip install " + " ".join(missing))
        return False
    print("✅ 依賴檢查通過")
    return True

def check_models() -> None:
    """檢查模型路徑是否存在。"""
    for path in MODEL_PATHS:
        if not os.path.exists(path):
            print(f"⚠️ 模型路徑不存在: {path}")
        else:
            print(f"✅ 模型存在: {path}")

def set_environment_variables() -> None:
    """設定建議環境變數 (若未被覆寫)。"""
    for k, v in RECOMMENDED_ENV.items():
        if k not in os.environ:
            os.environ[k] = v
            print(f"設定環境變數: {k}={v}")

HTTP_OK = 200

def start_server() -> subprocess.Popen | None:
    """啟動 Uvicorn 伺服器 (使用簡化 app)。"""
    print("\n🚀 啟動 SLM 伺服器...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "slm_server_simple:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--reload",
    ]
    try:
        proc = subprocess.Popen(cmd)
        print("等待服務啟動...")
        for i in range(30):
            try:
                r = requests.get("http://127.0.0.1:8001/health", timeout=2)
                if r.status_code == HTTP_OK:
                    print("✅ 服務啟動成功!")
                    print("API: http://127.0.0.1:8001")
                    print("Health: http://127.0.0.1:8001/health")
                    print("Docs: http://127.0.0.1:8001/docs")
                    return proc
            except Exception:
                time.sleep(1)
                print(f"啟動中... ({i+1}/30)")
        print("❌ 服務啟動逾時")
        proc.terminate()
        return None
    except Exception as e:  # noqa: BLE001
        print(f"❌ 啟動失敗: {e}")
        return None

def main() -> None:
    print("=== SLM 伺服器啟動檢查 ===")
    if not check_dependencies():
        sys.exit(1)
    check_models()
    set_environment_variables()
    proc = start_server()
    if not proc:
        print("啟動失敗，請檢查錯誤輸出")
        sys.exit(1)
    try:
        print("\n按 Ctrl+C 停止服務...")
        proc.wait()
    except KeyboardInterrupt:  # noqa: PIE786
        print("\n🛑 停止服務中...")
        proc.terminate()
        proc.wait()
        print("服務已停止")

if __name__ == "__main__":  # pragma: no cover
    main()
