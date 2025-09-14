# 從 y26hkx_server.start_server 重命名至 agent_slm_server.start_server
"""伺服器啟動與檢查腳本 (程式化版)
使用方式: python -m agent_slm_server.start_server
"""
import os, sys, time, subprocess
import requests

REQUIRED_PACKAGES = ["fastapi","uvicorn","transformers","torch","pydantic","psutil"]
MODEL_PATHS = ["./models/qwen/Qwen2.5-0.5B-Instruct","./models/embedding/paraphrase-MiniLM-L6-v2"]
RECOMMENDED_ENV = {"LOW_CPU_MEM_USAGE": "1","MODEL_UNLOAD_ENABLED": "1","MODEL_IDLE_TIMEOUT": "1800","MAX_INPUT_TOKENS": "1024","LOG_LEVEL": "INFO"}

def check_dependencies() -> bool:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"缺少依賴: {', '.join(missing)}\n請執行: pip install {' '.join(missing)}")
        return False
    print("依賴檢查通過")
    return True

def check_models() -> None:
    for path in MODEL_PATHS:
        if not os.path.exists(path):
            print(f"⚠ 模型路徑不存在: {path}")
        else:
            print(f"✅ 模型存在: {path}")

def set_environment_variables() -> None:
    for k, v in RECOMMENDED_ENV.items():
        if k not in os.environ:
            os.environ[k] = v
            print(f"設定環境變數: {k}={v}")

HTTP_OK = 200

def start_server() -> subprocess.Popen | None:
    print("啟動簡化版伺服器...")
    cmd = [sys.executable, "-m", "uvicorn", "agent_slm_server.slm_server_simple:app","--host","127.0.0.1","--port","8001","--reload"]
    try:
        proc = subprocess.Popen(cmd)
        for i in range(30):
            try:
                r = requests.get("http://127.0.0.1:8001/health", timeout=2)
                if r.status_code == HTTP_OK:
                    print("✅ 服務啟動成功 http://127.0.0.1:8001")
                    return proc
            except Exception:
                time.sleep(1)
                print(f"啟動中... ({i+1}/30)")
        print("❌ 服務啟動逾時")
        proc.terminate()
        return None
    except Exception as e:  # noqa: BLE001
        print(f"啟動失敗: {e}")
        return None

def main() -> None:
    print("=== 啟動檢查 ===")
    if not check_dependencies():
        sys.exit(1)
    check_models()
    set_environment_variables()
    proc = start_server()
    if not proc:
        sys.exit(1)
    try:
        print("按 Ctrl+C 停止服務...")
        proc.wait()
    except KeyboardInterrupt:
        print("停止中...")
        proc.terminate(); proc.wait()
        print("已停止")

if __name__ == "__main__":
    main()
