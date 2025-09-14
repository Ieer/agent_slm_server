import os
import importlib.util
import pytest

# 測試目標: llama.cpp 後端載入與基本 generate 輸出結構
# 條件:
#  1. 安裝了 llama-cpp-python
#  2. GGUF 模型檔案存在 (默認專案提供的 Q4_K_M)
# 否則自動 skip，保持 CI 穩定

MODEL_PATH_ENV = "models/qwen/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

@pytest.fixture(scope="session")
def has_llama_cpp() -> bool:
    try:
        importlib.import_module("llama_cpp")
        return True
    except Exception:
        return False

@pytest.fixture(scope="session")
def gguf_path() -> str:
    return os.getenv("MODEL_PATH", MODEL_PATH_ENV)

@pytest.mark.skipif(not os.path.exists(MODEL_PATH_ENV), reason="GGUF 模型檔不存在，跳過 llama.cpp backend 測試")
@pytest.mark.skipif(importlib.util.find_spec("llama_cpp") is None, reason="llama-cpp-python 未安裝，跳過")
def test_llamacpp_backend_basic(gguf_path: str):
    # 延後 import，避免未安裝時污染其餘測試
    from agent_slm_server.chat_backends import load_backend

    backend = load_backend(model_path=gguf_path, backend="llama.cpp")

    messages = [
        {"role": "user", "content": "你好，請回答兩個字：測試"}
    ]

    out = backend.generate(messages=messages, max_new_tokens=16, temperature=0.0, top_p=0.95)

    # 結構檢查
    assert isinstance(out, dict), "後端輸出應為 dict"
    for key in ["text", "prompt_tokens", "completion_tokens", "total_tokens"]:
        assert key in out, f"缺少欄位: {key}"
    assert isinstance(out["text"], str)
    assert isinstance(out["prompt_tokens"], int)
    assert isinstance(out["completion_tokens"], int)
    assert isinstance(out["total_tokens"], int)
    assert out["total_tokens"] == out["prompt_tokens"] + out["completion_tokens"]

    # 簡單語義檢查：temperature=0 期望 deterministic，回覆包含『測試』或同義語
    assert any(token in out["text"] for token in ["測試", "test", "測"]), "輸出未包含期望字樣，實際: " + out["text"][:50]

@pytest.mark.skipif(not os.path.exists(MODEL_PATH_ENV), reason="GGUF 模型檔不存在，跳過 llama.cpp backend 測試")
@pytest.mark.skipif(importlib.util.find_spec("llama_cpp") is None, reason="llama-cpp-python 未安裝，跳過")
def test_llamacpp_backend_token_accounting(gguf_path: str):
    from agent_slm_server.chat_backends import load_backend
    backend = load_backend(model_path=gguf_path, backend="llama.cpp")

    base_messages = [
        {"role": "user", "content": "你好"},
    ]
    out_short = backend.generate(messages=base_messages, max_new_tokens=4, temperature=0.0)
    out_long = backend.generate(messages=base_messages, max_new_tokens=12, temperature=0.0)

    # completion_tokens 應該隨 max_new_tokens 上升或至少不減少（溫度0下通常遞增）
    assert out_long["completion_tokens"] >= out_short["completion_tokens"], "completion_tokens 未隨長度增加"

    # total_tokens 一致關係檢查
    for out in (out_short, out_long):
        assert out["total_tokens"] == out["prompt_tokens"] + out["completion_tokens"]

@pytest.mark.skipif(importlib.util.find_spec("llama_cpp") is not None, reason="llama-cpp 已安裝，本測試僅在未安裝時驗證 skip 邏輯")
def test_llamacpp_uninstalled_skip():
    # 若進到這裡代表 llama_cpp 未安裝，僅確認標誌條件成立
    assert importlib.util.find_spec("llama_cpp") is None
