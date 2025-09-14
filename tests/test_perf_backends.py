"""簡易性能基準測試

目的:
  - 對比可用後端 (transformers / llama.cpp) 單輪生成 latency 與 tokens/sec
  - 若某後端不可用則 skip
  - 提供輸出摘要，方便手動觀察

使用方式:
  1. 啟動服務 (建議關閉 compile、保持一致 MAX_INPUT_TOKENS)
  2. pytest -q tests/test_perf_backends.py -s

注意:
  - 這不是嚴格科學基準，只做快速相對比較
  - 若要穩定結果可設定 GLOBAL_SEED 與固定溫度 0
"""
from __future__ import annotations

import time
import statistics
import requests
import importlib.util
import os
import pytest

BASE_URL = os.getenv("PERF_BASE_URL", "http://127.0.0.1:8001")
CHAT_ENDPOINT = f"{BASE_URL}/v1/chat/completions"
MODEL_TRANSFORMERS = "qwen2.5-0.5b-instruct"  # 現有 id
MODEL_LLAMA = "qwen2.5-0.5b-instruct"  # llama.cpp 實際上同 id 對外一致

PROMPT = "請用繁體中文回答：簡述人工智慧的兩個常見應用領域。"
N_WARMUP = 1
N_RUNS = 3
MAX_TOKENS = 64
TIMEOUT = 120


def _call_chat(model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 0.95,
    }
    r = requests.post(CHAT_ENDPOINT, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _backend_name() -> str:
    # 從 /v1/models 判斷 owned_by
    try:
        r = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        data = r.json().get("data", [])
        for m in data:
            if m.get("id") == MODEL_TRANSFORMERS:
                return m.get("owned_by", "unknown")
    except Exception:
        pass
    return "unknown"


def _service_up() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=2)
        return r.ok
    except Exception:
        return False


@pytest.mark.skipif(not _service_up(), reason="服務未啟動，跳過性能基準")
class TestPerf:
    def test_transformers_perf(self):
        if _backend_name() != "transformers":
            pytest.skip("當前主後端非 transformers")
        latencies = []
        tokens = []
        # warmup
        for _ in range(N_WARMUP):
            try:
                _call_chat(MODEL_TRANSFORMERS)
            except Exception:
                pytest.skip("transformers 後端呼叫失敗，跳過")
        for _ in range(N_RUNS):
            t0 = time.time()
            resp = _call_chat(MODEL_TRANSFORMERS)
            dt = time.time() - t0
            usage = resp.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            latencies.append(dt)
            tokens.append(total_tokens)
        avg_lat = statistics.mean(latencies)
        p95_lat = statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 1 else avg_lat
        avg_tokens = statistics.mean(tokens) if tokens else 0
        tps = avg_tokens / avg_lat if avg_lat > 0 else 0
        print(f"\n[transformers] runs={N_RUNS} avg_latency={avg_lat:.3f}s p95={p95_lat:.3f}s avg_tokens={avg_tokens:.1f} tokens_per_sec={tps:.1f}")
        # 基礎斷言：latency 與 tokens 數合理 (>0)
        assert avg_lat > 0 and avg_tokens >= 0

    def test_llamacpp_perf(self):
        # 僅在 llama.cpp 已啟動的情況下測試（owned_by=llama.cpp）
        if _backend_name() != "llama.cpp":
            pytest.skip("當前主後端非 llama.cpp")
        # 確認模組安裝（非必要，但若未安裝代表不應該是該後端）
        if importlib.util.find_spec("llama_cpp") is None:
            pytest.skip("llama-cpp-python 未安裝")
        latencies = []
        tokens = []
        for _ in range(N_WARMUP):
            try:
                _call_chat(MODEL_LLAMA)
            except Exception:
                pytest.skip("llama.cpp 後端呼叫失敗，跳過")
        for _ in range(N_RUNS):
            t0 = time.time()
            resp = _call_chat(MODEL_LLAMA)
            dt = time.time() - t0
            usage = resp.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            latencies.append(dt)
            tokens.append(total_tokens)
        avg_lat = statistics.mean(latencies)
        p95_lat = statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 1 else avg_lat
        avg_tokens = statistics.mean(tokens) if tokens else 0
        tps = avg_tokens / avg_lat if avg_lat > 0 else 0
        print(f"\n[llama.cpp] runs={N_RUNS} avg_latency={avg_lat:.3f}s p95={p95_lat:.3f}s avg_tokens={avg_tokens:.1f} tokens_per_sec={tps:.1f}")
        assert avg_lat > 0 and avg_tokens >= 0
