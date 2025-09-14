"""Minimal local smoke test without installing the package.

Adds `src` to sys.path then exercises health, models, embeddings, chat endpoints.
"""
from fastapi.testclient import TestClient
import sys
from pathlib import Path

SRC_PATH = Path(__file__).parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_slm_server import slm_server_simple as simple  # type: ignore  # noqa: E402

HTTP_OK = 200

c = TestClient(simple.app)
print("== basic health check via TestClient ==")
health_resp = c.get("/health")
print("GET /health", health_resp.status_code, health_resp.json())
models_resp = c.get("/v1/models")
print("GET /v1/models", models_resp.status_code, len(models_resp.json().get("data", [])))
emb_status = c.post(
    "/v1/embeddings",
    json={"model": "paraphrase-MiniLM-L6-v2", "input": "hello"},
).status_code
print("POST /v1/embeddings", emb_status)
try:
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-0.5b-instruct",
            "messages": [{"role": "user", "content": "只回你好兩字"}],
            "max_tokens": 16,
        },
        timeout=60,
    )
    print("POST /v1/chat/completions", r.status_code)
    if r.status_code == HTTP_OK:
        print("assistant snippet:", r.json()["choices"][0]["message"]["content"][:60])
except Exception as e:  # noqa: BLE001
    print("chat request failed:", e)
