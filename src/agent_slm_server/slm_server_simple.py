# 從 y26hkx_server.slm_server_simple 重命名至 agent_slm_server.slm_server_simple
import os
import logging
import threading
import time
import math
from pathlib import Path
from typing import Literal, Any
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "transformers").lower()
BACKEND_MODEL_PATH = os.getenv("MODEL_PATH", "./models/qwen/Qwen2.5-0.5B-Instruct")

try:
    if MODEL_BACKEND == "transformers":
        from .qwenchat import QwenChatAPI  # type: ignore
    else:
        QwenChatAPI = None  # type: ignore
except ImportError:  # pragma: no cover
    QwenChatAPI = None  # type: ignore

try:
    from .chat_backends import load_backend  # type: ignore
except Exception:  # pragma: no cover
    load_backend = None  # type: ignore
try:
    import tiktoken  # type: ignore
    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore
    _tiktoken_enc = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

SIMPLE_CONFIG = {
    "MODEL_PATH": os.getenv("MODEL_PATH", "./models/qwen/Qwen2.5-0.5B-Instruct"),
    "MODEL_QUANTIZATION": os.getenv("MODEL_QUANTIZATION", None),
    "TORCH_DTYPE": os.getenv("TORCH_DTYPE", "auto"),
    "COMPILE_MODEL": os.getenv("COMPILE_MODEL", "0") in {"1", "true", "TRUE", "True"},
    "LOW_CPU_MEM_USAGE": os.getenv("LOW_CPU_MEM_USAGE", "1") in {"1", "true", "TRUE", "True"},
    "MAX_INPUT_TOKENS": int(os.getenv("MAX_INPUT_TOKENS", "2048")),
    "MAX_INPUT_CHARS": int(os.getenv("MAX_INPUT_CHARS", "12000")),
    "MODEL_UNLOAD_ENABLED": os.getenv("MODEL_UNLOAD_ENABLED", "1") in {"1", "true", "TRUE", "True"},
    "MODEL_IDLE_TIMEOUT": int(os.getenv("MODEL_IDLE_TIMEOUT", "3600")),
    "MODEL_UNLOAD_CHECK_INTERVAL": int(os.getenv("MODEL_UNLOAD_CHECK_INTERVAL", "600")),
    "TORCH_NUM_THREADS": int(os.getenv("TORCH_NUM_THREADS", "0")),
    "GLOBAL_SEED": int(os.getenv("GLOBAL_SEED", "0")) if os.getenv("GLOBAL_SEED") else None,
    "DEFAULT_SYSTEM_PROMPT": os.getenv("DEFAULT_SYSTEM_PROMPT", None),
}

logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("slm_server_simple")
app = FastAPI(title="Qwen OpenAI-Compatible API (Simple)")

qwen = None  # backend instance or wrapper
_model_loaded_time = None
_model_last_used_time = None
qwen_init_lock = threading.Lock()
embedding_model_cache: dict[str, Any] = {}
_embedding_last_used_times: dict[str, float] = {}

import psutil  # type: ignore

def get_memory_info():
    try:
        memory = psutil.virtual_memory()
        return {
            "available_gb": memory.available / (1024**3),
            "usage_percent": memory.percent / 100.0,
            "total_gb": memory.total / (1024**3),
        }
    except Exception:  # noqa: BLE001
        return {"available_gb": 4.0, "usage_percent": 0.5, "total_gb": 8.0}

def discover_embedding_models():
    models: dict[str, str] = {}
    base_paths = [Path("models/embedding"), Path("models")]
    for base in base_paths:
        if not base.exists():
            continue
        for sub in base.iterdir():
            if sub.is_dir() and (sub / "config.json").exists():
                models[sub.name] = str(sub)
    if not models:
        models["paraphrase-MiniLM-L6-v2"] = "models/embedding/paraphrase-MiniLM-L6-v2"
    return models

EMBEDDING_MODELS = discover_embedding_models()
DEFAULT_EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None

class ChatCompletionsRequest(BaseModel):
    model: str = Field(default="qwen2.5-0.5b-instruct")
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    max_tokens: int | None = 512
    stream: bool | None = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"

class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionsResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage = ChatUsage()

class EmbeddingsRequest(BaseModel):
    model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = Field(default="float")

class EmbeddingDataItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]

class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingDataItem]
    model: str
    usage: ChatUsage = ChatUsage()

@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/health/memory")
def health_memory():
    memory_info = get_memory_info()
    return {"current_usage_percent": memory_info["usage_percent"],"available_gb": memory_info["available_gb"],"leak_detected": False}
@app.get("/health/detailed")
def health_detailed():
    memory_info = get_memory_info()
    HEALTH_USAGE_THRESHOLD = 0.9
    return {
        "status": "healthy" if memory_info["usage_percent"] < HEALTH_USAGE_THRESHOLD else "warning",
        "memory": {
            "current_usage_percent": memory_info["usage_percent"],
            "available_gb": memory_info["available_gb"],
            "total_gb": memory_info["total_gb"],
        },
        "models": {
            "main_model_loaded": qwen is not None,
            "embedding_models_count": len(embedding_model_cache),
            "embedding_models": list(embedding_model_cache.keys()),
        },
        "warnings": {
            "high_memory_usage": memory_info["usage_percent"] > HEALTH_USAGE_THRESHOLD,
            "memory_leak_detected": False,
        },
    }
@app.get("/v1/models")
def list_models():
    data = [{"id": "qwen2.5-0.5b-instruct", "object": "model", "owned_by": "local"}]
    for model_id in EMBEDDING_MODELS.keys():
        data.append({"id": model_id, "object": "model", "owned_by": "local"})
    return {"object": "list", "data": data}

@app.post("/v1/chat/completions", response_model=ChatCompletionsResponse)
def chat_completions(req: ChatCompletionsRequest):  # noqa: D401
    global qwen, _model_loaded_time, _model_last_used_time
    if qwen is None:
        with qwen_init_lock:
            if qwen is None:
                try:
                    logger.info(f"初始化後端: backend={MODEL_BACKEND} path={BACKEND_MODEL_PATH}")
                    if MODEL_BACKEND == "transformers":
                        if QwenChatAPI is None:  # pragma: no cover
                            raise RuntimeError("QwenChatAPI 未可用，請確認依賴是否安裝")
                        backend_obj = QwenChatAPI(
                            model_path=SIMPLE_CONFIG["MODEL_PATH"],
                            dtype=SIMPLE_CONFIG["TORCH_DTYPE"],
                            quantization=SIMPLE_CONFIG["MODEL_QUANTIZATION"],
                            compile_model=SIMPLE_CONFIG["COMPILE_MODEL"],
                            max_input_tokens=SIMPLE_CONFIG["MAX_INPUT_TOKENS"],
                            default_system_prompt=SIMPLE_CONFIG["DEFAULT_SYSTEM_PROMPT"],
                            global_seed=SIMPLE_CONFIG["GLOBAL_SEED"],
                            low_cpu_mem_usage=SIMPLE_CONFIG["LOW_CPU_MEM_USAGE"],
                        )
                        class _Wrapper:
                            def __init__(self, inner):
                                self.inner = inner
                            def generate(self, messages, max_new_tokens=256, temperature=0.7, top_p=0.9):
                                r = self.inner.chat_messages(
                                    messages=messages,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    return_usage=True,
                                )
                                return {
                                    "text": r.text,
                                    "prompt_tokens": r.prompt_tokens,
                                    "completion_tokens": r.completion_tokens,
                                    "total_tokens": r.total_tokens,
                                }
                        qwen_backend = _Wrapper(backend_obj)
                    else:
                        if load_backend is None:  # pragma: no cover
                            raise RuntimeError("chat_backends.load_backend 不可用")
                        qwen_backend = load_backend(
                            model_path=BACKEND_MODEL_PATH,
                            backend=MODEL_BACKEND,
                        )
                    qwen = qwen_backend
                    _model_loaded_time = time.time()
                    logger.info("後端初始化完成")
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    lower_msg = msg.lower()
                    logger.error(f"後端初始化失敗: {msg}")
                    if ("1455" in lower_msg or "page file" in lower_msg or "页面文件太小" in msg or "pagefile" in lower_msg) or ("memory" in lower_msg and "fail" in lower_msg):
                        suggestion = {
                            "error": "模型/後端載入失敗 (可能內存/分頁檔不足)",
                            "original_error": msg,
                            "backend": MODEL_BACKEND,
                            "suggestions": [
                                "增大 Windows Pagefile 至 >=16GB",
                                "嘗試設置 MODEL_BACKEND=llama.cpp 使用 GGUF + 量化模型 (Q4_K_M)",
                                "設定 MODEL_QUANTIZATION=4bit 並安裝 bitsandbytes (transformers 模式)",
                                "降低 MAX_INPUT_TOKENS 或關閉 COMPILE_MODEL",
                            ],
                        }
                        raise HTTPException(status_code=503, detail=suggestion) from e
                    raise HTTPException(status_code=500, detail=f"後端初始化失敗: {msg}") from e
    _model_last_used_time = time.time()
    messages = [{"role": m.role, "content": m.content or ""} for m in req.messages]
    try:
        output_text = qwen.generate(  # type: ignore[union-attr]
            messages=messages,
            max_new_tokens=req.max_tokens or 512,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
        )
        now = int(time.time())
        response_message = ChatMessage(role="assistant", content=output_text["text"])
        return ChatCompletionsResponse(
            id=f"chatcmpl-{now}",
            created=now,
            model=req.model,
            choices=[ChatChoice(index=0, message=response_message)],
            usage=ChatUsage(
                prompt_tokens=output_text.get("prompt_tokens", 0),
                completion_tokens=output_text.get("completion_tokens", 0),
                total_tokens=output_text.get("total_tokens", 0),
            ),
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"聊天完成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}") from e

@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
def create_embeddings(req: EmbeddingsRequest):
    if SentenceTransformer is None:
        raise HTTPException(status_code=500, detail="sentence-transformers 未安装")
    model_id = req.model
    if model_id not in EMBEDDING_MODELS:
        model_id = DEFAULT_EMBEDDING_MODEL
    if model_id not in embedding_model_cache:
        try:
            logger.info(f"正在加载嵌入模型: {model_id}")
            model_path = EMBEDDING_MODELS.get(model_id, model_id)
            model_obj = SentenceTransformer(model_path, device="cpu")  # type: ignore
            embedding_model_cache[model_id] = model_obj
            logger.info(f"嵌入模型加载完成: {model_id}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"嵌入模型加载失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}") from e
    model_obj = embedding_model_cache[model_id]
    _embedding_last_used_times[model_id] = time.time()
    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        raise HTTPException(status_code=400, detail="输入不能为空")
    try:
        vectors = model_obj.encode(inputs)
        data_items = []
        for i, vec in enumerate(vectors):
            data_items.append(EmbeddingDataItem(index=i, embedding=vec.tolist()))
        return EmbeddingsResponse(data=data_items, model=model_id, usage=ChatUsage(prompt_tokens=len(str(inputs)), total_tokens=len(str(inputs))))
    except Exception as e:  # noqa: BLE001
        logger.error(f"嵌入生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"嵌入生成失败: {str(e)}") from e

@app.get("/metrics")
def metrics():
    memory_info = get_memory_info()
    lines = [f"memory_usage_percent {memory_info['usage_percent']:.3f}",f"memory_available_gb {memory_info['available_gb']:.3f}",f"models_loaded {len(embedding_model_cache) + (1 if qwen else 0)}",f"embedding_models_loaded {len(embedding_model_cache)}"]
    if _model_loaded_time:
        lines.append(f"main_model_loaded_timestamp {_model_loaded_time}")
    return Response("\n".join(lines) + "\n", media_type="text/plain")

@app.on_event("startup")
async def startup_event():  # pragma: no cover
    logger.info("SLM 服务器启动中...")
    logger.info(f"模型路径: {SIMPLE_CONFIG['MODEL_PATH']}")
    logger.info(f"可用嵌入模型: {list(EMBEDDING_MODELS.keys())}")
    logger.info(f"内存管理: {'已启用' if SIMPLE_CONFIG['LOW_CPU_MEM_USAGE'] else '已禁用'}")
    memory_info = get_memory_info()
    logger.info(f"系统内存: {memory_info['total_gb']:.1f}GB 总计, {memory_info['available_gb']:.1f}GB 可用")
    logger.info("SLM 服务器启动完成")

if __name__ == "__main__":  # pragma: no cover
    import uvicorn  # type: ignore
    uvicorn.run(app, host="127.0.0.1", port=8001)
