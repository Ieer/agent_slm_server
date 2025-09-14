"""Full server implementation (renamed from y26hkx_server.slm_server).

啟動命令:
  uvicorn agent_slm_server.slm_server:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

# isort: skip_file

import asyncio
import gc
import logging
import math
import os
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .qwenchat import QwenChatAPI
from .chat_backends import load_backend
from .memory_monitor import get_memory_status
from .performance_config import PERFORMANCE_CONFIG

ENV = {
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
	"AUTO_MEMORY_MANAGEMENT": PERFORMANCE_CONFIG["AUTO_MEMORY_MANAGEMENT"],
	"MAX_CONCURRENT_REQUESTS": PERFORMANCE_CONFIG["MAX_CONCURRENT_REQUESTS"],
	"GLOBAL_SEED": int(os.getenv("GLOBAL_SEED", "0")) if os.getenv("GLOBAL_SEED") else None,
	"DEFAULT_SYSTEM_PROMPT": os.getenv("DEFAULT_SYSTEM_PROMPT", None),
	"STRICT_MEMORY_CHECK": os.getenv("STRICT_MEMORY_CHECK", "0") in {"1", "true", "TRUE", "True"},
	"MEMORY_EXPANSION_FP": float(os.getenv("MEMORY_EXPANSION_FP", "2.2")),
	"MEMORY_EXPANSION_8BIT": float(os.getenv("MEMORY_EXPANSION_8BIT", "1.3")),
	"MEMORY_EXPANSION_4BIT": float(os.getenv("MEMORY_EXPANSION_4BIT", "1.15")),
	"MIN_REMAIN_AFTER_LOAD_GB": float(os.getenv("MIN_REMAIN_AFTER_LOAD_GB", "1.0")),
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').propagate = False

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
	level=getattr(logging, LOG_LEVEL, logging.INFO),
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
app_logger = logging.getLogger("agent_slm_server")

try:
	from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
	SentenceTransformer = None  # type: ignore

try:
	import torch  # type: ignore
	if ENV["TORCH_NUM_THREADS"] > 0:
		torch.set_num_threads(ENV["TORCH_NUM_THREADS"])
		app_logger.info(f"已設定 PyTorch 執行緒數量為 {ENV['TORCH_NUM_THREADS']}")
except ImportError:
	torch = None  # type: ignore

app = FastAPI(title="Qwen OpenAI-Compatible API")

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "transformers").lower()
class _BackendProto(Protocol):
	def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> dict: ...

class _MainModelState:
	def __init__(self):
		self.backend: _BackendProto | None = None
		self.loaded_time: float | None = None
		self.last_used_time: float | None = None

MODEL_STATE = _MainModelState()
qwen_init_lock = threading.Lock()

embedding_model_cache: dict[str, Any] = {}
_embedding_last_used_times: dict[str, float] = {}

_metrics_lock = threading.Lock()
_metrics: dict[str, float] = {
	"requests_total": 0,
	"chat_requests_total": 0,
	"embeddings_requests_total": 0,
	"errors_total": 0,
	"errors_memory": 0,
	"errors_timeout": 0,
	"errors_validation": 0,
	"chat_latency_sum_seconds": 0.0,
	"chat_latency_max_seconds": 0.0,
}

def _metrics_inc(key: str, val: float = 1.0):
	with _metrics_lock:
		_metrics[key] = _metrics.get(key, 0) + val

def _metrics_observe_chat_latency(lat: float):
	with _metrics_lock:
		_metrics["chat_latency_sum_seconds"] += lat
		_metrics["chat_latency_max_seconds"] = max(_metrics["chat_latency_max_seconds"], lat)

@app.get("/metrics")
def metrics():
	lines = []
	with _metrics_lock:
		for k, v in _metrics.items():
			lines.append(f"agent_slm_{k} {v}")
	if MODEL_STATE.loaded_time is not None:
		lines.append(f"agent_slm_model_loaded_timestamp {MODEL_STATE.loaded_time}")
	return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

def _discover_embedding_models() -> dict[str, str]:
	base_paths = [Path("models/embedding"), Path("models")]
	mapping: dict[str, str] = {}
	for base in base_paths:
		if not base.exists():
			continue
		for sub in base.iterdir():
			if not sub.is_dir():
				continue
			if (sub / "config_sentence_transformers.json").exists() or (sub / "tokenizer.json").exists():
				mapping[sub.name] = str(sub)
	return mapping

class _EmbeddingState:
	def __init__(self):
		self.map: dict[str, str] = _discover_embedding_models()
		self.last_refresh: float = time.time()

EMBEDDINGS = _EmbeddingState()
DEFAULT_EMBEDDING_MODEL_ID = "paraphrase-MiniLM-L6-v2"
_EMBEDDING_MAP_LOCK = threading.Lock()
_EMBEDDING_MAP_TTL_SECONDS = 300

def get_embedding_model_map(force: bool = False) -> dict[str, str]:
	now = time.time()
	if force or (now - EMBEDDINGS.last_refresh > _EMBEDDING_MAP_TTL_SECONDS):
		with _EMBEDDING_MAP_LOCK:
			if force or (time.time() - EMBEDDINGS.last_refresh > _EMBEDDING_MAP_TTL_SECONDS):
				EMBEDDINGS.map = _discover_embedding_models()
				EMBEDDINGS.last_refresh = time.time()
	return EMBEDDINGS.map

def unload_models_if_idle():
	current_time = time.time()
	idle_timeout = ENV["MODEL_IDLE_TIMEOUT"]
	if not ENV["MODEL_UNLOAD_ENABLED"]:
		return
	if MODEL_STATE.backend is not None and MODEL_STATE.last_used_time is not None:
		idle_seconds = current_time - MODEL_STATE.last_used_time
		if idle_seconds > idle_timeout:
			with qwen_init_lock:
				if MODEL_STATE.backend is not None:
					app_logger.info(f"主模型已閒置 {idle_seconds:.1f} 秒，正在卸載...")
					MODEL_STATE.backend = None
					MODEL_STATE.loaded_time = None
					MODEL_STATE.last_used_time = None
					gc.collect()
					try:
						if torch and torch.cuda.is_available():  # type: ignore[attr-defined]
							torch.cuda.empty_cache()
					except Exception:
						pass
					app_logger.info("主模型已成功卸載")
	models_to_unload: list[str] = []
	for model_id, last_used in list(_embedding_last_used_times.items()):
		idle_seconds = current_time - last_used
		if idle_seconds > idle_timeout and model_id in embedding_model_cache:
			models_to_unload.append(model_id)
	for model_id in models_to_unload:
		try:
			app_logger.info(f"嵌入模型 '{model_id}' 閒置，卸載中")
			embedding_model_cache.pop(model_id, None)
			_embedding_last_used_times.pop(model_id, None)
			gc.collect()
		except Exception as e:
			app_logger.error(f"卸載嵌入模型 '{model_id}' 時發生錯誤: {e}")

@app.on_event("startup")
async def startup_tasks():
	mapping = get_embedding_model_map(force=True)
	app_logger.info(f"可用 embedding 模型: {list(mapping.keys())}")
	app_logger.info(f"預設 embedding 模型: {DEFAULT_EMBEDDING_MODEL_ID}")
	app_logger.info(f"模型路徑: {ENV['MODEL_PATH']}")
	app_logger.info(f"量化模式: {ENV['MODEL_QUANTIZATION'] or '無'}")
	if ENV["MODEL_UNLOAD_ENABLED"]:
		app_logger.info(f"模型卸載啟用，閒置超時: {ENV['MODEL_IDLE_TIMEOUT']} 秒")
		async def model_unload_monitor():
			try:
				while True:
					await asyncio.sleep(ENV["MODEL_UNLOAD_CHECK_INTERVAL"])
					unload_models_if_idle()
			except asyncio.CancelledError:
				return
			except Exception as e:  # pragma: no cover
				app_logger.error(f"模型卸載監控錯誤: {e}")
		asyncio.get_event_loop().create_task(model_unload_monitor())

class ReloadEmbeddingsResponse(BaseModel):
	refreshed: bool
	models: list[str]
	default_model: str
	ttl_seconds: int

@app.post("/v1/reload-embeddings", response_model=ReloadEmbeddingsResponse)
def reload_embeddings():
	mapping = get_embedding_model_map(force=True)
	return ReloadEmbeddingsResponse(
		refreshed=True,
		models=list(mapping.keys()),
		default_model=DEFAULT_EMBEDDING_MODEL_ID,
		ttl_seconds=_EMBEDDING_MAP_TTL_SECONDS,
	)

try:
	import tiktoken  # type: ignore
	_tiktoken_enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
	tiktoken = None  # type: ignore
	_tiktoken_enc = None

def _count_tokens(text: str) -> int:
	if _tiktoken_enc is not None:
		try:
			return len(_tiktoken_enc.encode(text))
		except Exception:
			pass
	return max(1, math.ceil(len(text) / 4))

def _truncate_messages_by_tokens(messages: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
	if limit is None or limit <= 0:
		return messages[-1:] if messages else []
	total = 0
	kept: list[dict[str, Any]] = []
	for m in reversed(messages):
		content = m.get("content")
		if not isinstance(content, str):
			kept.append(m)
			continue
		length = _count_tokens(content)
		if total + length <= limit:
			kept.append(m)
			total += length
		else:
			take = max(0, limit - total)
			if take > 0:
				approx_ratio = take / length
				cut_chars = int(len(content) * approx_ratio)
				m2 = dict(m)
				m2["content"] = content[-cut_chars:]
				kept.append(m2)
				total += take
			break
	return list(reversed(kept))

@app.get("/health")
def health():
	return {"status": "ok"}

@app.get("/health/memory")
def health_memory():
	return get_memory_status()

@app.get("/health/detailed")
def health_detailed():
	memory_status = get_memory_status()
	model_status = {
		"main_model_loaded": MODEL_STATE.backend is not None,
		"embedding_models_count": len(embedding_model_cache),
		"embedding_models": list(embedding_model_cache.keys()),
	}
	MEMORY_SAFE_THRESHOLD = 0.9
	val = memory_status.get("current_usage_percent", 1.0)
	if isinstance(val, (int, float, str)):
		try:
			current_usage = float(val)
		except (ValueError, TypeError):
			current_usage = 1.0
	else:
		current_usage = 1.0
	memory_safe = current_usage < MEMORY_SAFE_THRESHOLD
	leak_detected = memory_status["leak_detected"]
	overall_status = "healthy" if memory_safe and not leak_detected else "warning"
	return {
		"status": overall_status,
		"memory": memory_status,
		"models": model_status,
		"warnings": {
			"high_memory_usage": not memory_safe,
			"memory_leak_detected": leak_detected,
		},
	}

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
	model: str = Field(default=DEFAULT_EMBEDDING_MODEL_ID)
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

@app.get("/v1/models")
def list_models():
	data = [
		{"id": "qwen2.5-0.5b-instruct", "object": "model", "owned_by": MODEL_BACKEND},
	]
	for mid in get_embedding_model_map().keys():
		data.append({"id": mid, "object": "model", "owned_by": "local"})
	return {"object": "list", "data": data}

def _init_backend_if_needed():
	if MODEL_STATE.backend is not None:
		return
	with qwen_init_lock:
		if MODEL_STATE.backend is not None:
			return
		try:
			app_logger.info(f"[backend-init] backend={MODEL_BACKEND} path={ENV['MODEL_PATH']}")
			if MODEL_BACKEND == "transformers":
				backend_obj = QwenChatAPI(
					model_path=ENV["MODEL_PATH"],
					dtype=ENV["TORCH_DTYPE"],
					quantization=ENV["MODEL_QUANTIZATION"],
					compile_model=ENV["COMPILE_MODEL"],
					max_input_tokens=ENV["MAX_INPUT_TOKENS"],
					default_system_prompt=ENV["DEFAULT_SYSTEM_PROMPT"],
					global_seed=ENV["GLOBAL_SEED"],
					low_cpu_mem_usage=ENV["LOW_CPU_MEM_USAGE"],
				)
				class _Wrap:
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
				backend_instance = _Wrap(backend_obj)
			else:
				backend_instance = load_backend(
					model_path=ENV["MODEL_PATH"],
					backend=MODEL_BACKEND,
				)
			MODEL_STATE.backend = backend_instance
			MODEL_STATE.loaded_time = time.time()
			app_logger.info("[backend-init] success")
		except Exception as e:  # noqa: BLE001
			msg = str(e)
			lower = msg.lower()
			app_logger.error(f"[backend-init] failed: {msg}")
			if ("1455" in lower or "page file" in lower or "pagefile" in lower or "页面文件" in msg) or ("memory" in lower and "fail" in lower):
				suggestion = {
					"error": "模型/後端載入失敗 (內存或分頁檔不足)",
					"backend": MODEL_BACKEND,
					"original_error": msg,
					"suggestions": [
						"增大 Pagefile 至 >=16GB",
						"嘗試 MODEL_BACKEND=llama.cpp + GGUF Q4_K_M",
						"設定 MODEL_QUANTIZATION=4bit 並安裝 bitsandbytes",
						"降低 MAX_INPUT_TOKENS 或關閉 COMPILE_MODEL",
					],
				}
				raise HTTPException(status_code=503, detail=suggestion) from e
			raise HTTPException(status_code=500, detail=f"後端載入失敗: {msg}") from e

@app.post("/v1/chat/completions", response_model=ChatCompletionsResponse)
def chat_completions(req: ChatCompletionsRequest):
	start_t = time.time()
	_metrics_inc("requests_total")
	_metrics_inc("chat_requests_total")
	_init_backend_if_needed()
	messages = [{"role": m.role, "content": m.content or ""} for m in req.messages]
	if ENV["MAX_INPUT_TOKENS"] and ENV["MAX_INPUT_TOKENS"] > 0:
		messages = _truncate_messages_by_tokens(messages, ENV["MAX_INPUT_TOKENS"])
	try:
		if MODEL_STATE.backend is None or not hasattr(MODEL_STATE.backend, 'generate'):
			raise HTTPException(status_code=500, detail="backend 未初始化或不支援 generate")
		out = MODEL_STATE.backend.generate(
			messages=messages,
			max_new_tokens=req.max_tokens or 512,
			temperature=req.temperature or 0.7,
			top_p=req.top_p or 0.9,
		)
		latency = time.time() - start_t
		_metrics_observe_chat_latency(latency)
		now = int(time.time())
		return ChatCompletionsResponse(
			id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
			created=now,
			model=req.model,
			choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=out["text"]))],
			usage=ChatUsage(
				prompt_tokens=out.get("prompt_tokens", 0),
				completion_tokens=out.get("completion_tokens", 0),
				total_tokens=out.get("total_tokens", 0),
			),
		)
	except HTTPException:
		raise
	except Exception as e:  # noqa: BLE001
		_metrics_inc("errors_total")
		app_logger.error(f"chat error: {e}")
		raise HTTPException(status_code=500, detail=f"chat failed: {e}") from e
	finally:
		MODEL_STATE.last_used_time = time.time()

@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
def create_embeddings(req: EmbeddingsRequest):
	_metrics_inc("requests_total")
	_metrics_inc("embeddings_requests_total")
	if SentenceTransformer is None:
		raise HTTPException(status_code=500, detail="sentence-transformers 未安裝")
	model_id = req.model
	mapping = get_embedding_model_map()
	if model_id not in mapping:
		model_id = DEFAULT_EMBEDDING_MODEL_ID
	if model_id not in embedding_model_cache:
		try:
			app_logger.info(f"[embedding-load] {model_id}")
			embedding_model_cache[model_id] = SentenceTransformer(mapping[model_id], device="cpu")  # type: ignore
		except Exception as e:  # noqa: BLE001
			_metrics_inc("errors_total")
			raise HTTPException(status_code=500, detail=f"嵌入模型載入失敗: {e}") from e
	model_obj = embedding_model_cache[model_id]
	_embedding_last_used_times[model_id] = time.time()
	inputs = [req.input] if isinstance(req.input, str) else req.input
	if not inputs:
		raise HTTPException(status_code=400, detail="input 不可為空")
	try:
		vectors = model_obj.encode(inputs)
		data_items = [EmbeddingDataItem(index=i, embedding=vec.tolist()) for i, vec in enumerate(vectors)]
		usage = ChatUsage(prompt_tokens=len(str(inputs)), total_tokens=len(str(inputs)))
		return EmbeddingsResponse(data=data_items, model=model_id, usage=usage)
	except Exception as e:  # noqa: BLE001
		_metrics_inc("errors_total")
		raise HTTPException(status_code=500, detail=f"嵌入生成失敗: {e}") from e
