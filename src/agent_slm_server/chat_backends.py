"""Minimal backend abstraction for chat models.

Allows switching between transformers (existing QwenChatAPI) and llama.cpp (GGUF) backends
using environment variable MODEL_BACKEND=transformers|llama.cpp

This is intentionally lightweight and avoids any heavy optional import if not needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import os

# Protocol definition -------------------------------------------------------
class ChatBackend(Protocol):
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> dict[str, Any]:
        """Generate completion from list of messages.
        Return dict with keys: text, prompt_tokens, completion_tokens, total_tokens.
        """
        ...

# Transformers (existing) ---------------------------------------------------
@dataclass
class TransformersBackend:
    qwen_api: Any

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> dict[str, Any]:
        result = self.qwen_api.chat_messages(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_usage=True,
        )
        return {
            "text": result.text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }

# llama.cpp backend ---------------------------------------------------------
@dataclass
class LlamaCppBackend:
    llm: Any

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> dict[str, Any]:
        # Convert to llama.cpp chat format (already similar: list of {role, content})
        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            choice = output["choices"][0]["message"]
            text = choice.get("content", "")
            usage = output.get("usage", {})
            return {
                "text": text,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
            }
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"llama.cpp generation failed: {e}") from e

# Factory -------------------------------------------------------------------

def load_backend(
    model_path: str,
    backend: str = "transformers",
    **kwargs: Any,
) -> ChatBackend:
    backend = backend.lower()
    if backend == "transformers":
        from .qwenchat import QwenChatAPI  # local import to avoid heavy cost
        qwen = QwenChatAPI(model_path=model_path, **kwargs)
        return TransformersBackend(qwen_api=qwen)
    elif backend in {"llama.cpp", "llamacpp", "llama"}:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("llama_cpp not installed. pip install llama-cpp-python") from e
        # Minimal sane defaults; allow override via kwargs
        n_ctx = int(os.getenv("LLAMA_CTX", os.getenv("MAX_INPUT_TOKENS", "1024")))
        n_threads = int(os.getenv("LLAMA_THREADS", "4"))
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
        )
        return LlamaCppBackend(llm=llm)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported MODEL_BACKEND: {backend}")
