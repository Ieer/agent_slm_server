# 從 y26hkx_server.performance_config 重命名至 agent_slm_server.performance_config
import os

PERFORMANCE_CONFIG: dict[str, object] = {
    "MODEL_MEMORY_LIMIT_GB": float(os.getenv("MODEL_MEMORY_LIMIT_GB", "4.0")),
    "EMBEDDING_MEMORY_LIMIT_GB": float(os.getenv("EMBEDDING_MEMORY_LIMIT_GB", "3.0")),
    "AUTO_MEMORY_MANAGEMENT": os.getenv("AUTO_MEMORY_MANAGEMENT", "1") in {"1", "true", "TRUE", "True"},
    "MAX_BATCH_SIZE": int(os.getenv("MAX_BATCH_SIZE", "32")),
    "MIN_BATCH_SIZE": int(os.getenv("MIN_BATCH_SIZE", "1")),
    "ADAPTIVE_BATCH_SIZING": os.getenv("ADAPTIVE_BATCH_SIZING", "1") in {"1", "true", "TRUE", "True"},
    "MAX_CONCURRENT_REQUESTS": int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")),
    "REQUEST_TIMEOUT_SECONDS": float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120.0")),
    "ENABLE_MODEL_CACHE": os.getenv("ENABLE_MODEL_CACHE", "1") in {"1", "true", "TRUE", "True"},
    "CACHE_TTL_SECONDS": int(os.getenv("CACHE_TTL_SECONDS", "3600")),
    "PRELOAD_MAIN_MODEL": os.getenv("PRELOAD_MAIN_MODEL", "0") in {"1", "true", "TRUE", "True"},
    "PRELOAD_EMBEDDING_MODEL": os.getenv("PRELOAD_EMBEDDING_MODEL", "0") in {"1", "true", "TRUE", "True"},
    "ENABLE_DETAILED_METRICS": os.getenv("ENABLE_DETAILED_METRICS", "1") in {"1", "true", "TRUE", "True"},
    "METRICS_COLLECTION_INTERVAL": int(os.getenv("METRICS_COLLECTION_INTERVAL", "60")),
}

def get_optimal_batch_size(available_memory_gb: float, model_size_gb: float, text_length_avg: int = 100) -> int:
    if available_memory_gb <= model_size_gb:
        return PERFORMANCE_CONFIG["MIN_BATCH_SIZE"]  # type: ignore[index]
    processing_memory = available_memory_gb - model_size_gb
    memory_per_text_mb = max(1.0, text_length_avg / 100.0 * 10.0)
    batch_size = int((processing_memory * 1024) / memory_per_text_mb)
    return max(
        PERFORMANCE_CONFIG["MIN_BATCH_SIZE"],  # type: ignore[index]
        min(batch_size, PERFORMANCE_CONFIG["MAX_BATCH_SIZE"])  # type: ignore[index]
    )

def should_use_quantization(available_memory_gb: float) -> str | None:
    if available_memory_gb < 4.0:
        return "4bit"
    elif available_memory_gb < 8.0:
        return "8bit"
    else:
        return None
