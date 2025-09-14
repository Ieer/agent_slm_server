# 從 y26hkx_server.memory_monitor 重命名至 agent_slm_server.memory_monitor
"""
内存监控和管理工具
提供实时内存监控、内存泄漏检测和自动内存优化功能
"""
import psutil
import gc
import time
import logging
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """内存快照数据结构"""
    timestamp: float
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    swap_used_gb: float
    process_memory_gb: float

class MemoryMonitor:
    """内存监控器"""
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.snapshots: list[MemorySnapshot] = []
        self.max_snapshots = 100
    def get_current_memory_info(self) -> MemorySnapshot:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        process = psutil.Process()
        return MemorySnapshot(
            timestamp=time.time(),
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            used_memory_gb=memory.used / (1024**3),
            memory_percent=memory.percent / 100.0,
            swap_used_gb=swap.used / (1024**3),
            process_memory_gb=process.memory_info().rss / (1024**3)
        )
    def take_snapshot(self) -> MemorySnapshot:
        snapshot = self.get_current_memory_info()
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        self._check_memory_thresholds(snapshot)
        return snapshot
    def _check_memory_thresholds(self, snapshot: MemorySnapshot):
        if snapshot.memory_percent >= self.critical_threshold:
            logger.critical(f"内存使用率达到严重水平: {snapshot.memory_percent:.1%} (可用: {snapshot.available_memory_gb:.1f}GB)")
        elif snapshot.memory_percent >= self.warning_threshold:
            logger.warning(f"内存使用率较高: {snapshot.memory_percent:.1%} (可用: {snapshot.available_memory_gb:.1f}GB)")
    def get_memory_trend(self, minutes: int = 5) -> dict[str, float]:
        if len(self.snapshots) < 2:
            return {"trend": 0.0, "avg_usage": 0.0, "peak_usage": 0.0}
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        if len(recent_snapshots) < 2:
            recent_snapshots = self.snapshots[-2:]
        first_usage = recent_snapshots[0].memory_percent
        last_usage = recent_snapshots[-1].memory_percent
        trend = last_usage - first_usage
        usages = [s.memory_percent for s in recent_snapshots]
        avg_usage = sum(usages) / len(usages)
        peak_usage = max(usages)
        return {
            "trend": trend,
            "avg_usage": avg_usage,
            "peak_usage": peak_usage,
            "sample_count": len(recent_snapshots)
        }
    def detect_memory_leak(self, threshold_increase: float = 0.1, window_minutes: int = 10) -> bool:
        trend_info = self.get_memory_trend(window_minutes)
        if trend_info["trend"] > threshold_increase:
            logger.warning(f"检测到可能的内存泄漏: {window_minutes}分钟内内存增长 {trend_info['trend']:.1%}")
            return True
        return False
    def force_garbage_collection(self) -> dict[str, int]:
        before_snapshot = self.get_current_memory_info()
        collected = gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        after_snapshot = self.get_current_memory_info()
        freed_memory_gb = before_snapshot.used_memory_gb - after_snapshot.used_memory_gb
        logger.info(f"垃圾回收完成: 回收对象 {collected} 个, 释放内存 {freed_memory_gb:.2f}GB")
        return {
            "objects_collected": collected,
            "memory_freed_gb": freed_memory_gb,
            "before_usage_percent": before_snapshot.memory_percent,
            "after_usage_percent": after_snapshot.memory_percent
        }

@contextmanager
def memory_tracking(monitor: MemoryMonitor, operation_name: str = "operation"):
    before = monitor.take_snapshot()
    start_time = time.time()
    try:
        yield monitor
    finally:
        after = monitor.take_snapshot()
        duration = time.time() - start_time
        memory_delta = after.process_memory_gb - before.process_memory_gb
        if memory_delta > 0.1:
            logger.info(f"{operation_name} 内存使用情况: 增长 {memory_delta:.2f}GB, 耗时 {duration:.1f}s")

global_memory_monitor = MemoryMonitor()

def get_memory_status() -> dict[str, object]:
    snapshot = global_memory_monitor.take_snapshot()
    trend = global_memory_monitor.get_memory_trend()
    return {
        "current_usage_percent": snapshot.memory_percent,
        "available_gb": snapshot.available_memory_gb,
        "process_memory_gb": snapshot.process_memory_gb,
        "memory_trend": trend["trend"],
        "leak_detected": global_memory_monitor.detect_memory_leak()
    }
