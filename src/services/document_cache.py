"""
文档缓存 - 缓存文档解析结果
"""
from typing import Optional, Dict, Any
import time
from src.utils.logger import logger


class DocumentCache:
    """文档缓存 - 缓存文档解析结果"""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                del self.cache[key]
        return None

    def set(self, key: str, data: Dict[str, Any]) -> None:
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }

    def put(self, key: str, data: Dict[str, Any]) -> None:
        """设置缓存（put 方法的别名）"""
        self.set(key, data)

    def acquire(self, key: str) -> Optional[Dict[str, Any]]:
        """获取并删除缓存（原子操作）"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                data = entry["data"]
                del self.cache[key]  # 获取后删除
                return data
            else:
                del self.cache[key]
        return None

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
    
    def release(self, key: str) -> None:
        """释放缓存（删除缓存项）"""
        if key in self.cache:
            del self.cache[key]
