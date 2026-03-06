"""
查询路由器 - 决定查询使用哪些检索策略
"""
from typing import Dict, Any
from src.utils.logger import logger


class QueryRouter:
    """查询路由器 - 决定查询的处理策略"""

    def __init__(self):
        self.strategies = ["graph", "vector", "hybrid"]

    def route(self, query: str) -> Dict[str, Any]:
        """决定查询的路由策略"""
        # 简单策略：默认使用混合模式
        return {
            "strategy": "hybrid",
            "use_graph": True,
            "use_vector": True,
            "confidence": 0.8
        }
