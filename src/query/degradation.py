"""
降级管理器 - 当检索失败时的降级策略
"""
from typing import Optional, Dict, Any
from src.utils.logger import logger


class DegradationManager:
    """降级管理器 - 处理检索失败的情况"""

    def __init__(self):
        self.fallback_strategies = ["keyword", "direct"]

    def handle_degradation(self, query: str) -> Any:
        """处理降级"""
        # 简化的降级处理
        logger.warning(f"检索失败，使用降级策略处理查询: {query}")
        return None

    def should_degrade(self, context: dict) -> bool:
        """判断是否需要降级"""
        return not context.get("results")
