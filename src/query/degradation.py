"""
降级管理器 - 当检索失败时的降级策略

设计文档说明:
- 超时或检索失败时降级为纯向量检索
- 返回有效的 PipelineResult 而非 None
"""
from typing import Optional, Any
from src.utils.logger import logger


class DegradationManager:
    """降级管理器 - 处理检索失败的情况"""

    def __init__(self):
        self.fallback_strategies = ["vector", "direct"]

    def handle_degradation(self, query: str) -> Any:
        """处理降级，返回 PipelineResult 格式的降级结果"""
        # 避免循环导入，在函数内导入
        from src.query.pipeline import PipelineResult
        logger.warning(f"检索结果为空，触发降级: {query[:50]}...")
        return PipelineResult(
            answer="抱歉，未能从知识库中检索到与您问题相关的内容。请尝试换个方式提问，或上传更多相关文档。",
            degraded=True,
            degradation_reasons=["向量检索结果为空"],
            retrieval_paths_used=["degradation_fallback"],
        )

    def should_degrade(self, context: dict) -> bool:
        """判断是否需要降级"""
        return not context.get("results")
