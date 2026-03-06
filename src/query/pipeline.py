"""
查询管道 - 处理查询的完整流程
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from src.utils.logger import logger


@dataclass
class PipelineResult:
    """管道执行结果"""
    answer: str
    sources: List[dict] = field(default_factory=list)
    graph_context: Optional[dict] = None
    vector_context: Optional[List[dict]] = None
    metadata: Optional[dict] = field(default_factory=dict)


class QueryPipeline:
    """查询管道 - 协调各个组件完成查询"""

    def __init__(
        self,
        router,
        coe_engine,
        graph_traversal,
        synthesizer,
        cache,
        degradation_manager,
        budget_profile: str = "medium"
    ):
        self.router = router
        self.coe_engine = coe_engine
        self.graph_traversal = graph_traversal
        self.synthesizer = synthesizer
        self.cache = cache
        self.degradation_manager = degradation_manager
        self.budget_profile = budget_profile

    def execute(self, query: str, budget: Optional[dict] = None) -> PipelineResult:
        """执行查询管道"""
        budget = budget or {}
        logger.info(f"执行查询管道: {query[:50]}...")

        # 1. 路由决策
        route = self.router.route(query)

        # 2. 图检索
        graph_context = self._fetch_graph_context(query, route, budget)

        # 3. 向量检索
        vector_context = self._fetch_vector_context(query, route, budget)

        # 4. 降级处理
        if not graph_context and not vector_context:
            return self.degradation_manager.handle_degradation(query)

        # 5. 答案合成
        answer = self.synthesizer.synthesize(
            query=query,
            graph_context=graph_context,
            vector_context=vector_context
        )

        return PipelineResult(
            answer=answer,
            sources=[],
            graph_context=graph_context,
            vector_context=vector_context,
            metadata={"route": route}
        )

    def _fetch_graph_context(self, query: str, route: dict, budget: dict) -> dict:
        """获取图上下文"""
        max_nodes = budget.get("max_graph_nodes", 50)
        # 简化的图检索逻辑
        return {}

    def _fetch_vector_context(self, query: str, route: dict, budget: dict) -> List[dict]:
        """获取向量上下文"""
        max_results = budget.get("max_vector_results", 20)
        # 简化的向量检索逻辑
        return []
