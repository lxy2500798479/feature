"""
查询路由器 - 基于设计文档的 CoE 导航路由

查询类型:
- factual: 事实型  → 向量检索 + Meta-KG 层级导航（不用概念图）
- relational: 关联型 → 向量检索 + 图谱遍历 + 增强推理
- global: 全局型   → 社区导航 + 向量 + 层级导航
"""
from typing import Dict, Any
from src.utils.logger import logger

# 关联型查询的关键词特征
RELATIONAL_KEYWORDS = [
    "关系", "联系", "影响", "相关", "比较", "区别", "差异",
    "如何影响", "为什么", "原因", "导致", "因果",
    "关联", "依赖", "结合", "协同", "对比", "versus", "vs"
]

# 全局型查询的关键词特征
GLOBAL_KEYWORDS = [
    "总结", "概述", "整体", "全部", "所有", "总体",
    "overview", "summary", "全局", "整个", "宏观",
    "趋势", "分布", "统计", "整合", "综合"
]


class QueryRouter:
    """查询路由器 - 决定查询的处理策略"""

    def __init__(self):
        self.strategies = ["factual", "relational", "global"]

    def route(self, query: str, override_query_type: str = None) -> Dict[str, Any]:
        """决定查询的路由策略

        Args:
            query: 查询文本
            override_query_type: 强制指定查询类型（前端传入）

        Returns:
            路由决策结果
        """
        # 前端强制指定查询类型
        if override_query_type and override_query_type in self.strategies:
            query_type = override_query_type
            logger.info(f"路由: 使用强制类型 '{query_type}'")
        else:
            query_type = self._classify(query)
            logger.info(f"路由: 自动识别类型 '{query_type}' (query={query[:40]}...)")

        route_config = {
            "factual": {
                "query_type": "factual",
                "use_vector": True,
                "use_graph": False,   # 不触发图谱遍历
                "use_coe": True,      # 启用层级导航
                "use_lazy_enhance": False,
                "use_community": False,
                "confidence": 0.9,
                "description": "向量检索 + Meta-KG 层级导航"
            },
            "relational": {
                "query_type": "relational",
                "use_vector": True,
                "use_graph": True,    # 触发图谱遍历
                "use_coe": True,
                "use_lazy_enhance": True,  # 按需增强
                "use_community": False,
                "confidence": 0.85,
                "description": "向量检索 + 图谱遍历 + 增强推理"
            },
            "global": {
                "query_type": "global",
                "use_vector": True,
                "use_graph": True,
                "use_coe": True,
                "use_lazy_enhance": False,
                "use_community": True,    # 使用社区导航
                "confidence": 0.8,
                "description": "社区导航 + 向量 + 层级导航"
            },
        }

        return route_config.get(query_type, route_config["factual"])

    def _classify(self, query: str) -> str:
        """自动分类查询类型"""
        query_lower = query.lower()

        # 检查全局型关键词
        if any(kw in query_lower for kw in GLOBAL_KEYWORDS):
            return "global"

        # 检查关联型关键词
        if any(kw in query_lower for kw in RELATIONAL_KEYWORDS):
            return "relational"

        # 默认为事实型
        return "factual"
