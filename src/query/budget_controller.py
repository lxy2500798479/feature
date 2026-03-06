"""
预算控制器 - 定义查询预算配置
"""

# 预算配置文件
BUDGET_PROFILES = {
    "low": {
        "max_graph_nodes": 10,
        "max_vector_results": 5,
        "enable_rerank": False,
        "enable_summary": False,
    },
    "medium": {
        "max_graph_nodes": 50,
        "max_vector_results": 20,
        "enable_rerank": True,
        "enable_summary": True,
    },
    "high": {
        "max_graph_nodes": 100,
        "max_vector_results": 50,
        "enable_rerank": True,
        "enable_summary": True,
    },
}


def get_budget_profile(profile_name: str) -> dict:
    """获取预算配置"""
    return BUDGET_PROFILES.get(profile_name, BUDGET_PROFILES["medium"])
