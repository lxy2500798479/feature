"""
延迟增强器 - 按需增强查询
"""
from typing import Optional, Dict, Any
from src.utils.logger import logger


class LazyEnhancer:
    """延迟增强器 - 按需增强查询"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def enhance(self, query: str) -> Dict[str, Any]:
        """增强查询"""
        return {
            "enhanced_query": query,
            "keywords": [],
            "entities": []
        }

    def extract_keywords(self, query: str) -> list:
        """提取关键词"""
        return []

    def extract_entities(self, query: str) -> list:
        """提取实体"""
        return []
