"""
Query Service - 查询服务
"""
from typing import List, Optional, Dict, Any

from src.core.models import QueryRequest, QueryResponse
from src.storage.nebula_client import NebulaClient
from src.storage.vector_client import MilvusClient
from src.embedding.text_embedder import TextEmbedder
from src.utils.logger import logger


class QueryService:
    """查询服务"""
    
    def __init__(self):
        self.nebula_client = NebulaClient()
        self.nebula_client.connect()
        
        self.vector_client = MilvusClient()
        self.vector_client.connect()
        
        self.embedder = TextEmbedder()
    
    def query(self, request: QueryRequest) -> QueryResponse:
        """执行查询"""
        query_embedding = self.embedder.embed_single(request.query)
        
        # 向量检索
        vector_results = self.vector_client.search(
            query_vector=query_embedding,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # 构建返回结果
        results = []
        for item in vector_results:
            results.append({
                "chunk_id": item.get("id"),
                "text": item.get("text"),
                "score": item.get("score"),
            })
        
        return QueryResponse(
            query=request.query,
            results=results,
            total=len(results),
            took_ms=0.0,
            strategy_used="vector",
            degradation_reason=""
        )
