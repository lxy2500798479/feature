"""
向量数据库客户端 - Milvus
"""
from typing import List, Dict, Optional
import numpy as np

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

from src.config import settings
from src.utils.logger import logger


class MilvusClient:
    """Milvus 向量数据库客户端"""
    
    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = "finalrag_chunks"
        self.dimension = settings.EMBEDDING_DIMENSION
        
        self.collection = None
        self._connected = False
    
    def connect(self):
        """连接到 Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self._connected = True
            logger.info(f"已连接到 Milvus: {self.host}:{self.port}")
        
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise
    
    def init_collection(self):
        """初始化集合"""
        if not self._connected:
            self.connect()
        
        if utility.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' 已存在")
            self.collection = Collection(self.collection_name)
            try:
                self.collection.load()
                logger.info(f"Collection '{self.collection_name}' 已预加载到内存")
            except Exception as e:
                logger.warning(f"Collection 预加载失败: {e}")
            return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="section_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="token_count", dtype=DataType.INT64),
            FieldSchema(name="position", dtype=DataType.INT64),
            FieldSchema(name="concepts", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="communities", dtype=DataType.VARCHAR, max_length=256),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="FinalRAG text chunks with embeddings"
        )
        
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info(f"已创建 collection '{self.collection_name}'")
    
    def insert_embeddings(self, embeddings: List[Dict]):
        """插入嵌入向量"""
        if not self.collection:
            self.init_collection()
        
        data = [
            [
                e["id"],
                e["chunk_id"],
                e["doc_id"],
                e.get("section_id", ""),
                e["text"],
                e["embedding"].tolist() if hasattr(e["embedding"], 'tolist') else e["embedding"],
                e.get("token_count", 0),
                e.get("position", 0),
                e.get("concepts", "[]"),
                e.get("communities", "[]"),
            ]
            for e in embeddings
        ]
        
        self.collection.insert(data)
        self.collection.flush()
        logger.info(f"已插入 {len(embeddings)} 条向量")
    
    def search(
        self,
        query_vector: List[float],
        doc_id: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """向量检索"""
        if not self.collection:
            self.init_collection()
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        expr = f'doc_id == "{doc_id}"' if doc_id else None
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "doc_id", "text", "section_id"]
        )
        
        return [
            {
                "id": r.id,
                "distance": r.distance,
                "chunk_id": r.entity.get("chunk_id"),
                "doc_id": r.entity.get("doc_id"),
                "text": r.entity.get("text"),
                "section_id": r.entity.get("section_id"),
            }
            for r in results[0]
        ]
