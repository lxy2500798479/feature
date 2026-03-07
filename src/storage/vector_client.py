"""
向量数据库客户端 - Milvus
"""
from typing import List, Dict, Optional
import numpy as np
import json

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
            # 检查现有 collection 的维度是否匹配
            try:
                existing_collection = Collection(self.collection_name)
                existing_schema = existing_collection.schema
                existing_dim = None
                logger.info(f"现有 schema 字段: {existing_schema.fields}")
                for field in existing_schema.fields:
                    if field.name == "embedding":
                        # 尝试多种方式获取维度
                        existing_dim = field.params.get("dim") if hasattr(field, 'params') else None
                        if not existing_dim:
                            # 可能是新的 API 结构
                            existing_dim = getattr(field, 'dimension', None)
                        logger.info(f"现有 embedding 维度: {existing_dim}, 配置维度: {self.dimension}")
                        break
                
                if existing_dim and existing_dim != self.dimension:
                    logger.warning(f"现有 collection 维度 ({existing_dim}) 与配置不匹配 ({self.dimension})，正在删除并重建...")
                    utility.drop_collection(self.collection_name)
                elif existing_dim:
                    self.collection = existing_collection
                    try:
                        self.collection.load()
                        logger.info(f"Collection '{self.collection_name}' 已预加载到内存")
                    except Exception as e:
                        logger.warning(f"Collection 预加载失败: {e}")
                    return
                else:
                    logger.warning(f"无法获取现有 collection 维度，正在删除并重建...")
                    utility.drop_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"检查 collection 维度失败: {e}，正在删除并重建...")
                try:
                    utility.drop_collection(self.collection_name)
                except:
                    pass
        
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
        
        # 调试：检查第一个向量的维度
        if embeddings:
            first_emb = embeddings[0]["embedding"]
            if hasattr(first_emb, 'tolist'):
                first_emb = first_emb.tolist()
            logger.info(f"调试: 第一个向量维度 = {len(first_emb)}, 配置维度 = {self.dimension}")
        
        # Milvus 插入格式：按列 [[field1_values], [field2_values], ...]
        # 确保 concepts 和 communities 是字符串类型
        data = [
            [e["id"] for e in embeddings],
            [e["chunk_id"] for e in embeddings],
            [e["doc_id"] for e in embeddings],
            [e.get("section_id", "") for e in embeddings],
            [e["text"] for e in embeddings],
            [e["embedding"].tolist() if hasattr(e["embedding"], 'tolist') else e["embedding"] for e in embeddings],
            [e.get("token_count", 0) for e in embeddings],
            [e.get("position", 0) for e in embeddings],
            [json.dumps(e.get("concepts", [])) if isinstance(e.get("concepts"), (dict, list)) else str(e.get("concepts", "[]")) for e in embeddings],
            [json.dumps(e.get("communities", [])) if isinstance(e.get("communities"), (dict, list)) else str(e.get("communities", "[]")) for e in embeddings],
        ]
        
        self.collection.insert(data)
        self.collection.flush()
        logger.info(f"已插入 {len(embeddings)} 条向量")
    
    def search(
        self,
        query_vector: List[float],
        doc_id: Optional[str] = None,
        top_k: int = 10,
        section_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """向量检索
        
        Args:
            query_vector: 查询向量
            doc_id: 按文档 ID 过滤
            top_k: 返回条数
            section_ids: 按 Section ID 列表过滤（CoE Step 3 精确检索用）
        """
        if not self.collection:
            self.init_collection()
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        if section_ids:
            # 在指定 sections 内检索
            ids_quoted = ", ".join(f'"{sid}"' for sid in section_ids)
            expr = f"section_id in [{ids_quoted}]"
        elif doc_id:
            expr = f'doc_id == "{doc_id}"'
        else:
            expr = None
        
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

    def reset_collection(self):
        """删除并重建 collection"""
        if not self._connected:
            self.connect()

        # 删除已存在的 collection
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.warning(f"已删除 collection: {self.collection_name}")

        # 重新创建
        self.init_collection()
        logger.warning(f"已重建 collection: {self.collection_name}")
