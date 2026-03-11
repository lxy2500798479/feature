"""
向量数据库客户端 - Milvus

修复内容（对比原版）：
===========================================================================
【问题5】Milvus schema 缺少 section_title、doc_title 字段

  旧 schema：chunk_id / doc_id / section_id / text / embedding / ...
  → answer_synthesizer 拿不到章节名和文档名，prompt 里没有来源信息
  → 用户问"第几章提到的"，LLM 无法回答

  新 schema：新增两个 VARCHAR 字段：
    section_title: VARCHAR(512)  章节标题（如"第一章 唐三觉醒"）
    doc_title:     VARCHAR(256)  文档标题（如"斗罗大陆"）

  同步修改：
  - insert_embeddings()：写入时从 embedding dict 读取这两个字段（无则填空串）
  - search()：output_fields 增加 section_title、doc_title
  - search() 返回结果：dict 中增加 section_title、doc_title 字段

  ⚠️ 注意：新增字段需要重建 collection（旧数据不兼容）
     重建方法：调用 vector_client.reset_collection() 或删除已有 collection 后重启

  document_service._generate_embeddings_sync 在调用 embed_chunks 后，
  需要把 section.title 和 doc.title 注入到 embedding dict：
    r["section_title"] = section_title_map.get(r["section_id"], "")
    r["doc_title"] = doc_title
  （已在 document_service.py 中一并处理）
===========================================================================
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
        """初始化集合（幂等，含维度检查和新字段检查）"""
        if not self._connected:
            self.connect()

        if utility.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' 已存在，检查 schema...")
            try:
                existing_collection = Collection(self.collection_name)
                existing_schema = existing_collection.schema
                existing_dim = None
                has_section_title = False
                has_doc_title = False

                for field in existing_schema.fields:
                    if field.name == "embedding":
                        existing_dim = field.params.get("dim") if hasattr(field, 'params') else None
                        if not existing_dim:
                            existing_dim = getattr(field, 'dimension', None)
                        logger.info(f"现有 embedding 维度: {existing_dim}, 配置维度: {self.dimension}")
                    if field.name == "section_title":
                        has_section_title = True
                    if field.name == "doc_title":
                        has_doc_title = True

                dim_mismatch = existing_dim and existing_dim != self.dimension
                schema_outdated = not has_section_title or not has_doc_title

                if dim_mismatch or schema_outdated:
                    reason = []
                    if dim_mismatch:
                        reason.append(f"维度不匹配({existing_dim}!={self.dimension})")
                    if schema_outdated:
                        missing = []
                        if not has_section_title:
                            missing.append("section_title")
                        if not has_doc_title:
                            missing.append("doc_title")
                        reason.append(f"缺少字段{missing}")
                    logger.warning(f"Collection schema 需要重建: {', '.join(reason)}")
                    utility.drop_collection(self.collection_name)
                else:
                    self.collection = existing_collection
                    try:
                        self.collection.load()
                        logger.info(f"Collection '{self.collection_name}' 已加载")
                    except Exception as e:
                        logger.warning(f"Collection 预加载失败: {e}")
                    return

            except Exception as e:
                logger.warning(f"检查 collection schema 失败: {e}，重建...")
                try:
                    utility.drop_collection(self.collection_name)
                except Exception:
                    pass

        # 创建新 collection（含 section_title、doc_title）
        fields = [
            FieldSchema(name="id",            dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="chunk_id",       dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="doc_id",         dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="section_id",     dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="section_title",  dtype=DataType.VARCHAR, max_length=512),   # 新增
            FieldSchema(name="doc_title",      dtype=DataType.VARCHAR, max_length=256),   # 新增
            FieldSchema(name="text",           dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding",      dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="token_count",    dtype=DataType.INT64),
            FieldSchema(name="position",       dtype=DataType.INT64),
            FieldSchema(name="concepts",       dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="communities",    dtype=DataType.VARCHAR, max_length=256),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="FinalRAG text chunks with embeddings (v2: +section_title, +doc_title)"
        )

        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        self.collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
        )

        logger.info(
            f"已创建 collection '{self.collection_name}' "
            f"(dim={self.dimension}, +section_title, +doc_title)"
        )

    def insert_embeddings(self, embeddings: List[Dict]):
        """
        插入嵌入向量

        embeddings 每条 dict 支持字段：
          必须：id, chunk_id, doc_id, text, embedding
          可选：section_id, section_title, doc_title, token_count, position, concepts, communities
        """
        if not self.collection:
            self.init_collection()

        if not embeddings:
            return

        # 调试日志
        first_emb = embeddings[0]["embedding"]
        if hasattr(first_emb, 'tolist'):
            first_emb = first_emb.tolist()
        logger.info(f"插入向量: {len(embeddings)} 条, 维度={len(first_emb)}")

        # 按列组装（顺序必须与 schema fields 一致）
        data = [
            [e["id"] for e in embeddings],
            [e["chunk_id"] for e in embeddings],
            [e["doc_id"] for e in embeddings],
            [e.get("section_id", "") for e in embeddings],
            [e.get("section_title", "")[:512] for e in embeddings],   # 新增
            [e.get("doc_title", "")[:256] for e in embeddings],        # 新增
            [e["text"] for e in embeddings],
            [
                e["embedding"].tolist() if hasattr(e["embedding"], 'tolist') else e["embedding"]
                for e in embeddings
            ],
            [e.get("token_count", 0) for e in embeddings],
            [e.get("position", 0) for e in embeddings],
            [
                json.dumps(e.get("concepts", [])) if isinstance(e.get("concepts"), (dict, list))
                else str(e.get("concepts", "[]"))
                for e in embeddings
            ],
            [
                json.dumps(e.get("communities", [])) if isinstance(e.get("communities"), (dict, list))
                else str(e.get("communities", "[]"))
                for e in embeddings
            ],
        ]

        self.collection.insert(data)
        self.collection.flush()
        logger.info(f"已插入 {len(embeddings)} 条向量")

    def search(
        self,
        query_vector: List[float],
        doc_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 10,
        section_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        向量检索

        返回结果新增 section_title、doc_title 字段，
        answer_synthesizer 可直接使用，无需二次查询。
        """
        if not self.collection:
            self.init_collection()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        if section_ids:
            ids_quoted = ", ".join(f'"{sid}"' for sid in section_ids)
            expr = f"section_id in [{ids_quoted}]"
        elif doc_id:
            expr = f'doc_id == "{doc_id}"'
        elif doc_ids:
            ids_quoted = ", ".join(f'"{did}"' for did in doc_ids)
            expr = f"doc_id in [{ids_quoted}]"
        else:
            expr = None

        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[
                "chunk_id", "doc_id", "text", "section_id",
                "section_title", "doc_title",   # 新增
            ]
        )

        return [
            {
                "id": r.id,
                "distance": r.distance,
                "chunk_id": r.entity.get("chunk_id"),
                "doc_id": r.entity.get("doc_id"),
                "text": r.entity.get("text"),
                "section_id": r.entity.get("section_id"),
                "section_title": r.entity.get("section_title", ""),   # 新增
                "doc_title": r.entity.get("doc_title", ""),            # 新增
            }
            for r in results[0]
        ]

    def query_by_chunk_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """根据 chunk_id 列表查询 chunks（用于子图增强检索）"""
        if not self.collection or not chunk_ids:
            return []
        try:
            ids_quoted = ", ".join(
                f'"{str(cid).replace(chr(34), "")}"' for cid in chunk_ids[:20]
            )
            expr = f"chunk_id in [{ids_quoted}]"
            hits = self.collection.query(
                expr=expr,
                output_fields=["chunk_id", "doc_id", "section_id", "section_title", "doc_title", "text"],
                limit=len(chunk_ids) + 10,
            )
            return [
                {
                    "chunk_id": h.get("chunk_id", ""),
                    "doc_id": h.get("doc_id", ""),
                    "section_id": h.get("section_id", ""),
                    "section_title": h.get("section_title", ""),
                    "doc_title": h.get("doc_title", ""),
                    "text": h.get("text", ""),
                    "score": 0.0,
                    "source": "graph_traversal",
                }
                for h in (hits or [])
            ]
        except Exception as e:
            logger.warning(f"query_by_chunk_ids 失败: {e}")
            return []

    def reset_collection(self):
        """删除并重建 collection（用于 schema 升级后的迁移）"""
        if not self._connected:
            self.connect()
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.warning(f"已删除 collection: {self.collection_name}")
        self.init_collection()
        logger.warning(f"已重建 collection: {self.collection_name}")