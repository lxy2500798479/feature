"""
文本嵌入模块
"""
import hashlib
from collections import OrderedDict
from typing import List, Optional, Dict
import numpy as np

from src.core.models import ChunkNode
from src.config import settings
from src.utils.logger import logger


class TextEmbedder:
    """文本嵌入器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        self.mode = self.config.get("mode", settings.EMBEDDING_MODE)
        self.model_name = self.config.get("model", settings.EMBEDDING_MODEL)
        self.batch_size = self.config.get("batch_size", settings.EMBEDDING_BATCH_SIZE)
        self.dimension = self.config.get("dimension", settings.EMBEDDING_DIMENSION)
        
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        
        self.embedder = None
        self._init_embedder()
    
    def _init_embedder(self):
        """初始化嵌入器"""
        if self.mode == "remote":
            try:
                from src.embedding.remote_embedder import RemoteEmbedder
                self.embedder = RemoteEmbedder()
                logger.info(f"已初始化远程嵌入器: {settings.EMBEDDING_SERVICE_URL}")
            except Exception as e:
                logger.error(f"远程嵌入器初始化失败: {e}")
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(self.model_name)
                logger.info(f"已加载本地嵌入模型: {self.model_name}")
            except ImportError:
                logger.error("未安装 sentence-transformers")
            except Exception as e:
                logger.error(f"本地嵌入模型加载失败: {e}")
    
    def embed_chunks(
        self,
        chunks: List[ChunkNode],
        concept_mapping: Optional[Dict[str, List[str]]] = None,
        community_mapping: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """为文本块生成嵌入向量（优化大文档内存占用）"""
        import time
        start_time = time.time()
        
        if not self.embedder:
            logger.warning("嵌入器未初始化，返回空嵌入")
            return []
        
        total_chunks = len(chunks)
        logger.info(f"🔢 开始为 {total_chunks} 个文本块生成嵌入向量...")
        
        # 大文档阈值：超过 2000 个 chunks 使用流式批处理
        LARGE_DOC_THRESHOLD = 2000
        
        if total_chunks > LARGE_DOC_THRESHOLD:
            logger.info(f"检测到大文档 ({total_chunks} chunks)，使用流式批处理")
            return self._embed_chunks_batched(chunks, concept_mapping, community_mapping)
        
        # 小文档：正常处理
        texts = [chunk.text for chunk in chunks]
        embeddings = self._encode_texts(texts)
        
        results = []
        for i, chunk in enumerate(chunks):
            concepts = concept_mapping.get(chunk.chunk_id, []) if concept_mapping else []
            
            communities = []
            if community_mapping and concepts:
                communities = list(set([
                    community_mapping.get(concept, -1)
                    for concept in concepts
                ]))
            
            result = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section_id": chunk.section_id,
                "embedding": embeddings[i],
                "text": chunk.text,
                "token_count": chunk.token_count,
                "position": chunk.position,
                "concepts": concepts,
                "communities": communities,
            }
            results.append(result)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ 嵌入生成完成: {len(results)} 个, 耗时 {elapsed:.2f}秒")
        
        return results
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        if hasattr(self.embedder, 'embed_batch'):
            return self.embedder.embed_batch(texts)
        elif hasattr(self.embedder, 'encode'):
            return self.embedder.encode(texts)
        else:
            raise ValueError("嵌入器不支持批量编码")

    def embed_single(self, text: str) -> List[float]:
        """为单条文本生成嵌入向量（查询时使用）"""
        if not self.embedder:
            logger.warning("嵌入器未初始化，返回零向量")
            return [0.0] * self.dimension

        result = self._encode_texts([text])
        vec = result[0]
        if hasattr(vec, 'tolist'):
            return vec.tolist()
        return list(vec)

    def _embed_chunks_batched(
        self,
        chunks: List[ChunkNode],
        concept_mapping: Optional[Dict[str, List[str]]] = None,
        community_mapping: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """流式批处理大文档嵌入（降低内存峰值）"""
        import time
        
        start_time = time.time()
        results = []
        
        # 每批处理 500 个 chunks
        BATCH_SIZE = 500
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"大文档嵌入分批处理: {len(chunks)} chunks → {total_batches} 批次")
        
        for batch_idx in range(total_batches):
            batch_start = time.time()
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            # 提取文本
            texts = [chunk.text for chunk in batch_chunks]
            
            # 生成嵌入
            embeddings = self._encode_texts(texts)
            
            # 构建结果
            for i, chunk in enumerate(batch_chunks):
                concepts = concept_mapping.get(chunk.chunk_id, []) if concept_mapping else []
                
                communities = []
                if community_mapping and concepts:
                    communities = list(set([
                        community_mapping.get(concept, -1)
                        for concept in concepts
                    ]))
                
                result = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "section_id": chunk.section_id,
                    "embedding": embeddings[i],
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "position": chunk.position,
                    "concepts": concepts,
                    "communities": communities,
                }
                results.append(result)
            
            # 释放内存
            del texts
            del embeddings
            
            batch_time = time.time() - batch_start
            if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                logger.info(
                    f"  嵌入批次进度: {batch_idx + 1}/{total_batches} "
                    f"({(batch_idx + 1) * 100 // total_batches}%), "
                    f"本批耗时: {batch_time:.2f}秒"
                )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ 大文档嵌入完成: {len(results)} 个, 总耗时 {elapsed:.2f}秒")
        
        return results
