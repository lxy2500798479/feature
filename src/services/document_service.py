"""
文档处理服务
"""
import time
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.config import settings
from src.core.models import ParsedDocument
from src.parsers import ParserRegistry
from src.storage.nebula_client import NebulaClient
from src.storage.vector_client import MilvusClient
from src.graph.concept_graph_builder import ConceptGraphBuilder
from src.embedding.text_embedder import TextEmbedder
from src.services.image_service import ImageService
from src.utils.logger import logger


class DocumentService:
    """文档处理服务"""
    
    def __init__(self):
        # 初始化客户端
        self.nebula_client = NebulaClient()
        self.nebula_client.connect()
        self.vector_client = MilvusClient()
        self.vector_client.connect()
        
        # 初始化组件
        concept_config = {}
        if getattr(settings, "CONCEPT_GRAPH_N_PROCESS", None) is not None:
            concept_config["n_process"] = settings.CONCEPT_GRAPH_N_PROCESS
        if getattr(settings, "CONCEPT_GRAPH_COOCCUR_WORKERS", None) is not None:
            concept_config["cooccur_workers"] = settings.CONCEPT_GRAPH_COOCCUR_WORKERS
        concept_config["max_edges_per_node"] = getattr(
            settings, "CONCEPT_GRAPH_MAX_EDGES_PER_NODE", 20
        )
        
        self.concept_builder = ConceptGraphBuilder(concept_config)
        self.text_embedder = TextEmbedder()
        self.image_service = ImageService()
        
        self.async_embedding_enabled = getattr(settings, 'ASYNC_EMBEDDING_ENABLED', True)
        
        # 文档缓存（异步模式用）
        from src.services.document_cache import DocumentCache
        self._doc_cache = DocumentCache()
        
        logger.info("DocumentService 初始化完成")
    
    def process_document(self, file_path: str, async_mode: bool = True) -> Dict[str, Any]:
        """处理文档"""
        # 1. 解析文档
        parsed_doc = self._parse_document(file_path)
        logger.info(f"✅ 文档解析完成: {parsed_doc.metadata.doc_id}")
        
        # 2. 存储到图数据库
        store_time = self._store_to_graph(parsed_doc)
        logger.info(f"✅ 图数据库存储完成 (耗时: {store_time:.2f}秒)")
        
        # 3. 构建概念图谱
        concept_time, concept_graph = self._build_concept_graph(parsed_doc)
        logger.info(f"✅ 概念图谱构建完成 (耗时: {concept_time:.2f}秒)")
        
        # 4. 生成摘要
        doc_id = parsed_doc.metadata.doc_id
        need_community_summary = settings.SUMMARY_ENABLED and concept_graph.get("communities")
        community_summary_time = 0.0
        
        if need_community_summary:
            community_summary_time = self._generate_community_summary(
                parsed_doc=parsed_doc,
                concept_graph=concept_graph,
                doc_id=doc_id,
                need_community_summary=need_community_summary,
            )
        
        # 更新文档状态
        self._update_document_status(
            doc_id,
            graph_ready=True,
            embeddings_ready=False
        )
        
        processing_time = {
            "store": f"{store_time:.2f}s",
            "concept_graph": f"{concept_time:.2f}s",
            "community_summary": f"{community_summary_time:.2f}s",
        }
        
        if async_mode:
            self._doc_cache.put(doc_id, (parsed_doc, concept_graph))
            
            # 启动后台任务生成向量
            asyncio.create_task(self._generate_embeddings_async(doc_id))
            
            return {
                "doc_id": doc_id,
                "status": "graph_ready",
                "graph_ready": True,
                "embeddings_ready": False,
                "sections_count": len(parsed_doc.sections),
                "chunks_count": len(parsed_doc.chunks),
                "processing_time": processing_time,
                "message": "文档图谱已构建完成，可以开始查询。向量嵌入正在后台生成中。"
            }
        else:
            # 同步模式：生成向量
            embeddings = self._generate_embeddings(parsed_doc, concept_graph)
            self._store_embeddings(embeddings)
            
            return {
                "doc_id": doc_id,
                "status": "completed",
                "graph_ready": True,
                "embeddings_ready": True,
                "sections_count": len(parsed_doc.sections),
                "chunks_count": len(parsed_doc.chunks),
                "processing_time": processing_time,
                "message": "文档处理完成，图谱和向量嵌入均已就绪。"
            }
    
    def _parse_document(self, file_path: str) -> ParsedDocument:
        """解析文档"""
        parser = ParserRegistry.get_suitable_parser(file_path)
        if not parser:
            raise ValueError(f"找不到合适的解析器: {file_path}")
        
        return parser.parse(file_path)
    
    def _store_to_graph(self, parsed_doc: ParsedDocument) -> float:
        """存储到图数据库"""
        import time
        start = time.time()
        
        # 插入文档节点
        self.nebula_client.insert_document(parsed_doc.metadata)
        
        # 插入章节节点
        self.nebula_client.insert_sections(parsed_doc.sections)
        
        # 插入文本块节点
        self.nebula_client.insert_chunks(parsed_doc.chunks)
        
        # 插入边关系
        self.nebula_client.insert_edges(parsed_doc.edges)
        
        return time.time() - start
    
    def _build_concept_graph(self, parsed_doc: ParsedDocument) -> tuple:
        """构建概念图谱"""
        import time
        start = time.time()
        
        concept_graph = self.concept_builder.build_from_chunks(
            chunks=parsed_doc.chunks,
            doc_id=parsed_doc.metadata.doc_id
        )
        
        # 存储到图数据库
        self.nebula_client.insert_concept_graph(
            doc_id=parsed_doc.metadata.doc_id,
            nodes=concept_graph["nodes"],
            edges=concept_graph["edges"],
        )
        
        elapsed = time.time() - start
        return elapsed, concept_graph
    
    def _generate_community_summary(
        self,
        parsed_doc: ParsedDocument,
        concept_graph: Dict,
        doc_id: str,
        need_community_summary: bool,
    ) -> float:
        """生成社区摘要"""
        if not need_community_summary:
            return 0.0
        
        try:
            from src.services.community_summary_service import CommunitySummaryService
            
            step_start = time.time()
            community_svc = CommunitySummaryService()
            community_summaries = community_svc.generate_summaries(
                concept_graph_result=concept_graph,
                chunks=parsed_doc.chunks,
                doc_id=doc_id,
            )
            if community_summaries:
                self.nebula_client.store_community_summaries(
                    doc_id=doc_id,
                    summaries=community_summaries,
                )
            community_summary_time = time.time() - step_start
            logger.info(f"✅ 社区摘要生成完成 (耗时: {community_summary_time:.2f}秒)")
            return community_summary_time
        except Exception as e:
            logger.error(f"社区摘要生成失败: {e}")
            return 0.0
    
    async def _generate_embeddings_async(self, doc_id: str):
        """异步生成向量嵌入"""
        await asyncio.sleep(0.1)  # 让出控制权
        
        cached = self._doc_cache.acquire(doc_id)
        if cached is None:
            logger.error(f"文档数据未缓存: {doc_id}")
            return
        
        parsed_doc, concept_graph = cached
        
        try:
            embeddings = self._generate_embeddings(parsed_doc, concept_graph)
            self._store_embeddings(embeddings)
            
            self._update_document_status(
                doc_id,
                embeddings_ready=True,
                embedding_task_status="completed"
            )
            logger.info(f"🎉 向量嵌入生成完成: {doc_id}")
        except Exception as e:
            logger.error(f"向量嵌入生成失败: {e}")
            self._update_document_status(
                doc_id,
                embedding_task_status="failed",
                embedding_error=str(e)
            )
        finally:
            self._doc_cache.release(doc_id)
    
    def _generate_embeddings(self, parsed_doc: ParsedDocument, concept_graph: Dict) -> List[Dict]:
        """生成向量嵌入"""
        import time
        start = time.time()
        
        # 构建概念映射
        concept_mapping = {}
        community_mapping = concept_graph.get("communities", {})
        
        for node in concept_graph.get("nodes", []):
            concept = node.get("phrase", "")
            community = node.get("community", -1)
            # 简化：只关联到所有 chunk
            for chunk in parsed_doc.chunks:
                if concept in chunk.text:
                    if chunk.chunk_id not in concept_mapping:
                        concept_mapping[chunk.chunk_id] = []
                    concept_mapping[chunk.chunk_id].append(concept)
        
        embeddings = self.text_embedder.embed_chunks(
            chunks=parsed_doc.chunks,
            concept_mapping=concept_mapping,
            community_mapping=community_mapping,
        )
        
        elapsed = time.time() - start
        logger.info(f"✅ 向量生成完成 (耗时: {elapsed:.2f}秒)")
        
        return embeddings
    
    def _store_embeddings(self, embeddings: List[Dict]):
        """存储向量嵌入"""
        # 添加 ID
        for e in embeddings:
            e["id"] = f"{e['doc_id']}_{e['chunk_id']}"
        
        self.vector_client.insert_embeddings(embeddings)
        logger.info(f"✅ 向量存储完成 ({len(embeddings)} 条)")
    
    def _update_document_status(self, doc_id: str, **kwargs):
        """更新文档状态"""
        self.nebula_client.update_document(doc_id, kwargs)
