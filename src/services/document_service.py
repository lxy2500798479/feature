"""
文档处理服务 - 可插拔模块化设计
"""
import time
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.config import settings
from src.core.models import ParsedDocument, GraphNode, GraphEdge
from src.parsers import ParserRegistry
from src.storage.nebula_client import NebulaClient
from src.storage.vector_client import MilvusClient
from src.embedding.text_embedder import TextEmbedder
from src.utils.logger import logger


class DocumentService:
    """文档处理服务 - 可插拔模块化设计"""

    def __init__(self):
        # 核心组件 - 始终初始化
        self.nebula_client = None
        self.vector_client = None
        self.text_embedder = None

        # 可插拔模块 - 懒加载
        self._concept_builder = None
        self._image_service = None
        self._doc_cache = None

        # 模块开关状态
        self._modules_initialized = False

        # 初始化核心组件
        self._init_core_components()

        logger.info("DocumentService 初始化完成")

    def _init_core_components(self):
        """初始化核心组件"""
        self.nebula_client = NebulaClient()
        self.nebula_client.connect()
        self.vector_client = MilvusClient()
        self.vector_client.connect()
        self.text_embedder = TextEmbedder()

        logger.info("核心组件初始化完成")

    @property
    def concept_builder(self):
        """概念图谱构建器 - 懒加载"""
        if self._concept_builder is None:
            if not getattr(settings, 'ENABLE_CONCEPT_GRAPH', True):
                logger.info("概念图谱模块已禁用")
                return None

            from src.graph.concept_graph_builder import ConceptGraphBuilder
            concept_config = {}
            if getattr(settings, "CONCEPT_GRAPH_N_PROCESS", None) is not None:
                concept_config["n_process"] = settings.CONCEPT_GRAPH_N_PROCESS
            if getattr(settings, "CONCEPT_GRAPH_COOCCUR_WORKERS", None) is not None:
                concept_config["cooccur_workers"] = settings.CONCEPT_GRAPH_COOCCUR_WORKERS
            concept_config["max_edges_per_node"] = getattr(
                settings, "CONCEPT_GRAPH_MAX_EDGES_PER_NODE", 20
            )
            self._concept_builder = ConceptGraphBuilder(concept_config)
            logger.info("概念图谱模块已加载")
        return self._concept_builder

    @property
    def image_service(self):
        """图片处理服务 - 懒加载"""
        if self._image_service is None:
            if not getattr(settings, 'ENABLE_IMAGE_ENTITY_EXTRACTION', True):
                logger.info("图片实体提取模块已禁用")
                return None

            from src.services.image_service import ImageService
            self._image_service = ImageService()
            logger.info("图片实体提取模块已加载")
        return self._image_service

    @property
    def doc_cache(self):
        """文档缓存 - 懒加载"""
        if self._doc_cache is None:
            from src.services.document_cache import DocumentCache
            self._doc_cache = DocumentCache()
        return self._doc_cache

    @property
    def async_embedding_enabled(self):
        """异步嵌入开关"""
        return getattr(settings, 'ASYNC_EMBEDDING_ENABLED', True)

    @property
    def summary_enabled(self):
        """摘要生成开关"""
        return getattr(settings, 'ENABLE_SUMMARY', True)

    def process_document(self, file_path: str, async_mode: bool = True) -> Dict[str, Any]:
        """处理文档"""
        # 1. 解析文档
        parsed_doc = self._parse_document(file_path)
        logger.info(f"✅ 文档解析完成: {parsed_doc.metadata.doc_id}")

        # 2. 存储到图数据库
        store_time = self._store_to_graph(parsed_doc)
        logger.info(f"✅ 图数据库存储完成 (耗时: {store_time:.2f}秒)")

        # 3. 构建概念图谱 (可插拔)
        concept_time, concept_graph = 0.0, {}
        if self.concept_builder is not None:
            concept_time, concept_graph = self._build_concept_graph(parsed_doc)
            logger.info(f"✅ 概念图谱构建完成 (耗时: {concept_time:.2f}秒)")

        # 4. 生成摘要 (可插拔)
        doc_id = parsed_doc.metadata.doc_id
        need_community_summary = self.summary_enabled and concept_graph.get("communities")
        community_summary_time = 0.0
        section_summary_time = 0.0

        if need_community_summary:
            community_summary_time = self._generate_community_summary(
                parsed_doc=parsed_doc,
                concept_graph=concept_graph,
                doc_id=doc_id,
                need_community_summary=need_community_summary,
            )

        # 4b. 生成 Section Summary（CoE Step 2 精排用）
        if self.summary_enabled and parsed_doc.sections:
            import threading
            def _bg_section_summary():
                self._generate_section_summaries(parsed_doc)
            threading.Thread(target=_bg_section_summary, daemon=True, name="section-summary").start()
            logger.info(f"Section 摘要生成已在后台启动 ({len(parsed_doc.sections)} 个章节)")

        # 4c. LLM 实体关系抽取（Enhanced-KG） — 异步后台执行
        import threading
        def _bg_entity_extract():
            self._extract_and_store_entities(parsed_doc)
        threading.Thread(target=_bg_entity_extract, daemon=True, name="entity-extract").start()
        logger.info(f"Entity 抽取已在后台启动 ({len(parsed_doc.chunks)} 个 chunk)")

        # 更新文档状态
        self._update_document_status(
            doc_id,
            graph_ready=True,
            embeddings_ready=False
        )

        # 5. 异步生成向量嵌入
        embedding_time = 0.0
        if async_mode and self.async_embedding_enabled:
            embedding_time = self._generate_embeddings_async(parsed_doc)
            logger.info(f"✅ 向量嵌入异步生成中 (预估耗时: {embedding_time:.2f}秒)")
        else:
            embedding_time = self._generate_embeddings_sync(parsed_doc)
            logger.info(f"✅ 向量嵌入生成完成 (耗时: {embedding_time:.2f}秒)")

        # 6. 图片实体提取 (可插拔)
        image_entity_time = 0.0
        if self.image_service is not None and getattr(settings, 'IMAGE_PROCESSING_ENABLED', True):
            import threading
            def _bg_image_entity():
                self._process_image_entities(parsed_doc)
            threading.Thread(target=_bg_image_entity, daemon=True, name="image-entity").start()
            logger.info("图片实体提取已在后台启动")

        return {
            "doc_id": doc_id,
            "status": "processing" if async_mode else "completed",
            "timing": {
                "parse": 0,
                "store": store_time,
                "concept_graph": concept_time,
                "community_summary": community_summary_time,
                "section_summary": section_summary_time,
                "embedding": embedding_time,
                "image_entity": image_entity_time,
            }
        }

    def _parse_document(self, file_path: str) -> ParsedDocument:
        """解析文档"""
        from src.parsers import ParserRegistry
        parser = ParserRegistry.get_parser(file_path)
        return parser.parse(file_path)

    def _store_to_graph(self, parsed_doc: ParsedDocument) -> float:
        """存储到图数据库"""
        import time
        start = time.time()

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

        concept_graph = self.concept_builder.build(
            doc_id=parsed_doc.metadata.doc_id,
            chunks=parsed_doc.chunks,
            sections=parsed_doc.sections
        )

        # 存储到图数据库
        if concept_graph.get("nodes"):
            self.nebula_client.insert_concepts(concept_graph["nodes"])
        if concept_graph.get("edges"):
            self.nebula_client.insert_concept_edges(concept_graph["edges"])

        return time.time() - start, concept_graph

    def _generate_community_summary(self, parsed_doc, concept_graph, doc_id, need_community_summary):
        """生成社区摘要"""
        from src.services.summary_service import SummaryService
        summary_service = SummaryService()
        return summary_service.generate_community_summary(
            parsed_doc=parsed_doc,
            concept_graph=concept_graph,
            doc_id=doc_id,
            need_community_summary=need_community_summary,
        )

    def _generate_section_summaries(self, parsed_doc: ParsedDocument):
        """生成章节摘要"""
        from src.services.summary_service import SummaryService
        summary_service = SummaryService()
        summary_service.generate_section_summaries(parsed_doc)

    def _extract_and_store_entities(self, parsed_doc: ParsedDocument):
        """抽取实体关系并存储"""
        from src.graph.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        extractor.extract_and_store(parsed_doc)

    def _generate_embeddings_async(self, parsed_doc: ParsedDocument) -> float:
        """异步生成向量嵌入"""
        import time
        start = time.time()

        # 存储到缓存，由后台任务处理
        self.doc_cache.put(parsed_doc.metadata.doc_id, parsed_doc)

        from src.services.embedding_task import EmbeddingTask
        task = EmbeddingTask()
        task.submit(parsed_doc.metadata.doc_id)

        return time.time() - start

    def _generate_embeddings_sync(self, parsed_doc: ParsedDocument) -> float:
        """同步生成向量嵌入"""
        import time
        start = time.time()

        for chunk in parsed_doc.chunks:
            embedding = self.text_embedder.embed([chunk.text])
            chunk.embedding_id = self.vector_client.insert(
                collection_name="chunks",
                text=chunk.text,
                vector=embedding[0],
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "section_id": chunk.section_id
                }
            )

        return time.time() - start

    def _process_image_entities(self, parsed_doc: ParsedDocument):
        """处理图片实体"""
        try:
            graph_entities = self.image_service.extract_entities_from_images(parsed_doc)
            if graph_entities:
                self._store_image_entities_to_graph(parsed_doc.metadata.doc_id, graph_entities)
        except Exception as e:
            logger.error(f"图片实体提取失败: {e}")

    def _store_image_entities_to_graph(self, doc_id: str, graph_entities: List[Dict]):
        """将图片中的实体关系存储到图数据库"""
        nodes = []
        edges = []

        for entity_info in graph_entities:
            chunk_id = entity_info.get("chunk_id", "")
            image_type = entity_info.get("image_type", "unknown")
            entities = entity_info.get("entities", [])
            relations = entity_info.get("relations", [])
            description = entity_info.get("description", "")

            # 为每个实体创建节点
            entity_ids = []
            for i, entity_name in enumerate(entities):
                entity_id = f"{chunk_id}_entity_{i}"
                entity_ids.append(entity_id)

                nodes.append(GraphNode(
                    node_id=entity_id,
                    node_type="ImageEntity",
                    doc_id=doc_id,
                    properties={
                        "name": entity_name,
                        "image_type": image_type,
                        "source_chunk_id": chunk_id,
                        "description": description
                    }
                ))

            # 为每个关系创建边
            for relation in relations:
                if len(relation) == 3:
                    src, rel_type, dst = relation

                    # 找到对应的 entity_id
                    src_id = None
                    dst_id = None
                    for i, e in enumerate(entities):
                        if e == src:
                            src_id = entity_ids[i]
                        if e == dst:
                            dst_id = entity_ids[i]

                    if src_id and dst_id:
                        edges.append(GraphEdge(
                            src_id=src_id,
                            dst_id=dst_id,
                            edge_type=rel_type,
                            properties={"source": "image_entity"}
                        ))

        # 批量插入
        if nodes:
            self.nebula_client.insert_nodes(nodes)
        if edges:
            self.nebula_client.insert_edges(edges)

    def _update_document_status(self, doc_id: str, **kwargs):
        """更新文档状态"""
        # TODO: 实现文档状态更新
        pass

    def process_document_sync(self, file_path: str) -> Dict[str, Any]:
        """同步处理文档（完整流程）"""
        return self.process_document(file_path, async_mode=False)

    async def process_document_async(self, file_path: str) -> Dict[str, Any]:
        """异步处理文档"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_document, file_path, True)

    def get_document(self, doc_id: str) -> Optional[ParsedDocument]:
        """获取文档"""
        return self.doc_cache.get(doc_id)

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """列出文档"""
        # TODO: 实现文档列表
        return []

    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        # 删除图数据
        self.nebula_client.delete_document(doc_id)
        # 删除向量数据
        self.vector_client.delete_by_doc_id(doc_id)
        # 删除缓存
        self.doc_cache.delete(doc_id)
        return True
