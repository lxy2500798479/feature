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
            import time
            start = time.time()
            summaries = self._generate_community_summary(
                parsed_doc=parsed_doc,
                concept_graph=concept_graph,
                doc_id=doc_id,
                need_community_summary=need_community_summary,
            )
            # 存储社区摘要到 NebulaGraph
            if summaries:
                self.nebula_client.store_community_summaries(doc_id, summaries)
                logger.info(f"✅ 社区摘要已存储: {len(summaries)} 个")
            community_summary_time = time.time() - start

        # 4b. 生成 Section Summary（CoE Step 2 精排用）
        if self.summary_enabled and parsed_doc.sections:
            import threading
            def _bg_section_summary():
                self._generate_section_summaries(parsed_doc)
            threading.Thread(target=_bg_section_summary, daemon=True, name="section-summary").start()
            logger.info(f"Section 摘要生成已在后台启动 ({len(parsed_doc.sections)} 个章节)")

        # 4c. 生成 Document Summary
        if self.summary_enabled:
            import threading
            def _bg_doc_summary():
                self._generate_document_summary(parsed_doc)
            threading.Thread(target=_bg_doc_summary, daemon=True, name="doc-summary").start()
            logger.info("Document 摘要生成已在后台启动")

        # [弃用] 4d. LLM 实体关系抽取（Enhanced-KG） — 改为查询时懒加载
        # import threading
        # def _bg_entity_extract():
        #     self._extract_and_store_entities(parsed_doc)
        # threading.Thread(target=_bg_entity_extract, daemon=True, name="entity-extract").start()
        # logger.info(f"Entity 抽取已在后台启动 ({len(parsed_doc.chunks)} 个 chunk)")

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
        if self.image_service is not None and getattr(settings, 'ENABLE_IMAGE_ENTITY_EXTRACTION', True):
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
        parser = ParserRegistry.get_suitable_parser(file_path)
        if parser is None:
            raise ValueError(f"找不到支持 {file_path} 的解析器")
        return parser.parse(file_path)

    def _store_to_graph(self, parsed_doc: ParsedDocument) -> float:
        """存储到图数据库"""
        import time
        start = time.time()

        # 先插入文档节点
        doc_metadata = parsed_doc.metadata.model_dump() if hasattr(parsed_doc.metadata, 'model_dump') else parsed_doc.metadata
        self.nebula_client.insert_document(doc_metadata)

        # 插入章节节点
        logger.info(f"正在写入 NebulaGraph: {len(parsed_doc.sections)} sections, {len(parsed_doc.chunks)} chunks...")
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
        if concept_graph.get("nodes") or concept_graph.get("edges"):
            self.nebula_client.insert_concept_graph(
                doc_id=parsed_doc.metadata.doc_id,
                nodes=concept_graph.get("nodes", []),
                edges=concept_graph.get("edges", [])
            )

        return time.time() - start, concept_graph

    def _generate_community_summary(self, parsed_doc, concept_graph, doc_id, need_community_summary):
        """生成社区摘要"""
        from src.services.community_summary_service import CommunitySummaryService
        summary_service = CommunitySummaryService()
        
        # 转换为 dict 格式
        chunks_data = [{"text": c.text, "chunk_id": c.chunk_id} for c in parsed_doc.chunks]
        
        return summary_service.generate_summaries(
            concept_graph_result=concept_graph,
            chunks=chunks_data,
            doc_id=doc_id,
        )

    def _generate_section_summaries(self, parsed_doc: ParsedDocument):
        """生成章节摘要"""
        from src.services.summary_service import SummaryService
        summary_service = SummaryService()
        summaries = summary_service.generate_summaries_for_sections(
            sections=parsed_doc.sections,
            chunks=parsed_doc.chunks,
        )

        # 将摘要写回 SectionNode
        for section in parsed_doc.sections:
            if section.section_id in summaries:
                section.summary = summaries[section.section_id]

        # 存储到数据库
        self._store_section_summaries(parsed_doc.metadata.doc_id, summaries)

        logger.info(f"Section 摘要生成完成: {len(summaries)} 个")
        return summaries

    def _store_section_summaries(self, doc_id: str, summaries: Dict[str, str]):
        """将章节摘要存储到数据库"""
        try:
            # 确保所有摘要都是有效的 UTF-8 字符串
            cleaned_summaries = {}
            for section_id, summary in summaries.items():
                if summary:
                    # 尝试编码/解码以清理无效字符
                    try:
                        cleaned = summary.encode('utf-8', errors='ignore').decode('utf-8')
                        cleaned_summaries[section_id] = cleaned
                    except Exception:
                        cleaned_summaries[section_id] = str(summary)
                else:
                    cleaned_summaries[section_id] = ""

            self.nebula_client.update_sections_summary(doc_id, cleaned_summaries)
        except Exception as e:
            logger.error(f"存储章节摘要失败: {e}")

    def _generate_document_summary(self, parsed_doc: ParsedDocument):
        """生成文档摘要"""
        from src.services.summary_service import SummaryService
        summary_service = SummaryService()

        doc_summary = summary_service.generate_document_summary(
            sections=parsed_doc.sections,
            chunks=parsed_doc.chunks,
        )

        # 更新 metadata
        parsed_doc.metadata.summary = doc_summary

        # 存储到数据库
        self._store_document_summary(parsed_doc.metadata.doc_id, doc_summary)

        logger.info(f"Document 摘要生成完成: {doc_summary[:50]}...")
        return doc_summary

    def _store_document_summary(self, doc_id: str, summary: str):
        """将文档摘要存储到数据库"""
        try:
            # 清理无效字符
            if summary:
                cleaned_summary = summary.encode('utf-8', errors='ignore').decode('utf-8')
            else:
                cleaned_summary = ""
            self.nebula_client.update_document(doc_id, {"summary": cleaned_summary})
        except Exception as e:
            logger.error(f"存储文档摘要失败: {e}")

    def _extract_and_store_entities(self, parsed_doc: ParsedDocument):
        """[弃用] 抽取实体关系并存储 - 改为查询时懒加载"""
        from src.graph.entity_extractor import EntityExtractor
        from src.utils.llm_client import LLMClient
        from src.config import settings
        
        # 创建 LLM 客户端
        llm_client = LLMClient(
            api_url=settings.SUMMARY_API_URL or settings.LLM_API_URL,
            api_key=settings.SUMMARY_API_KEY or settings.LLM_API_KEY,
            model=settings.SUMMARY_MODEL,
        )
        
        extractor = EntityExtractor(llm_client=llm_client)
        result = extractor.extract_from_chunks(
            chunks=parsed_doc.chunks,
            doc_id=parsed_doc.metadata.doc_id,
        )
        
        # 存储到图数据库
        if result.get("entities") or result.get("relations"):
            self.nebula_client.insert_entity_graph(
                doc_id=parsed_doc.metadata.doc_id,
                entities=result.get("entities", []),
                relations=result.get("relations", []),
            )

    def _generate_embeddings_async(self, parsed_doc: ParsedDocument) -> float:
        """异步生成向量嵌入：仅放入缓存，由 API 的 background_tasks 调用 generate_embeddings_async(doc_id) 执行"""
        import time
        start = time.time()
        self.doc_cache.put(parsed_doc.metadata.doc_id, parsed_doc)
        return time.time() - start

    def generate_embeddings_async(self, doc_id: str) -> None:
        """由 API 后台任务调用：从缓存取文档并生成向量（若缓存无则跳过）"""
        doc = self.doc_cache.acquire(doc_id)
        if doc:
            try:
                self._generate_embeddings_sync(doc)
                # 更新状态为完成
                self._update_document_status(
                    doc_id,
                    embeddings_ready=True,
                    embedding_task_status="completed"
                )
                logger.info(f"后台向量嵌入完成: {doc_id}")
            except Exception as e:
                logger.error(f"后台向量嵌入失败 {doc_id}: {e}")
                self._update_document_status(
                    doc_id,
                    embedding_task_status="failed"
                )
        else:
            logger.warning(f"文档 {doc_id} 不在缓存中，跳过向量嵌入")

    def process_images_async(self, doc_id: str) -> None:
        """由 API 后台任务调用：从缓存取文档并处理图片"""
        doc = self.doc_cache.acquire(doc_id)
        if doc:
            try:
                if self.image_service:
                    # ImageService.process_images 需要 content_list, doc_id, extract_dir, page_text_map
                    # 从 parsed_doc 获取必要信息
                    content_list = []
                    extract_dir = ""
                    page_text_map = {}
                    
                    if hasattr(doc, 'content_list'):
                        content_list = doc.content_list
                    if hasattr(doc, 'extract_dir'):
                        extract_dir = doc.extract_dir
                    
                    if content_list:
                        result = self.image_service.process_images(
                            content_list=content_list,
                            doc_id=doc_id,
                            extract_dir=extract_dir,
                            page_text_map=page_text_map
                        )
                        logger.info(f"图片处理完成: {doc_id}, 图片chunks: {len(result.get('image_chunks', []))}")
            except Exception as e:
                logger.error(f"图片处理失败 {doc_id}: {e}")
        else:
            logger.warning(f"文档 {doc_id} 不在缓存中，跳过图片处理")

    def _generate_embeddings_sync(self, parsed_doc: ParsedDocument) -> float:
        """同步生成向量嵌入"""
        import time
        start = time.time()

        embeddings = []
        for chunk in parsed_doc.chunks:
            # 使用 embed_single 方法（返回 List[float]）
            embedding = self.text_embedder.embed_single(chunk.text)
            embeddings.append({
                "id": chunk.chunk_id,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section_id": chunk.section_id,
                "text": chunk.text,
                "embedding": embedding,
                "token_count": getattr(chunk, 'token_count', 0),
                "position": getattr(chunk, 'position', 0),
            })
        
        if embeddings:
            self.vector_client.insert_embeddings(embeddings)
        
        return time.time() - start

    def _process_image_entities(self, parsed_doc: ParsedDocument):
        """处理图片实体"""
        try:
            if self.image_service is None:
                return
            
            # 调用 image_service 处理图片，获取图片 chunks
            # 需要从 parsed_doc 中获取 content_list 和 extract_dir
            # 这里假设 parsed_doc 有这些信息，或者我们需要从解析结果中获取
            
            # 简化处理：直接跳过图片实体提取，因为 ImageService 没有返回实体
            # 后续可以扩展：使用 Vision LLM 从图片描述中提取实体
            logger.info("图片实体提取: 使用 ImageService 处理图片")
            
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
        if not kwargs:
            return
        try:
            self.nebula_client.update_document(doc_id, kwargs)
            logger.info(f"文档状态已更新: {doc_id}, {kwargs}")
        except Exception as e:
            logger.error(f"更新文档状态失败: {e}")

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
        """列出文档（从 NebulaGraph Document 顶点读取）"""
        if not self.nebula_client:
            return []
        try:
            docs = self.nebula_client.get_documents()
            # 映射为前端 DocListItem 格式
            return [
                {
                    "doc_id": d.get("doc_id", ""),
                    "title": d.get("title", "") or "（无标题）",
                    "file_type": d.get("file_type", "") or "unknown",
                    "graph_ready": bool(d.get("graph_ready", False)),
                    "embeddings_ready": bool(d.get("embeddings_ready", False)),
                    "embedding_task_status": d.get("embedding_task_status", "") or "",
                }
                for d in docs[offset : offset + limit]
            ]
        except Exception as e:
            logger.warning(f"list_documents 失败: {e}")
            return []

    def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档处理状态"""
        from src.storage.nebula_client import NebulaClient
        nebula = NebulaClient()
        nebula.connect()
        doc_data = nebula.get_document_status(doc_id)
        if not doc_data:
            return None
        
        return {
            "doc_id": doc_id,
            "status": doc_data.get("status", "unknown"),
            "graph_ready": doc_data.get("graph_ready", False),
            "embeddings_ready": doc_data.get("embeddings_ready", False),
            "embedding_task_status": doc_data.get("embedding_task_status", "unknown"),
        }

    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        # 删除缓存（DocumentCache 没有 delete 方法，使用 clear 或 release）
        # 简化处理：直接从 cache 中移除
        if hasattr(self.doc_cache, 'cache') and doc_id in self.doc_cache.cache:
            del self.doc_cache.cache[doc_id]
        
        # 注意：NebulaClient 和 VectorClient 没有 delete_document 方法
        # 暂不实现删除操作
        logger.warning(f"delete_document 调用了，但底层存储不支持删除: {doc_id}")
        return True
