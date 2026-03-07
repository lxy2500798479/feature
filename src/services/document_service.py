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
        section_summary_time = 0.0

        if need_community_summary:
            community_summary_time = self._generate_community_summary(
                parsed_doc=parsed_doc,
                concept_graph=concept_graph,
                doc_id=doc_id,
                need_community_summary=need_community_summary,
            )

        # 4b. 生成 Section Summary（CoE Step 2 精排用）
        # 异步后台执行，不阻塞文档入库响应（推理模型可能耗时较长）
        if settings.SUMMARY_ENABLED and parsed_doc.sections:
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
        
        processing_time = {
            "store": f"{store_time:.2f}s",
            "concept_graph": f"{concept_time:.2f}s",
            "community_summary": f"{community_summary_time:.2f}s",
            "section_summary": "async(background)",
        }
        
        if async_mode:
            # 存入缓存：包含 parsed_doc, concept_graph, extract_dir（图片处理需要）
            self._doc_cache.put(doc_id, {
                "parsed_doc": parsed_doc,
                "concept_graph": concept_graph,
                "extract_dir": parsed_doc.extract_dir
            })
            
            # 启动后台任务生成向量
            asyncio.create_task(self._generate_embeddings_async(doc_id))
            
            # 启动后台任务处理图片
            asyncio.create_task(self._process_images_async(doc_id))
            
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

    def _generate_section_summaries(self, parsed_doc) -> float:
        """为文档所有 Section 生成摘要并写入 NebulaGraph
        
        注意：如果摘要模型响应较慢（如推理模型），此步骤可能耗时较长。
        失败时不影响文档正常入库，只是 CoE Step2 精排时无摘要可用（降级为频率排序）。
        """
        step_start = time.time()
        try:
            from src.services.summary_service import SummaryService
            svc = SummaryService()
            summaries = svc.generate_summaries_for_sections(
                sections=parsed_doc.sections,
                chunks=parsed_doc.chunks,
            )
            non_empty = {sid: s for sid, s in summaries.items() if s}
            if non_empty:
                self.nebula_client.update_section_summaries(non_empty)
                logger.info(f"✅ Section 摘要生成完成: {len(non_empty)} 个 (耗时: {time.time()-step_start:.2f}秒)")
            else:
                logger.info("Section 摘要生成结果为空（模型超时或返回空）")
            return time.time() - step_start
        except Exception as e:
            logger.warning(f"Section 摘要生成失败（不影响文档处理）: {e}")
            return 0.0
    
    async def _generate_embeddings_async(self, doc_id: str):
        """异步生成向量嵌入"""
        await asyncio.sleep(0.1)  # 让出控制权
        
        cached = self._doc_cache.get(doc_id)
        if cached is None:
            logger.error(f"文档数据未缓存: {doc_id}")
            return
        
        # 新缓存结构是 dict
        parsed_doc = cached.get("parsed_doc")
        concept_graph = cached.get("concept_graph")
        
        if parsed_doc is None or concept_graph is None:
            logger.error(f"文档缓存数据不完整: {doc_id}")
            return
        
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
        # 注意：不再在这里释放缓存，因为图片处理任务可能还在使用
        # 缓存将依赖 TTL 自动过期
    
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

    async def generate_embeddings_async(self, doc_id: str):
        """异步生成向量嵌入（公共方法）"""
        await self._generate_embeddings_async(doc_id)

    async def process_images_async(self, doc_id: str):
        """异步处理图片（公共方法）"""
        await self._process_images_async(doc_id)

    async def _process_images_async(self, doc_id: str):
        """异步处理图片"""
        await asyncio.sleep(0.1)  # 让出控制力
        
        cached = self._doc_cache.get(doc_id)
        if cached is None:
            logger.error(f"文档数据未缓存: {doc_id}")
            return
        
        # 新缓存结构是 dict
        parsed_doc = cached.get("parsed_doc")
        extract_dir = cached.get("extract_dir")
        
        if parsed_doc is None or extract_dir is None:
            logger.error(f"文档缓存数据不完整: {doc_id}")
            return
        
        try:
            # 构建 page_text_map
            page_text_map = {}
            for chunk in parsed_doc.chunks:
                # 从 chunk_id 提取 page_idx（如果有的话）
                # 这里简化处理，直接从 raw_content_list 获取
                pass
            
            # 如果有 raw_content_list，使用它构建 page_text_map
            raw_content_list = getattr(parsed_doc, 'raw_content_list', [])
            if raw_content_list:
                # 构建 page -> text 映射
                for item in raw_content_list:
                    if item.get("type") == "text":
                        page = item.get("page_idx", 0)
                        text = item.get("text", "")
                        if page not in page_text_map:
                            page_text_map[page] = ""
                        page_text_map[page] += text
            
            # 处理图片
            if raw_content_list:
                result = self.image_service.process_images(
                    content_list=raw_content_list,
                    doc_id=doc_id,
                    extract_dir=extract_dir,
                    page_text_map=page_text_map
                )
                
                image_chunks = result.get("image_chunks", [])
                table_chunks = result.get("table_chunks", [])
                
                if image_chunks or table_chunks:
                    # 生成图片/表格 chunk 的向量
                    all_chunks = image_chunks + table_chunks
                    embeddings = []
                    for chunk in all_chunks:
                        embedding = self.text_embedder.embed_text(chunk.text)
                        embeddings.append({
                            "doc_id": doc_id,
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "embedding": embedding,
                            "metadata": {
                                "type": "image" if chunk.chunk_id.startswith("img_") else "table",
                                "section_id": chunk.section_id
                            }
                        })
                    
                    if embeddings:
                        self._store_embeddings(embeddings)
                        logger.info(f"✅ 图片/表格向量存储完成: {len(embeddings)} 条")
                    
                    # 更新文档状态
                    self._update_document_status(
                        doc_id,
                        image_chunks_count=len(image_chunks),
                        table_chunks_count=len(table_chunks),
                        image_processing_status="completed"
                    )
                else:
                    logger.info(f"没有需要处理的图片/表格")
                    self._update_document_status(
                        doc_id,
                        image_chunks_count=0,
                        table_chunks_count=0,
                        image_processing_status="completed"
                    )
                
                logger.info(f"🎉 图片处理完成: {doc_id}")
            else:
                logger.warning(f"没有 raw_content_list 数据: {doc_id}")
                
        except Exception as e:
            logger.error(f"图片处理失败: {e}")
            self._update_document_status(
                doc_id,
                image_processing_status="failed",
                image_processing_error=str(e)
            )

    def get_document_status(self, doc_id: str) -> Dict[str, Any]:
        """获取文档状态"""
        return self.nebula_client.get_document_status(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档"""
        try:
            documents = self.nebula_client.get_documents()
            return documents
        except Exception as e:
            logger.error(f"列出文档失败: {e}")
            raise

    def _extract_and_store_entities(self, parsed_doc) -> None:
        """后台任务：用 Qwen3.5 从 chunk 抽取实体关系并写入 NebulaGraph（Enhanced-KG）"""
        from src.graph.entity_extractor import EntityExtractor
        from src.utils.llm_client import LLMClient

        doc_id = parsed_doc.metadata.doc_id
        start = time.time()
        try:
            summary_llm = LLMClient(
                api_url=settings.SUMMARY_API_URL,
                api_key=settings.SUMMARY_API_KEY,
                model=settings.SUMMARY_MODEL,
                timeout=getattr(settings, "SUMMARY_TIMEOUT", 300),
            )
            extractor = EntityExtractor(llm_client=summary_llm, max_workers=3)
            result = extractor.extract_from_chunks(
                chunks=parsed_doc.chunks,
                doc_id=doc_id,
                max_chunks=50,
            )
            if result["entities"]:
                self.nebula_client.insert_entity_graph(
                    doc_id=doc_id,
                    entities=result["entities"],
                    relations=result["relations"],
                )
                logger.info(
                    f"✅ Entity 抽取完成: {len(result['entities'])} 实体, "
                    f"{len(result['relations'])} 关系 (耗时: {time.time()-start:.1f}s)"
                )
            else:
                logger.info(f"Entity 抽取结果为空 (doc={doc_id})")
        except Exception as e:
            logger.error(f"Entity 抽取失败 (doc={doc_id}): {e}")
