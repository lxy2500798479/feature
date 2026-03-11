"""
文档处理服务 - 严格按 FinalRAG 设计文档实现

架构说明（对应设计文档）：
  Meta-KG 三层：Document → Section → Chunk  （上传时同步构建，无 LLM 成本）
  摘要生成：Section Summary / Document Summary  （后台异步，LLM 调用）
  Enhanced-KG：Entity / Relation  （查询时 LazyEnhancer 按需构建，不在上传时预构建）

去掉的内容：
  ❌ SpaCy Concept Graph  —— 设计文档无此模块，且是上传时性能杀手
  ❌ Community Summary    —— 依赖 Concept Graph，随之去掉

上传流程（同步快速路径，秒级返回）：
  1. 解析文档（txt_parser 流式处理）
  2. 写入 NebulaGraph（Doc/Section/Chunk 三层节点 + 边）
  3. 标记 graph_ready=True，立即返回 200

后台异步（不阻塞接口）：
  4. Section Summary 生成（LLM，并发）
  5. Document Summary 生成（LLM）
  6. 向量嵌入生成（Embedding）
  7. 图片实体提取（可选）
"""
import time
import asyncio
import threading
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
    """文档处理服务"""

    def __init__(self):
        self.nebula_client = None
        self.vector_client = None
        self.text_embedder = None
        self._image_service = None
        self._doc_cache = None
        self._init_core_components()
        logger.info("DocumentService 初始化完成")

    def _init_core_components(self):
        self.nebula_client = NebulaClient()
        self.nebula_client.connect()
        self.vector_client = MilvusClient()
        self.vector_client.connect()
        self.text_embedder = TextEmbedder()
        logger.info("核心组件初始化完成")

    # ── 懒加载属性 ──────────────────────────────────────────────────────────

    @property
    def image_service(self):
        if self._image_service is None:
            if not getattr(settings, 'ENABLE_IMAGE_ENTITY_EXTRACTION', False):
                return None
            from src.services.image_service import ImageService
            self._image_service = ImageService()
            logger.info("图片实体提取模块已加载")
        return self._image_service

    @property
    def doc_cache(self):
        if self._doc_cache is None:
            from src.services.document_cache import DocumentCache
            self._doc_cache = DocumentCache()
        return self._doc_cache

    @property
    def async_embedding_enabled(self):
        return getattr(settings, 'ASYNC_EMBEDDING_ENABLED', True)

    @property
    def summary_enabled(self):
        return getattr(settings, 'ENABLE_SUMMARY', True)

    # ── 主流程 ───────────────────────────────────────────────────────────────

    def process_document(self, file_path: str, async_mode: bool = True) -> Dict[str, Any]:
        """
        处理文档主流程

        同步（决定接口响应时间，必须快）：
          步骤1：解析文档 → ParsedDocument
          步骤2：写入 Meta-KG（NebulaGraph Doc/Section/Chunk）

        后台异步（不阻塞接口返回）：
          步骤3：Section Summary 生成（LLM 调用，耗时分钟级）
          步骤4：Document Summary 生成（LLM 调用）
          步骤5：向量嵌入生成（Embedding）
          步骤6：图片实体提取（可选）
        """
        t0 = time.time()

        # ── 步骤1：解析文档 ──────────────────────────────────────────────────
        parsed_doc = self._parse_document(file_path)
        doc_id = parsed_doc.metadata.doc_id
        parse_time = time.time() - t0
        logger.info(
            f"✅ 文档解析完成: {doc_id} | "
            f"{len(parsed_doc.sections)} sections, {len(parsed_doc.chunks)} chunks | "
            f"耗时 {parse_time:.2f}s"
        )

        # ── 步骤2：写入 Meta-KG ──────────────────────────────────────────────
        store_time = self._store_to_graph(parsed_doc)
        logger.info(f"✅ Meta-KG 写入完成: {doc_id} | 耗时 {store_time:.2f}s")

        # 立即标记 graph_ready，让查询端知道结构化图谱已可用
        self._update_document_status(doc_id, graph_ready=True, embeddings_ready=False)

        # ── 步骤3：Section Summary（后台，LLM 调用）─────────────────────────
        if self.summary_enabled and parsed_doc.sections:
            def _bg_section_summary():
                try:
                    self._generate_section_summaries(parsed_doc)
                except Exception as e:
                    logger.error(f"[后台] Section 摘要生成失败 {doc_id}: {e}")

            threading.Thread(
                target=_bg_section_summary,
                daemon=True,
                name=f"sec-summary-{doc_id[:8]}"
            ).start()
            logger.info(f"Section 摘要后台启动 ({len(parsed_doc.sections)} sections)")

        # ── 步骤4：Document Summary（后台，LLM 调用）────────────────────────
        if self.summary_enabled:
            def _bg_doc_summary():
                try:
                    self._generate_document_summary(parsed_doc)
                except Exception as e:
                    logger.error(f"[后台] Document 摘要生成失败 {doc_id}: {e}")

            threading.Thread(
                target=_bg_doc_summary,
                daemon=True,
                name=f"doc-summary-{doc_id[:8]}"
            ).start()
            logger.info("Document 摘要后台启动")

        # ── 步骤5：向量嵌入（后台）──────────────────────────────────────────
        embedding_time = 0.0
        if async_mode and self.async_embedding_enabled:
            embedding_time = self._generate_embeddings_async(parsed_doc)
            logger.info(f"向量嵌入后台已排队: {doc_id}")
        else:
            embedding_time = self._generate_embeddings_sync(parsed_doc)
            logger.info(f"✅ 向量嵌入完成: {doc_id} | 耗时 {embedding_time:.2f}s")

        # ── 步骤6：图片实体提取（后台，可选）────────────────────────────────
        if self.image_service is not None and getattr(settings, 'ENABLE_IMAGE_ENTITY_EXTRACTION', False):
            def _bg_image_entity():
                try:
                    self._process_image_entities(parsed_doc)
                except Exception as e:
                    logger.error(f"[后台] 图片实体提取失败 {doc_id}: {e}")

            threading.Thread(
                target=_bg_image_entity,
                daemon=True,
                name=f"img-entity-{doc_id[:8]}"
            ).start()
            logger.info("图片实体提取后台启动")

        total_sync_time = time.time() - t0
        logger.info(f"✅ 上传同步流程完成: {doc_id} | 总耗时 {total_sync_time:.2f}s（接口即将返回）")

        return {
            "doc_id": doc_id,
            "status": "processing",
            "timing": {
                "parse_s": round(parse_time, 2),
                "store_s": round(store_time, 2),
                "total_sync_s": round(total_sync_time, 2),
                "section_summary": "async",
                "doc_summary": "async",
                "embedding": "async" if (async_mode and self.async_embedding_enabled) else round(embedding_time, 2),
            }
        }

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _parse_document(self, file_path: str) -> ParsedDocument:
        parser = ParserRegistry.get_suitable_parser(file_path)
        if parser is None:
            raise ValueError(f"找不到支持 {file_path} 的解析器")
        return parser.parse(file_path)

    def _store_to_graph(self, parsed_doc: ParsedDocument) -> float:
        """写入 Meta-KG 三层结构（Doc / Section / Chunk）到 NebulaGraph"""
        start = time.time()
        doc_metadata = (
            parsed_doc.metadata.model_dump()
            if hasattr(parsed_doc.metadata, 'model_dump')
            else parsed_doc.metadata
        )
        self.nebula_client.insert_document(doc_metadata)
        logger.info(
            f"写入 NebulaGraph: "
            f"{len(parsed_doc.sections)} sections, {len(parsed_doc.chunks)} chunks ..."
        )
        self.nebula_client.insert_sections(parsed_doc.sections)
        self.nebula_client.insert_chunks(parsed_doc.chunks)
        self.nebula_client.insert_edges(parsed_doc.edges)
        return time.time() - start

    def _generate_section_summaries(self, parsed_doc: ParsedDocument):
        """
        生成 Section Summary（设计文档：CoE Step2 精排用）

        生成完成后同时写入：
        ① NebulaGraph（用于关键词降级路径查询）
        ② SectionSummaryIndex Milvus collection（用于 CoE Step2 向量精排）
        """
        from src.services.summary_service import SummaryService
        summary_service = SummaryService()
        summaries = summary_service.generate_summaries_for_sections(
            sections=parsed_doc.sections,
            chunks=parsed_doc.chunks,
        )
        for section in parsed_doc.sections:
            if section.section_id in summaries:
                section.summary = summaries[section.section_id]

        # ① 写入 NebulaGraph（原有逻辑）
        self._store_section_summaries(parsed_doc.metadata.doc_id, summaries)

        # ② 新增：写入 SectionSummaryIndex，供 CoE Step2 向量精排使用
        self._index_section_summaries(parsed_doc, summaries)

        logger.info(f"[后台] ✅ Section 摘要完成: {len(summaries)} 个")
        return summaries

    def _index_section_summaries(self, parsed_doc: ParsedDocument, summaries: Dict[str, str]):
        """
        将 section summary embedding 写入 SectionSummaryIndex

        使用 title + summary 拼接文本生成 embedding：
        - title 重复一次，赋予更高权重（让向量更贴近章节主题）
        - summary 补充语义细节

        在 _generate_section_summaries 之后调用，完全异步无阻塞。
        任何异常都 catch 掉，不影响主摘要流程。
        """
        try:
            from src.query.coe_engine import SectionSummaryIndex
            dim = getattr(self.text_embedder, "dimension", 1024)
            idx = SectionSummaryIndex(dimension=dim)
            if not idx.ready:
                logger.warning("[后台] SectionSummaryIndex 未就绪，跳过 section 向量写入")
                return

            sections_data = []
            for section in parsed_doc.sections:
                if section.level != 1:
                    continue  # 只对一级 section 建向量索引
                summary_text = summaries.get(section.section_id, "") or ""

                # title 出现两次：给章节主题更高向量权重
                embed_input = f"{section.title} {section.title} {summary_text}".strip()
                if not embed_input:
                    continue

                try:
                    embedding = self.text_embedder.embed_single(embed_input)
                    sections_data.append({
                        "section_id": section.section_id,
                        "doc_id": section.doc_id,
                        "title": section.title,
                        "summary": summary_text,
                        "order": section.order,
                        "embedding": embedding,
                    })
                except Exception as emb_err:
                    logger.warning(f"Section embedding 失败 {section.section_id}: {emb_err}")

            if sections_data:
                ok = idx.upsert(sections_data)
                logger.info(
                    f"[后台] SectionSummaryIndex 写入: "
                    f"{len(sections_data)} sections, ok={ok}"
                )
        except Exception as e:
            logger.warning(f"[后台] SectionSummaryIndex 写入失败（不影响主流程）: {e}")

    def _store_section_summaries(self, doc_id: str, summaries: Dict[str, str]):
        try:
            cleaned = {
                sid: (s.encode('utf-8', errors='ignore').decode('utf-8') if s else "")
                for sid, s in summaries.items()
            }
            self.nebula_client.update_sections_summary(doc_id, cleaned)
        except Exception as e:
            logger.error(f"存储 Section 摘要失败: {e}")

    def _generate_document_summary(self, parsed_doc: ParsedDocument):
        """生成 Document Summary"""
        from src.services.summary_service import SummaryService
        summary_service = SummaryService()
        doc_summary = summary_service.generate_document_summary(
            sections=parsed_doc.sections,
            chunks=parsed_doc.chunks,
        )
        parsed_doc.metadata.summary = doc_summary
        self._store_document_summary(parsed_doc.metadata.doc_id, doc_summary)
        logger.info(f"[后台] ✅ Document 摘要完成: {doc_summary[:50]}...")
        return doc_summary

    def _store_document_summary(self, doc_id: str, summary: str):
        try:
            cleaned = summary.encode('utf-8', errors='ignore').decode('utf-8') if summary else ""
            self.nebula_client.update_document(doc_id, {"summary": cleaned})
        except Exception as e:
            logger.error(f"存储 Document 摘要失败: {e}")

    def _generate_embeddings_async(self, parsed_doc: ParsedDocument) -> float:
        """把 parsed_doc 放入缓存，由 FastAPI background_tasks 取出执行"""
        start = time.time()
        self.doc_cache.put(parsed_doc.metadata.doc_id, parsed_doc)
        return time.time() - start

    def generate_embeddings_async(self, doc_id: str) -> None:
        """由 FastAPI background_tasks 调用：从缓存取文档并生成向量"""
        doc = self.doc_cache.acquire(doc_id)
        if doc:
            try:
                self._generate_embeddings_sync(doc)
                self._update_document_status(
                    doc_id,
                    embeddings_ready=True,
                    embedding_task_status="completed"
                )
                logger.info(f"[后台] ✅ 向量嵌入完成: {doc_id}")
            except Exception as e:
                logger.error(f"[后台] 向量嵌入失败 {doc_id}: {e}")
                self._update_document_status(doc_id, embedding_task_status="failed")
        else:
            logger.warning(f"文档 {doc_id} 不在缓存中，跳过向量嵌入")

    def process_images_async(self, doc_id: str) -> None:
        """由 FastAPI background_tasks 调用：处理图片"""
        doc = self.doc_cache.acquire(doc_id)
        if doc:
            try:
                if self.image_service:
                    content_list = getattr(doc, 'content_list', [])
                    extract_dir = getattr(doc, 'extract_dir', '')
                    if content_list:
                        result = self.image_service.process_images(
                            content_list=content_list,
                            doc_id=doc_id,
                            extract_dir=extract_dir,
                            page_text_map={},
                        )
                        logger.info(
                            f"[后台] 图片处理完成: {doc_id}, "
                            f"图片chunks: {len(result.get('image_chunks', []))}"
                        )
            except Exception as e:
                logger.error(f"[后台] 图片处理失败 {doc_id}: {e}")
        else:
            logger.warning(f"文档 {doc_id} 不在缓存中，跳过图片处理")

    def _generate_embeddings_sync(self, parsed_doc: ParsedDocument) -> float:
        """
        同步生成向量嵌入并写入 Milvus

        修复1（问题1）：原来逐条 embed_single()，N 个 chunk 就是 N 次 HTTP 请求。
          现在统一调用 text_embedder.embed_chunks() 批量处理（500条/批）。

        修复2（问题5）：写入时注入 section_title 和 doc_title，
          answer_synthesizer 展示来源时无需二次查询。
        """
        start = time.time()

        if not parsed_doc.chunks:
            return 0.0

        logger.info(f"开始批量生成嵌入: {len(parsed_doc.chunks)} 个 chunks")

        # 构建 section_id → section_title 映射（用于注入来源信息）
        section_title_map = {
            s.section_id: s.title
            for s in parsed_doc.sections
        }
        doc_title = getattr(parsed_doc.metadata, "title", "") or ""

        # 批量生成 embedding（embed_chunks 内部已处理大文档分批）
        embed_results = self.text_embedder.embed_chunks(parsed_doc.chunks)

        # 补充 id / section_title / doc_title 字段
        embeddings = []
        for r in embed_results:
            r["id"] = r["chunk_id"]
            r["section_title"] = section_title_map.get(r.get("section_id", ""), "")
            r["doc_title"] = doc_title
            embeddings.append(r)

        if embeddings:
            self.vector_client.insert_embeddings(embeddings)
            logger.info(f"向量写入 Milvus 完成: {len(embeddings)} 条")

        return time.time() - start

    def _process_image_entities(self, parsed_doc: ParsedDocument):
        try:
            if self.image_service is None:
                return
            logger.info("图片实体提取: 使用 ImageService 处理图片")
        except Exception as e:
            logger.error(f"图片实体提取失败: {e}")

    def _update_document_status(self, doc_id: str, **kwargs):
        if not kwargs:
            return
        try:
            self.nebula_client.update_document(doc_id, kwargs)
            logger.info(f"文档状态更新: {doc_id} → {kwargs}")
        except Exception as e:
            logger.error(f"更新文档状态失败: {e}")

    # ── 对外查询接口 ─────────────────────────────────────────────────────────

    def process_document_sync(self, file_path: str) -> Dict[str, Any]:
        return self.process_document(file_path, async_mode=False)

    async def process_document_async(self, file_path: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_document, file_path, True)

    def get_document(self, doc_id: str) -> Optional[ParsedDocument]:
        return self.doc_cache.get(doc_id)

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        if not self.nebula_client:
            return []
        try:
            docs = self.nebula_client.get_documents()
            return [
                {
                    "doc_id": d.get("doc_id", ""),
                    "title": d.get("title", "") or "（无标题）",
                    "file_type": d.get("file_type", "") or "unknown",
                    "graph_ready": bool(d.get("graph_ready", False)),
                    "embeddings_ready": bool(d.get("embeddings_ready", False)),
                    "embedding_task_status": d.get("embedding_task_status", "") or "",
                }
                for d in docs[offset: offset + limit]
            ]
        except Exception as e:
            logger.warning(f"list_documents 失败: {e}")
            return []

    def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
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
        if hasattr(self.doc_cache, 'cache') and doc_id in self.doc_cache.cache:
            del self.doc_cache.cache[doc_id]
        logger.warning(f"delete_document: 底层存储不支持删除: {doc_id}")
        return True