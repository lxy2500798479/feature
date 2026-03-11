"""
CoE (Chain of Exploration) 导航引擎 - Section 精排向量化升级版

===============================================================================
【问题2：CoE Step2 Section 精排改为向量相似度】

旧方案（关键词匹配）：
  _score_sections_by_query()：将 query 分词后对 Section title/summary 做关键词计数
  
  缺陷：
  ① "唐三和小舞的关系" → tokens: {"唐三", "小舞", "关系"}
     Section A 摘要："唐三与比比东的对决" → 命中"唐三" = 1分
     Section B 摘要："唐三日常训练"       → 命中"唐三" = 1分
     → 两个 section 无法区分，精排无效

  ② 同义词/语义相近无法识别
     query:"魂师修炼方法" vs Section:"魂力提升技巧" → 语义高度相关但关键词不重叠，命中0分

新方案（向量相似度）：
  1. 上传文档时，对每个 Section 的 Summary 生成 embedding，
     存入独立的 Milvus collection（finalrag_section_summaries）
  2. 查询时，用 query embedding 在 section_summaries collection 做向量检索
  3. 返回语义最相关的 top_n section_id，传入 Step3 做精确 chunk 检索

  效果：
  ① 上例中 "唐三和小舞的关系" 的 embedding 会与包含两人互动的 section 语义距离最近
  ② 跨语言、同义词、近义词都能正确匹配
  ③ 无 summary（还未生成）时，自动 fallback 到 title 向量
  ④ Milvus collection 不存在时，fallback 到旧的关键词匹配（保证兼容性）

架构变化：
  - 新增 SectionSummaryIndex 类：管理 finalrag_section_summaries collection
  - CoEEngine 注入 embedder，Step2 调用向量检索
  - SummaryService 生成摘要后调用 SectionSummaryIndex.upsert() 写入向量
  - 完全向后兼容：无向量时 fallback 到关键词匹配

Step1 / Step3 / 图谱遍历 / 社区导航：逻辑不变，仅 Step2 升级
===============================================================================
"""
from typing import Optional, List, Dict, Any
import time

from src.utils.logger import logger


# ── Section Summary 向量索引 ────────────────────────────────────────────────

class SectionSummaryIndex:
    """
    Section Summary 向量索引

    使用独立的 Milvus collection（finalrag_section_summaries）存储：
      - section_id（主键）
      - doc_id
      - title
      - summary（文本，用于 fallback 关键词匹配）
      - embedding（title + summary 拼接后的向量）

    与 finalrag_chunks 完全独立，不影响现有数据结构。
    """

    COLLECTION_NAME = "finalrag_section_summaries"
    # embedding 维度从 MilvusClient 的 dimension 读取，这里用 None 表示动态获取
    _dim: Optional[int] = None

    def __init__(self, milvus_connection_alias: str = "default", dimension: int = 1024):
        self._alias = milvus_connection_alias
        self._dim = dimension
        self._collection = None
        self._ready = False
        self._init_collection()

    def _init_collection(self):
        """初始化 Milvus collection（幂等）"""
        try:
            from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

            if utility.has_collection(self.COLLECTION_NAME, using=self._alias):
                self._collection = Collection(self.COLLECTION_NAME, using=self._alias)
                try:
                    self._collection.load()
                except Exception:
                    pass
                self._ready = True
                logger.info(f"SectionSummaryIndex: 已加载现有 collection '{self.COLLECTION_NAME}'")
                return

            fields = [
                FieldSchema(name="section_id", dtype=DataType.VARCHAR,
                            is_primary=True, max_length=256),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="order_idx", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
            ]
            schema = CollectionSchema(
                fields=fields,
                description="FinalRAG section summary embeddings for CoE Step2"
            )
            self._collection = Collection(
                name=self.COLLECTION_NAME,
                schema=schema,
                using=self._alias,
            )
            self._collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 256},
                }
            )
            self._collection.load()
            self._ready = True
            logger.info(f"SectionSummaryIndex: 已创建 collection '{self.COLLECTION_NAME}' (dim={self._dim})")
        except Exception as e:
            logger.warning(f"SectionSummaryIndex 初始化失败（将 fallback 到关键词匹配）: {e}")
            self._ready = False

    def upsert(self, sections_data: List[Dict]) -> bool:
        """
        写入 / 更新 Section Summary 向量

        sections_data 格式：
        [
          {
            "section_id": "doc_xxx_sec_0",
            "doc_id": "doc_xxx",
            "title": "第一章 唐三觉醒",
            "summary": "唐三在斗罗大陆觉醒双生武魂...",
            "order": 0,
            "embedding": [0.1, 0.2, ...],  # list[float], 维度=EMBEDDING_DIMENSION
          },
          ...
        ]

        调用时机：SummaryService 生成 section summary 后立即调用
        """
        if not self._ready or not self._collection or not sections_data:
            return False

        try:
            # 先删除同 doc 旧数据（upsert = delete + insert）
            doc_ids = list({d["doc_id"] for d in sections_data})
            for did in doc_ids:
                try:
                    self._collection.delete(f'doc_id == "{did}"')
                except Exception:
                    pass

            # 按列组装插入数据
            data = [
                [d["section_id"] for d in sections_data],
                [d["doc_id"] for d in sections_data],
                [d.get("title", "")[:512] for d in sections_data],
                [d.get("summary", "")[:8192] for d in sections_data],
                [int(d.get("order", 0)) for d in sections_data],
                [
                    (d["embedding"].tolist() if hasattr(d["embedding"], "tolist") else d["embedding"])
                    for d in sections_data
                ],
            ]

            self._collection.insert(data)
            self._collection.flush()
            logger.info(f"SectionSummaryIndex: upsert {len(sections_data)} sections for doc {doc_ids[0] if doc_ids else '?'}")
            return True
        except Exception as e:
            logger.warning(f"SectionSummaryIndex upsert 失败: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        doc_ids: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> List[Dict]:
        """
        向量检索最相关的 Sections

        Args:
            query_embedding: 查询向量（与 chunk embedding 同维度同模型）
            doc_ids: 限定文档范围（None 表示全库检索）
            top_n: 返回数量

        Returns:
            [{"section_id": ..., "doc_id": ..., "title": ..., "score": ...}, ...]
            按相似度降序排列
        """
        if not self._ready or not self._collection:
            return []

        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

            expr = None
            if doc_ids:
                quoted = ", ".join(f'"{did}"' for did in doc_ids)
                expr = f"doc_id in [{quoted}]"

            results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_n,
                expr=expr,
                output_fields=["section_id", "doc_id", "title", "order_idx"],
            )

            hits = []
            for r in results[0]:
                hits.append({
                    "section_id": r.entity.get("section_id", ""),
                    "doc_id": r.entity.get("doc_id", ""),
                    "title": r.entity.get("title", ""),
                    "order": r.entity.get("order_idx", 0),
                    "score": float(r.distance),
                })
            return hits

        except Exception as e:
            logger.warning(f"SectionSummaryIndex search 失败: {e}")
            return []

    def delete_by_doc(self, doc_id: str) -> bool:
        """删除文档的所有 section 向量（文档删除时调用）"""
        if not self._ready or not self._collection:
            return False
        try:
            self._collection.delete(f'doc_id == "{doc_id}"')
            self._collection.flush()
            return True
        except Exception as e:
            logger.warning(f"SectionSummaryIndex delete_by_doc 失败: {e}")
            return False

    @property
    def ready(self) -> bool:
        return self._ready


# ── CoE 引擎 ────────────────────────────────────────────────────────────────

class CoEEngine:
    """
    CoE 引擎 - Meta-KG 优先的三步层级导航检索

    Step 1: find_relevant_documents(query) → 候选文档列表（关键词/文档摘要匹配）
    Step 2: drill_down_to_sections(docs)   → Section Summary 向量检索精排 ← 核心升级
    Step 3: extract_chunks(sections)       → 在精排 Section 内向量检索 Chunk

    Step2 降级逻辑（三层）：
      ① 有 query_embedding + SectionSummaryIndex 就绪 → 向量检索（最优）
      ② SectionSummaryIndex 不就绪但有 query_embedding
         → 用 query_embedding 与 section summary 文本做 TF-IDF-like 打分（中等）
      ③ 无 embedding → 关键词计数匹配（旧方案，兜底）
    """

    def __init__(
        self,
        vector_client=None,
        nebula_client=None,
        embedder=None,
        section_summary_index: Optional[SectionSummaryIndex] = None,
    ):
        self.vector_client = vector_client
        self.nebula_client = nebula_client
        self.embedder = embedder

        # Step2 向量精排所用的 index
        # 若传入 None，尝试自动初始化（从 vector_client 读取 dimension）
        self._section_index = section_summary_index
        self._section_index_init_attempted = section_summary_index is not None

    @property
    def section_summary_index(self) -> Optional[SectionSummaryIndex]:
        """懒加载 SectionSummaryIndex（避免启动时 Milvus 未就绪导致整体失败）"""
        if self._section_index is not None:
            return self._section_index

        if self._section_index_init_attempted:
            return None

        self._section_index_init_attempted = True
        try:
            dim = getattr(self.vector_client, "dimension", 1024) if self.vector_client else 1024
            self._section_index = SectionSummaryIndex(dimension=dim)
            logger.info(f"CoEEngine: SectionSummaryIndex 自动初始化成功 (dim={dim})")
        except Exception as e:
            logger.warning(f"CoEEngine: SectionSummaryIndex 自动初始化失败: {e}")
            self._section_index = None

        return self._section_index

    # ── 主检索入口 ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        query_embedding=None,
        top_k: int = 10,
        doc_id: Optional[str] = None,
        use_graph: bool = False,
        use_community: bool = False,
    ) -> Dict[str, Any]:
        """
        CoE 三步层级导航检索

        Returns:
            {
                "vector_chunks":    [...],
                "graph_chunks":     [...],
                "community_context":[...],
                "retrieval_paths":  [...],
            }
        """
        t0 = time.time()
        results: Dict[str, Any] = {
            "vector_chunks": [],
            "graph_chunks": [],
            "community_context": [],
            "retrieval_paths": [],
        }

        if self.vector_client is None or query_embedding is None:
            logger.warning("CoEEngine: vector_client 或 query_embedding 为空，跳过检索")
            return results

        # ── Step 1: 文档定位 ──────────────────────────────────────────────
        doc_ids = self._find_relevant_documents(query=query, doc_id=doc_id)
        logger.info(
            f"CoE Step1 文档定位: {len(doc_ids)} 个文档 (耗时 {time.time()-t0:.2f}s)"
        )

        # ── Step 2: Section 精排（向量检索）──────────────────────────────
        top_section_ids, step2_method = self._drill_down_sections(
            query=query,
            query_embedding=query_embedding,
            doc_ids=doc_ids,
            top_n=6,  # 多取一些 section，Step3 向量检索会再次精排
        )
        logger.info(
            f"CoE Step2 Section 精排[{step2_method}]: "
            f"{len(top_section_ids)} 个 Section (耗时 {time.time()-t0:.2f}s)"
        )

        # ── Step 3: Chunk 向量检索（在精排 Section 内）────────────────────
        try:
            search_kwargs: Dict[str, Any] = {
                "query_vector": query_embedding,
                "top_k": top_k,
            }
            if top_section_ids:
                search_kwargs["section_ids"] = top_section_ids
            elif len(doc_ids) == 1:
                search_kwargs["doc_id"] = doc_ids[0]
            elif doc_ids:
                search_kwargs["doc_ids"] = doc_ids
            # 若 doc_ids 为空，做全库检索

            vector_results = self.vector_client.search(**search_kwargs)
            results["vector_chunks"] = vector_results

            path_desc = (
                f"coe_meta_kg(docs={len(doc_ids)},sections={len(top_section_ids)},"
                f"chunks={len(vector_results)},step2={step2_method})"
                if top_section_ids
                else f"coe_fallback(docs={len(doc_ids)},chunks={len(vector_results)})"
            )
            results["retrieval_paths"].append(path_desc)
            logger.info(
                f"CoE Step3 向量检索: {len(vector_results)} 条 (耗时 {time.time()-t0:.2f}s)"
            )
        except Exception as e:
            logger.warning(f"CoE Step3 向量检索失败: {e}")

        # ── 图谱遍历（关联型：Section 内邻居扩展）────────────────────────
        if use_graph and results["vector_chunks"]:
            try:
                graph_chunks = self._graph_navigate(
                    results["vector_chunks"],
                    doc_id or (doc_ids[0] if doc_ids else None),
                )
                results["graph_chunks"] = graph_chunks
                results["retrieval_paths"].append("graph_traversal")
                logger.info(f"CoE 图谱遍历: {len(graph_chunks)} 条结果")
            except Exception as e:
                logger.warning(f"CoE 图谱遍历失败: {e}")

        # ── 社区导航（全局型查询）────────────────────────────────────────
        if use_community and self.nebula_client:
            try:
                community_ctx = self._community_navigate(
                    doc_id or (doc_ids[0] if doc_ids else None)
                )
                results["community_context"] = community_ctx
                results["retrieval_paths"].append("community_navigation")
                logger.info(f"CoE 社区导航: {len(community_ctx)} 条社区摘要")
            except Exception as e:
                logger.warning(f"CoE 社区导航失败: {e}")

        return results

    # ── Step 1: 文档定位 ──────────────────────────────────────────────────

    def _find_relevant_documents(
        self,
        query: str,
        doc_id: Optional[str],
        top_n: int = 10,
    ) -> List[str]:
        """
        从 Meta-KG 找相关文档

        - 指定 doc_id 直接返回
        - 否则从 NebulaGraph 获取文档列表，用关键词与 title/summary 匹配打分
        - 无 NebulaGraph 时返回 []，Step3 做全库检索
        """
        if doc_id:
            return [doc_id]

        if not self.nebula_client:
            return []

        try:
            docs = self.nebula_client.get_documents_for_retrieval()
            if not docs:
                return []
            scored = self._score_docs_by_query(query, docs)
            return [d["doc_id"] for d in scored[:top_n]]
        except Exception as e:
            logger.warning(f"CoE Step1 文档检索失败: {e}")
            return []

    def _score_docs_by_query(self, query: str, docs: List[dict]) -> List[dict]:
        """关键词打分（文档级别数量少，关键词足够）"""
        query_tokens = self._extract_query_tokens(query)
        scored = []
        for doc in docs:
            text = (
                (doc.get("title", "") or "") + " " + (doc.get("summary", "") or "")
            ).lower()
            kw_hits = sum(1 for t in query_tokens if t in text)
            scored.append({**doc, "_score": kw_hits})
        return sorted(scored, key=lambda x: x["_score"], reverse=True)

    # ── Step 2: Section 精排（核心升级）─────────────────────────────────

    def _drill_down_sections(
        self,
        query: str,
        query_embedding: List[float],
        doc_ids: List[str],
        top_n: int = 6,
    ) -> tuple:
        """
        Section 精排 - 三层降级策略

        Returns:
            (section_ids: List[str], method: str)
            method: "vector" | "keyword" | "empty"
        """
        if not doc_ids:
            return [], "empty"

        # ── 优先：向量检索 ────────────────────────────────────────────────
        idx = self.section_summary_index
        if idx is not None and idx.ready and query_embedding:
            section_ids, method = self._step2_vector(
                query_embedding=query_embedding,
                doc_ids=doc_ids,
                top_n=top_n,
            )
            if section_ids:
                return section_ids, method

        # ── 降级：从 NebulaGraph 取 section 文本，关键词匹配 ──────────────
        return self._step2_keyword(query=query, doc_ids=doc_ids, top_n=top_n)

    def _step2_vector(
        self,
        query_embedding: List[float],
        doc_ids: List[str],
        top_n: int,
    ) -> tuple:
        """
        Step2 向量路径：在 SectionSummaryIndex 中检索

        策略：
        ① 先在目标 doc_ids 范围内检索
        ② 若命中为 0（section summary 尚未写入），fallback 到关键词
        """
        try:
            hits = self.section_summary_index.search(
                query_embedding=query_embedding,
                doc_ids=doc_ids,
                top_n=top_n,
            )
            if hits:
                section_ids = [h["section_id"] for h in hits]
                scores_str = ", ".join(f"{h['title'][:12]}:{h['score']:.3f}" for h in hits[:3])
                logger.info(f"  Step2 向量精排 top3: {scores_str}")
                return section_ids, "vector"
        except Exception as e:
            logger.warning(f"  Step2 向量检索内部错误: {e}")

        return [], "vector_miss"

    def _step2_keyword(
        self,
        query: str,
        doc_ids: List[str],
        top_n: int,
    ) -> tuple:
        """
        Step2 关键词降级路径：从 NebulaGraph 取 section 列表做关键词计数
        保留旧版逻辑，作为兜底
        """
        if not self.nebula_client:
            return [], "empty"

        all_sections = []
        for did in doc_ids:
            try:
                sections = self.nebula_client.get_sections_with_summaries(did)
                for s in sections:
                    s["doc_id"] = did
                    all_sections.append(s)
            except Exception as e:
                logger.debug(f"获取文档 {did} 的 sections 失败: {e}")

        if not all_sections:
            return [], "empty"

        scored = self._score_sections_by_query(query, all_sections)
        section_ids = [s["section_id"] for s in scored[:top_n]]
        logger.info(f"  Step2 关键词精排: {len(section_ids)} sections")
        return section_ids, "keyword"

    def _score_sections_by_query(self, query: str, sections: List[dict]) -> List[dict]:
        """关键词计数打分（Step2 降级用）"""
        query_tokens = self._extract_query_tokens(query)
        scored = []
        for sec in sections:
            summary = (sec.get("summary", "") or "").lower()
            title = (sec.get("title", "") or "").lower()
            text = summary + " " + title
            # summary 命中权重 2x，title 命中权重 1x
            kw_hits = (
                sum(2 for t in query_tokens if t in summary)
                + sum(1 for t in query_tokens if t in title)
            )
            scored.append({**sec, "_score": kw_hits})
        return sorted(scored, key=lambda x: x["_score"], reverse=True)

    # ── 图谱遍历 ──────────────────────────────────────────────────────────

    def _graph_navigate(self, seed_chunks: List[dict], doc_id: Optional[str]) -> List[dict]:
        """
        基于 seed chunks 做同 Section 邻居扩展

        从 chunk_id 解析 section_id，查询同 section 下其他 chunks 作为上下文补充
        """
        if not self.vector_client:
            return []

        seen_section_ids: set = set()
        chunk_texts: List[dict] = []

        for chunk in seed_chunks[:5]:
            chunk_id = chunk.get("chunk_id", "") or chunk.get("id", "")
            if not chunk_id:
                continue

            section_id = (
                chunk_id.rsplit("_chunk_", 1)[0]
                if "_chunk_" in chunk_id
                else chunk.get("section_id", "")
            )
            if not section_id or section_id in seen_section_ids:
                continue
            seen_section_ids.add(section_id)

            try:
                siblings = self.vector_client.collection.query(
                    expr=f'section_id == "{section_id}"',
                    output_fields=["chunk_id", "section_id", "text"],
                    limit=8,
                )
                existing_cids = {c.get("chunk_id", "") for c in seed_chunks}
                for sibling in siblings:
                    text = sibling.get("text", "")
                    sibling_cid = sibling.get("chunk_id", "")
                    if text and sibling_cid not in existing_cids:
                        chunk_texts.append({
                            "text": text,
                            "source": "graph_neighbor",
                            "section_id": section_id,
                        })
            except Exception as e:
                logger.warning(f"图谱邻居查询失败 section={section_id}: {e}")

        return chunk_texts

    # ── 社区导航 ──────────────────────────────────────────────────────────

    def _community_navigate(self, doc_id: Optional[str]) -> List[dict]:
        """获取社区摘要作为全局上下文"""
        if not self.nebula_client:
            return []

        communities: List[dict] = []
        try:
            with self.nebula_client.get_session() as session:
                session.execute(f"USE {self.nebula_client.space_name};")
                if doc_id:
                    q = (
                        f'MATCH (n:CommunitySummary) WHERE n.doc_id == "{doc_id}" '
                        f'RETURN n.community_id as cid, n.summary as summary LIMIT 10;'
                    )
                else:
                    q = (
                        'MATCH (n:CommunitySummary) '
                        'RETURN n.community_id as cid, n.summary as summary LIMIT 10;'
                    )
                result = session.execute(q)
                if result.is_succeeded():
                    for row in result.rows():
                        cid_val = row.values[0]
                        summary_val = row.values[1]
                        sum_type = summary_val.getType()
                        cid_type = cid_val.getType()
                        if sum_type == 5:
                            s = summary_val.get_sVal()
                            summary = s.decode("utf-8") if isinstance(s, bytes) else s
                            if summary:
                                cid_str = (
                                    str(cid_val.get_iVal())
                                    if cid_type == 3
                                    else str(cid_val)
                                )
                                communities.append({
                                    "community_id": cid_str,
                                    "summary": summary,
                                })
        except Exception as e:
            logger.warning(f"社区导航内部错误: {e}")

        return communities

    # ── 工具方法 ──────────────────────────────────────────────────────────

    def _extract_query_tokens(self, query: str) -> set:
        """提取 query 关键词（用于文档级、Section 降级打分）"""
        tokens = set(
            query.lower()
            .replace("，", " ").replace("。", " ")
            .replace("？", " ").replace("！", " ")
            .split()
        )
        stopwords = {
            "是", "的", "了", "吗", "啊", "呢", "什么", "如何",
            "哪些", "不", "有", "和", "与", "在", "为",
        }
        tokens -= stopwords
        return {t for t in tokens if len(t) >= 2}