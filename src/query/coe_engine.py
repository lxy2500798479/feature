"""
CoE (Chain of Exploration) 导航引擎

按设计文档：Meta-KG 优先，自上而下检索
  Step 1: find_relevant_documents(query) → 候选文档列表（Meta-KG）
  Step 2: drill_down_to_sections(docs) → 用 Section Summary 精排（Meta-KG）
  Step 3: extract_chunks(sections) → 在精排 Section 内向量检索
"""
from typing import Optional, List, Dict, Any
import time

from src.utils.logger import logger


class CoEEngine:
    """CoE 引擎 - Meta-KG 优先的三步层级导航检索"""

    def __init__(
        self,
        vector_client=None,
        nebula_client=None,
        embedder=None
    ):
        self.vector_client = vector_client
        self.nebula_client = nebula_client
        self.embedder = embedder

    def search(
        self,
        query: str,
        query_embedding=None,
        top_k: int = 10,
        doc_id: Optional[str] = None,
        use_graph: bool = False,
        use_community: bool = False,
    ) -> Dict[str, Any]:
        """CoE 三步层级导航检索（设计：Meta-KG 优先）

        Returns:
            {
                "vector_chunks": [...],
                "graph_chunks": [...],
                "community_context": [],
                "retrieval_paths": [...]
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
            return results

        # ── Step 1: find_relevant_documents（Meta-KG 优先）────────────────
        doc_ids = self._find_relevant_documents(query=query, doc_id=doc_id)
        logger.info(f"CoE Step1 文档定位: {len(doc_ids)} 个文档 (耗时: {time.time()-t0:.2f}s)")

        # ── Step 2: drill_down_to_sections（Section Summary 精排）─────────
        top_section_ids = self._drill_down_sections(
            query=query,
            doc_ids=doc_ids,
            top_n=5,
        )
        logger.info(
            f"CoE Step2 Section 精排: {len(top_section_ids)} 个 Section "
            f"(耗时: {time.time()-t0:.2f}s)"
        )

        # ── Step 3: extract_chunks（在精排 Section 内向量检索）─────────────
        try:
            filter_doc_id = doc_ids[0] if len(doc_ids) == 1 else None
            filter_doc_ids = doc_ids if len(doc_ids) > 1 else None
            filter_section_ids = top_section_ids if top_section_ids else None

            search_kwargs = {
                "query_vector": query_embedding,
                "top_k": top_k,
            }
            if filter_section_ids:
                search_kwargs["section_ids"] = filter_section_ids
            elif filter_doc_id:
                search_kwargs["doc_id"] = filter_doc_id
            elif filter_doc_ids:
                search_kwargs["doc_ids"] = filter_doc_ids

            vector_results = self.vector_client.search(**search_kwargs)
            results["vector_chunks"] = vector_results
            path_desc = (
                f"coe_meta_kg(docs={len(doc_ids)},sections={len(top_section_ids) or 0},chunks={len(vector_results)})"
                if top_section_ids
                else f"coe_fallback(docs={len(doc_ids)},chunks={len(vector_results)})"
            )
            results["retrieval_paths"].append(path_desc)
            logger.info(
                f"CoE Step3 向量检索: {len(vector_results)} 条 "
                f"(耗时: {time.time()-t0:.2f}s)"
            )
        except Exception as e:
            logger.warning(f"CoE Step3 向量检索失败: {e}")

        # ── 图谱遍历（关联型：Section 内邻居扩展）──────────────────────────
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

    def _find_relevant_documents(
        self,
        query: str,
        doc_id: Optional[str],
        top_n: int = 10,
    ) -> List[str]:
        """Step 1: 从 Meta-KG 找相关文档（设计：先查 Meta-KG）

        - 若指定 doc_id：直接返回 [doc_id]
        - 否则：从 NebulaGraph 获取文档，用 query 与 title/summary 匹配打分，返回 top_n
        - 无 NebulaGraph 时：返回 []，后续 Step3 将做全库向量检索
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
        """用 query 关键词与 Document title/summary 匹配打分"""
        query_tokens = self._extract_query_tokens(query)
        scored = []
        for doc in docs:
            text = ((doc.get("title", "") or "") + " " + (doc.get("summary", "") or "")).lower()
            kw_hits = sum(1 for t in query_tokens if t in text)
            scored.append({**doc, "_score": kw_hits})
        return sorted(scored, key=lambda x: x["_score"], reverse=True)

    def _drill_down_sections(
        self,
        query: str,
        doc_ids: List[str],
        top_n: int = 5,
    ) -> List[str]:
        """Step 2: 用 Section Summary 精排（设计：Meta-KG 自上而下）

        从 NebulaGraph 获取文档的 Section 列表，用 query 与 title/summary 匹配打分，
        返回 top_n 个 section_id。不依赖向量检索结果。
        """
        if not self.nebula_client or not doc_ids:
            return []

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
            return []

        scored = self._score_sections_by_query(query, all_sections)
        return [s["section_id"] for s in scored[:top_n]]

    def _score_sections_by_query(self, query: str, sections: List[dict]) -> List[dict]:
        """用 query 关键词与 Section title/summary 匹配打分"""
        query_tokens = self._extract_query_tokens(query)
        scored = []
        for sec in sections:
            summary = (sec.get("summary", "") or "").lower()
            title = (sec.get("title", "") or "").lower()
            text = summary + " " + title
            kw_hits = sum(1 for t in query_tokens if t in text)
            scored.append({**sec, "_score": kw_hits})
        return sorted(scored, key=lambda x: x["_score"], reverse=True)

    def _extract_query_tokens(self, query: str) -> set:
        """提取 query 关键词（用于 Document/Section 打分）"""
        tokens = set(query.lower().replace("，", " ").replace("。", " ").split())
        stopwords = {"是", "的", "了", "吗", "啊", "呢", "什么", "如何", "哪些", "吗", "不"}
        tokens -= stopwords
        return {t for t in tokens if len(t) >= 2}

    def _graph_navigate(self, seed_chunks: List[dict], doc_id: Optional[str]) -> List[dict]:
        """基于 seed chunks 在图谱中做同 Section 邻居扩展

        策略：从 Milvus chunk_id 中解析出 section_id，
        向 Milvus 查询同一 Section 下的其他 chunks，作为图谱上下文补充。
        """
        if not self.vector_client:
            return []

        seen_section_ids = set()
        chunk_texts = []

        for chunk in seed_chunks[:5]:
            chunk_id = chunk.get("chunk_id", "") or chunk.get("id", "")
            if not chunk_id:
                continue

            if "_chunk_" in chunk_id:
                section_id = chunk_id.rsplit("_chunk_", 1)[0]
            else:
                section_id = chunk.get("section_id", "")

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
                        chunk_texts.append({"text": text, "source": "graph_neighbor", "section_id": section_id})
            except Exception as e:
                logger.warning(f"图谱邻居查询失败 section={section_id}: {e}")

        return chunk_texts

    def _community_navigate(self, doc_id: Optional[str]) -> List[dict]:
        """获取社区摘要作为全局上下文"""
        if not self.nebula_client:
            return []

        communities = []
        try:
            with self.nebula_client.get_session() as session:
                session.execute(f"USE {self.nebula_client.space_name};")
                if doc_id:
                    q = f'MATCH (n:CommunitySummary) WHERE n.doc_id == "{doc_id}" RETURN n.community_id as cid, n.summary as summary LIMIT 10;'
                else:
                    q = 'MATCH (n:CommunitySummary) RETURN n.community_id as cid, n.summary as summary LIMIT 10;'
                result = session.execute(q)
                if result.is_succeeded():
                    for row in result.rows():
                        cid_val = row.values[0]
                        summary_val = row.values[1]
                        cid_type = cid_val.getType()
                        sum_type = summary_val.getType()
                        if sum_type == 5:
                            s = summary_val.get_sVal()
                            summary = s.decode('utf-8') if isinstance(s, bytes) else s
                            if summary:
                                cid_str = str(cid_val.get_iVal()) if cid_type == 3 else str(cid_val)
                                communities.append({"community_id": cid_str, "summary": summary})
        except Exception as e:
            logger.warning(f"社区导航内部错误: {e}")

        return communities
