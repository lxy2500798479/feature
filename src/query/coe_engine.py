"""
CoE (Chain of Exploration) 导航引擎

完整三步 CoE 导航:
  Step 1: 全局向量粗检索 → 候选 chunks（快速锁定相关区域）
  Step 2: Section Summary 精排 → 锁定 Top-N 最相关 Section（提升召回率）
  Step 3: 在精排 Section 内二次向量检索 → 高精度 chunks
"""
from typing import Optional, List, Dict, Any
import time

from src.utils.logger import logger


class CoEEngine:
    """CoE 引擎 - 三步层级导航检索"""

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
        """CoE 三步层级导航检索

        Returns:
            {
                "vector_chunks": [...],  # Step 1 + Step 3 的 chunks
                "graph_chunks": [...],   # 图谱邻居补充内容
                "community_context": [] # 社区摘要（全局型查询用）
                "retrieval_paths": [...] # 检索路径（可溯源）
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

        # ── Step 1: 全局向量粗检索，快速定位候选区域 ──────────────────────
        try:
            coarse_k = min(top_k * 3, 30)  # 粗检索取更多，再精排
            coarse_results = self.vector_client.search(
                query_vector=query_embedding,
                doc_id=doc_id,
                top_k=coarse_k,
            )
            logger.info(f"CoE Step1 粗检索: {len(coarse_results)} 条 (耗时: {time.time()-t0:.2f}s)")
        except Exception as e:
            logger.warning(f"CoE Step1 粗检索失败: {e}")
            coarse_results = []

        if not coarse_results:
            return results

        # ── Step 2: Section Summary 精排 → 锁定 Top Section ──────────────
        top_section_ids = self._drill_down_sections(
            query=query,
            coarse_chunks=coarse_results,
            doc_id=doc_id,
            top_n=5,
        )

        # ── Step 3: 在精排 Section 内做精确向量检索 ────────────────────────
        if top_section_ids:
            try:
                precise_results = self.vector_client.search(
                    query_vector=query_embedding,
                    section_ids=top_section_ids,
                    top_k=top_k,
                )
                results["vector_chunks"] = precise_results
                results["retrieval_paths"].append(
                    f"coe_3step(coarse={len(coarse_results)},sections={len(top_section_ids)},precise={len(precise_results)})"
                )
                logger.info(
                    f"CoE Step3 精确检索: {len(precise_results)} 条 "
                    f"(sections={len(top_section_ids)}, 耗时: {time.time()-t0:.2f}s)"
                )
            except Exception as e:
                logger.warning(f"CoE Step3 精确检索失败，降级用粗检索结果: {e}")
                results["vector_chunks"] = coarse_results[:top_k]
                results["retrieval_paths"].append(f"vector_search(top_k={top_k},fallback)")
        else:
            # Section 精排失败，直接用粗检索 top_k 结果
            results["vector_chunks"] = coarse_results[:top_k]
            results["retrieval_paths"].append(f"vector_search(top_k={top_k})")
            logger.info(f"CoE 使用粗检索结果（无 Section 精排）: {len(results['vector_chunks'])} 条")

        # ── 图谱遍历（关联型查询）────────────────────────────────────────
        if use_graph and self.nebula_client and results["vector_chunks"]:
            try:
                graph_chunks = self._graph_navigate(results["vector_chunks"], doc_id)
                results["graph_chunks"] = graph_chunks
                results["retrieval_paths"].append("graph_traversal")
                logger.info(f"CoE 图谱遍历: {len(graph_chunks)} 条结果")
            except Exception as e:
                logger.warning(f"CoE 图谱遍历失败: {e}")

        # ── 社区导航（全局型查询）────────────────────────────────────────
        if use_community and self.nebula_client:
            try:
                community_ctx = self._community_navigate(doc_id)
                results["community_context"] = community_ctx
                results["retrieval_paths"].append("community_navigation")
                logger.info(f"CoE 社区导航: {len(community_ctx)} 条社区摘要")
            except Exception as e:
                logger.warning(f"CoE 社区导航失败: {e}")

        return results

    def _drill_down_sections(
        self,
        query: str,
        coarse_chunks: List[dict],
        doc_id: Optional[str],
        top_n: int = 5,
    ) -> List[str]:
        """Step 2: 用 Section Summary 精排，返回 Top-N Section IDs

        策略：
        1. 从粗检索结果提取涉及的 section_id 集合
        2. 从 NebulaGraph 读取这些 Section 的 summary
        3. 用关键词匹配对 Section 打分（简单高效，无需额外 embedding）
        4. 若 NebulaGraph 无 summary，直接按命中频率排序
        """
        # 从粗检索结果收集候选 section_id
        section_hit_count: Dict[str, int] = {}
        for chunk in coarse_chunks:
            sid = chunk.get("section_id", "")
            if not sid and "_chunk_" in chunk.get("chunk_id", ""):
                sid = chunk["chunk_id"].rsplit("_chunk_", 1)[0]
            if sid:
                section_hit_count[sid] = section_hit_count.get(sid, 0) + 1

        if not section_hit_count:
            return []

        # 尝试从 NebulaGraph 获取 Section Summary 做精排
        if self.nebula_client and doc_id:
            try:
                sections = self.nebula_client.get_sections_with_summaries(doc_id)
                if sections:
                    # 用候选 section_id 过滤
                    candidate_sections = [
                        s for s in sections if s["section_id"] in section_hit_count
                    ]
                    if candidate_sections:
                        scored = self._score_sections_by_query(query, candidate_sections, section_hit_count)
                        top_ids = [s["section_id"] for s in scored[:top_n]]
                        logger.info(
                            f"CoE Step2 Section 精排: {len(candidate_sections)} 个候选 → top {len(top_ids)} 个"
                        )
                        return top_ids
            except Exception as e:
                logger.warning(f"CoE Step2 Section 精排失败: {e}")

        # 降级：按命中频率排序
        sorted_sections = sorted(section_hit_count.items(), key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in sorted_sections[:top_n]]

    def _score_sections_by_query(
        self,
        query: str,
        sections: List[dict],
        hit_counts: Dict[str, int],
    ) -> List[dict]:
        """用关键词匹配对 Section 打分（summary 相关性 + 命中频率）"""
        query_tokens = set(query.lower().replace("，", " ").replace("。", " ").split())
        # 过滤停用词
        stopwords = {"是", "的", "了", "吗", "啊", "呢", "什么", "如何", "哪些"}
        query_tokens -= stopwords
        query_tokens = {t for t in query_tokens if len(t) >= 2}

        scored = []
        for sec in sections:
            summary = (sec.get("summary", "") or "").lower()
            title = (sec.get("title", "") or "").lower()
            text = summary + " " + title

            # 关键词命中数
            kw_hits = sum(1 for t in query_tokens if t in text)
            # 命中频率（Step 1 命中越多分越高）
            freq_score = hit_counts.get(sec["section_id"], 0)
            # 综合分
            score = kw_hits * 2 + freq_score
            scored.append({**sec, "_score": score})

        return sorted(scored, key=lambda x: x["_score"], reverse=True)

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
