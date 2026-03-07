"""
社区摘要服务 — Fusion GraphRAG 核心机制

在 Leiden 社区检测完成后，为每个社区生成摘要：
1. 收集社区内的 Top-N 高频概念
2. 收集与社区概念关联的 chunk 文本片段
3. 调用 LLM 生成社区级摘要（支持并发）
4. 存入 NebulaGraph（Community 节点）

社区摘要在查询阶段用于：
- GLOBAL 查询的全局上下文理解
- 图谱遍历时的社区主题识别
"""
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import settings
from src.utils.logger import logger
from src.utils.llm_client import LLMClient


COMMUNITY_SUMMARY_SYSTEM = "你是一个文档分析专家。请根据提供的关键概念和相关文本片段，生成一段简洁的主题摘要（100-200字）。"

COMMUNITY_SUMMARY_PROMPT = """以下是一个概念社区的信息：

社区编号：{community_id}
核心概念（按频率排序）：{concepts}

相关文本片段：
{text_snippets}

请用100-200字总结这个概念社区的主题和核心内容。"""


class CommunitySummaryService:
    """社区摘要生成服务（支持并发）"""

    # 默认并发数
    DEFAULT_CONCURRENCY = 5

    def __init__(self, llm_client: Optional[LLMClient] = None, concurrency: Optional[int] = None):
        self.llm = llm_client or LLMClient()
        self.concurrency = concurrency or getattr(
            settings, 'SUMMARY_CONCURRENCY', self.DEFAULT_CONCURRENCY
        )

    def generate_summaries(
        self,
        concept_graph_result: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        doc_id: str,
    ) -> Dict[int, str]:
        """
        为概念图谱中的每个社区生成摘要（支持并发）

        Args:
            concept_graph_result: concept_graph_builder 的输出，包含 nodes, edges, communities
            chunks: 文档的文本块列表
            doc_id: 文档 ID

        Returns:
            Dict[int, str]: {community_id: summary_text}
        """
        start_time = time.time()

        nodes = concept_graph_result.get("nodes", [])
        communities_map = concept_graph_result.get("communities", {})

        if not communities_map:
            logger.info("无社区数据，跳过社区摘要生成")
            return {}

        # 1. 按社区分组概念，按频率排序
        community_concepts = self._group_concepts_by_community(nodes)

        # 2. 为每个概念找到相关的 chunk 文本
        concept_to_chunks = self._map_concepts_to_chunks(nodes, chunks)

        # 3. 为每个社区生成摘要（支持并发）
        summaries = self._generate_summaries_concurrent(
            community_concepts=community_concepts,
            concept_to_chunks=concept_to_chunks,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"✅ 社区摘要生成完成 (并发: {self.concurrency}, "
            f"耗时: {elapsed:.2f}秒, 社区数: {len(summaries)}/{len(community_concepts)})"
        )

        return summaries

    def _generate_summaries_concurrent(
        self,
        community_concepts: Dict[int, List[Dict[str, Any]]],
        concept_to_chunks: Dict[str, List[str]],
    ) -> Dict[int, str]:
        """
        并发为多个社区生成摘要

        Args:
            community_concepts: {community_id: [concepts]}
            concept_to_chunks: {phrase: [snippets]}

        Returns:
            {community_id: summary_text}
        """
        summaries: Dict[int, str] = {}

        # 过滤有效的社区
        valid_communities = [
            (cid, concepts)
            for cid, concepts in community_concepts.items()
            if cid >= 0
        ]

        if not valid_communities:
            return summaries

        # 如果社区数少或并发数为 1，使用串行
        if len(valid_communities) <= 1 or self.concurrency <= 1:
            for community_id, concepts in valid_communities:
                summary = self._generate_single_summary(
                    community_id=community_id,
                    concepts=concepts,
                    concept_to_chunks=concept_to_chunks,
                )
                if summary:
                    summaries[community_id] = summary
            return summaries

        # 并发生成摘要
        with ThreadPoolExecutor(
            max_workers=self.concurrency,
            thread_name_prefix="community_summary"
        ) as executor:
            # 提交所有任务
            future_to_community = {
                executor.submit(
                    self._generate_single_summary,
                    community_id=community_id,
                    concepts=concepts,
                    concept_to_chunks=concept_to_chunks,
                ): community_id
                for community_id, concepts in valid_communities
            }

            # 收集结果
            for future in as_completed(future_to_community):
                community_id = future_to_community[future]
                try:
                    summary = future.result()
                    if summary:
                        summaries[community_id] = summary
                except Exception as e:
                    logger.warning(f"社区 {community_id} 摘要生成失败: {e}")

        return summaries

    def _group_concepts_by_community(
        self, nodes: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """按社区分组概念并按频率排序"""
        groups = defaultdict(list)
        for node in nodes:
            cid = node.get("community", -1)
            groups[cid].append(node)

        # 每个社区内按频率降序排序
        for cid in groups:
            groups[cid].sort(key=lambda n: n.get("freq", 0), reverse=True)

        return dict(groups)

    def _map_concepts_to_chunks(
        self,
        nodes: List[Dict[str, Any]],
        chunks: List[Any],
    ) -> Dict[str, List[str]]:
        """为每个概念找到包含它的 chunk 文本片段"""
        concept_phrases = {n.get("phrase", "") for n in nodes if n.get("phrase")}
        concept_to_texts = defaultdict(list)

        for chunk in chunks:
            # 兼容 ChunkNode 对象和 dict
            text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
            if not text:
                continue
            for phrase in concept_phrases:
                if phrase in text:
                    # 提取概念周围的上下文（前后各100字符）
                    idx = text.find(phrase)
                    start = max(0, idx - 100)
                    end = min(len(text), idx + len(phrase) + 100)
                    snippet = text[start:end]
                    concept_to_texts[phrase].append(snippet)

        return dict(concept_to_texts)

    def _generate_single_summary(
        self,
        community_id: int,
        concepts: List[Dict[str, Any]],
        concept_to_chunks: Dict[str, List[str]],
        max_concepts: int = 10,
        max_snippets: int = 5,
    ) -> Optional[str]:
        """为单个社区生成摘要"""
        # Top-N 概念
        top_concepts = concepts[:max_concepts]
        concept_strs = ", ".join(
            f"{c['phrase']}(频率:{c.get('freq', 0)})" for c in top_concepts
        )

        # 收集相关文本片段
        snippets = []
        for concept in top_concepts:
            phrase = concept.get("phrase", "")
            if phrase in concept_to_chunks:
                snippets.extend(concept_to_chunks[phrase][:2])
            if len(snippets) >= max_snippets:
                break
        snippets = snippets[:max_snippets]

        if not snippets:
            # 无文本片段，仅用概念列表生成
            text_section = "（无直接相关文本片段）"
        else:
            text_section = "\n".join(f"  - {s}" for s in snippets)

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            community_id=community_id,
            concepts=concept_strs,
            text_snippets=text_section,
        )

        try:
            result = self.llm.chat(
                prompt=prompt,
                system_prompt=COMMUNITY_SUMMARY_SYSTEM,
                max_tokens=4096,
                temperature=0.1,
                no_think=True,
            )
            if result:
                logger.debug(
                    f"社区 {community_id} 摘要生成完成 "
                    f"({len(result)} 字, {len(top_concepts)} 个概念)"
                )
            return result
        except Exception as e:
            logger.warning(f"社区 {community_id} 摘要生成失败: {e}")
            return None
