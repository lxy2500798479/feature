"""
延迟增强器 (Lazy Enhancer) - Phase 3 核心组件

设计目标：
  针对 relational（关联型）查询，在基础向量检索后按需触发：
  1. 用 LLM 从查询 + 已检索 chunks 中抽取核心实体和关键词
  2. 用抽取结果构造扩展查询，在 Milvus 中二次检索
  3. 在 NebulaGraph 中查找实体的 1-hop 概念邻居
  目的是补全第一次检索遗漏的关联内容，提升 relational 查询 F1。

按需（Lazy）体现在：
  - factual 查询不触发（成本节省）
  - global 查询不触发（用社区导航代替）
  - 只有 relational 且 enable_lazy_enhance=True 时才触发

查询相似度复用：
  - 若当前查询与近期查询语义相似（embedding 余弦 > 0.92），直接复用抽取结果，跳过 LLM
  - 用于连续追问、相似问题等场景，显著降低延迟
"""
import json
import re
from collections import deque
from typing import Optional, Dict, Any, List

import numpy as np

from src.utils.logger import logger

# 查询相似度缓存：key=doc_id, value=deque of (embedding, extraction)
_EXTRACTION_CACHE: Dict[str, deque] = {}
_CACHE_MAX_SIZE = 20
_SIMILARITY_THRESHOLD = 0.92


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# 实体抽取 Prompt
_EXTRACT_PROMPT = """从以下查询和参考文本中提取核心实体和关键词，用于扩展检索。

查询: {query}

参考文本片段:
{context}

请以 JSON 格式返回（只返回 JSON，不要其他内容）:
{{
  "entities": ["实体1", "实体2", ...],
  "keywords": ["关键词1", "关键词2", ...],
  "expanded_query": "扩展后的查询语句（结合实体和关键词）"
}}

要求：
- entities: 人名、组织名、产品名、地名等命名实体，最多 5 个
- keywords: 核心概念词，最多 8 个
- expanded_query: 在原查询基础上补充实体和关键词，不超过 100 字
"""


class LazyEnhancer:
    """延迟增强器 - 关联型查询的按需图谱推理"""

    def __init__(self, llm_client=None, vector_client=None, nebula_client=None, entity_types: List[str] = None):
        self.llm_client = llm_client
        self.vector_client = vector_client
        self.nebula_client = nebula_client
        self.entity_types = entity_types or ["PERSON", "ORG", "PRODUCT", "LOCATION", "EVENT", "CONCEPT"]

    def _get_cached_extraction(
        self, query_embedding, doc_id: Optional[str]
    ) -> Optional[Dict]:
        """若当前查询与近期查询语义相似，复用抽取结果"""
        if query_embedding is None:
            return None
        try:
            emb = np.array(query_embedding, dtype=np.float32)
            if emb.size < 10:
                return None
        except Exception:
            return None

        cache_key = doc_id or "_global"
        if cache_key not in _EXTRACTION_CACHE:
            return None

        best_sim, best_res = 0.0, None
        for cached_emb, cached_res in _EXTRACTION_CACHE[cache_key]:
            try:
                cemb = np.array(cached_emb, dtype=np.float32)
                if cemb.shape != emb.shape:
                    continue
                sim = _cosine_sim(emb, cemb)
                if sim > best_sim:
                    best_sim, best_res = sim, cached_res
            except Exception:
                continue

        if best_sim >= _SIMILARITY_THRESHOLD and best_res:
            logger.info(
                f"LazyEnhancer 查询相似复用: sim={best_sim:.2f}, "
                f"entities={best_res.get('entities', [])[:3]}"
            )
            return best_res
        return None

    def _put_cached_extraction(
        self, query_embedding, doc_id: Optional[str], extraction: Dict
    ):
        """将抽取结果加入缓存"""
        if query_embedding is None:
            return
        cache_key = doc_id or "_global"
        if cache_key not in _EXTRACTION_CACHE:
            _EXTRACTION_CACHE[cache_key] = deque(maxlen=_CACHE_MAX_SIZE)
        _EXTRACTION_CACHE[cache_key].append((list(query_embedding), extraction))

    def build(
        self,
        doc_id: str,
        chunks: List[Dict],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """构建/查询持久化实体图谱（懒加载）

        首次查询时从 chunks 抽取实体存入 NebulaGraph，
        后续查询直接从 NebulaGraph 读取。

        Args:
            doc_id: 文档 ID
            chunks: 当前检索到的 chunks
            force_rebuild: 是否强制重建

        Returns:
            {
               ],
                "relations": [...],
                "entities": [... "from_cache": bool,
                "new_entities": int,
                "new_relations": int,
            }
        """
        from src.graph.lazy_entity_builder import LazyEntityBuilder
        
        builder = LazyEntityBuilder(
            llm_client=self.llm_client,
            nebula_client=self.nebula_client,
            entity_types=self.entity_types,
        )
        return builder.build(doc_id=doc_id, chunks=chunks, force_rebuild=force_rebuild)

    def enhance(
        self,
        query: str,
        seed_chunks: Optional[List[dict]] = None,
        doc_id: Optional[str] = None,
        query_embedding=None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """关联型查询的延迟增强：抽取实体 → 扩展检索

        Args:
            query: 原始查询
            seed_chunks: 已有的向量检索结果，用作上下文
            doc_id: 文档 ID 过滤
            query_embedding: 原始查询向量（备用）
            top_k: 扩展检索返回数量

        Returns:
            {
                "entities": [...],
                "keywords": [...],
                "expanded_query": "...",
                "extra_chunks": [...],      # 扩展检索到的新 chunks
                "graph_entities": [...]     # 图谱中找到的实体邻居
            }
        """
        result = {
            "entities": [],
            "keywords": [],
            "expanded_query": query,
            "extra_chunks": [],
            "graph_entities": [],
        }

        # Step 1: 用 LLM 抽取实体和关键词（查询相似时复用缓存，跳过 LLM）
        extraction = self._get_cached_extraction(query_embedding, doc_id)
        if extraction is None:
            extraction = self._extract_entities(query, seed_chunks or [])
            if query_embedding is not None:
                self._put_cached_extraction(query_embedding, doc_id, extraction)

        result["entities"] = extraction.get("entities", [])
        result["keywords"] = extraction.get("keywords", [])
        expanded_query = extraction.get("expanded_query", query)
        result["expanded_query"] = expanded_query

        logger.info(
            f"LazyEnhancer 抽取: entities={result['entities'][:3]}, "
            f"keywords={result['keywords'][:3]}"
        )

        # Step 2: 用扩展查询在 Milvus 中二次关键词检索
        existing_cids = {c.get("chunk_id", "") for c in (seed_chunks or [])}
        if (result["entities"] or result["keywords"]) and self.vector_client:
            extra = self._keyword_search(
                entities=result["entities"],
                keywords=result["keywords"],
                doc_id=doc_id,
                existing_chunk_ids=existing_cids,
                top_k=top_k,
            )
            for ec in extra:
                existing_cids.add(ec.get("chunk_id", ""))
            result["extra_chunks"] = extra
            if extra:
                logger.info(f"LazyEnhancer 扩展检索: +{len(extra)} 条新 chunks")

        # Step 3: 图谱遍历（Entity→Relation 邻居）→ 用邻居实体 chunk_ids 拉取 chunks 增强检索
        if result["entities"] and self.nebula_client:
            graph_ents = self._graph_entity_lookup(result["entities"], doc_id)
            result["graph_entities"] = graph_ents
            logger.info(
                f"LazyEnhancer 图谱遍历: entities={len(result['entities'])}, "
                f"graph_relations={len(graph_ents)}"
            )

            # Step 4: 用子图遍历结果增强检索（按 plane 设计：用子图做图谱遍历 → 增强检索）
            graph_chunks = self._fetch_chunks_by_graph_entities(
                entities=result["entities"],
                graph_entities=graph_ents,
                doc_id=doc_id,
                existing_chunk_ids=existing_cids,
            )
            if graph_chunks:
                result["extra_chunks"].extend(graph_chunks)
                logger.info(f"LazyEnhancer 子图增强检索: +{len(graph_chunks)} 条 chunks")

        return result

    def _extract_entities(self, query: str, seed_chunks: List[dict]) -> Dict:
        """用 LLM 从查询和上下文中抽取实体与关键词"""
        if not self.llm_client:
            return self._fallback_extract(query)

        # 取前 3 个 chunk 作为上下文
        context_texts = []
        for c in seed_chunks[:3]:
            text = c.get("text", "")[:200]
            if text:
                context_texts.append(text)
        context = "\n".join(context_texts) or "（无参考文本）"

        prompt = _EXTRACT_PROMPT.format(query=query, context=context)
        try:
            raw = self.llm_client.chat(
                prompt=prompt,
                system_prompt="你是一个信息抽取助手，只输出 JSON。",
                max_tokens=4096,
                temperature=0.1,
                no_think=True,
            )
            # 尝试解析 JSON
            raw = raw.strip()
            # 去掉可能的 markdown 代码块
            raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"LazyEnhancer LLM 抽取失败，使用规则回退: {e}")
            return self._fallback_extract(query)

    def _fallback_extract(self, query: str) -> Dict:
        """LLM 不可用时的规则回退：简单分词提取关键词"""
        # 去掉疑问词、助词等停用词，取剩余词
        stopwords = {"是", "的", "了", "吗", "啊", "呢", "吧", "在", "有", "和", "与",
                     "什么", "怎么", "如何", "哪些", "谁", "为什么", "哪个", "哪里"}
        tokens = [t for t in re.split(r"[\s，。？！,.?!]+", query) if t and t not in stopwords]
        keywords = [t for t in tokens if len(t) >= 2][:6]
        return {
            "entities": [],
            "keywords": keywords,
            "expanded_query": query,
        }

    def _keyword_search(
        self,
        entities: List[str],
        keywords: List[str],
        doc_id: Optional[str],
        existing_chunk_ids: set,
        top_k: int,
    ) -> List[dict]:
        """在 Milvus 中用关键词做文本过滤扩展检索"""
        if not self.vector_client:
            return []

        extra_chunks = []
        # 合并实体和关键词，每个词独立查一次（限制只用最重要的前3个）
        search_terms = (entities + keywords)[:4]

        for term in search_terms:
            if not term:
                continue
            try:
                # 用 Milvus 的 VARCHAR 过滤：text like "%term%"
                expr = f'text like "%{term}%"'
                if doc_id:
                    expr = f'doc_id == "{doc_id}" && {expr}'

                hits = self.vector_client.collection.query(
                    expr=expr,
                    output_fields=["chunk_id", "section_id", "doc_id", "text"],
                    limit=top_k,
                )
                for h in hits:
                    cid = h.get("chunk_id", "")
                    if cid and cid not in existing_chunk_ids:
                        existing_chunk_ids.add(cid)
                        extra_chunks.append({
                            "chunk_id": cid,
                            "section_id": h.get("section_id", ""),
                            "doc_id": h.get("doc_id", ""),
                            "text": h.get("text", ""),
                            "score": 0.0,
                            "source": f"lazy_enhance:{term}",
                        })
            except Exception as e:
                logger.warning(f"LazyEnhancer 关键词检索 '{term}' 失败: {e}")

        return extra_chunks[:top_k * 2]

    def _graph_entity_lookup(self, entities: List[str], doc_id: Optional[str]) -> List[dict]:
        """在 NebulaGraph 中查找实体的 Entity→Relation 邻居（优先走 Enhanced-KG）"""
        if not self.nebula_client:
            return []

        result_entities = []
        try:
            for entity in entities[:4]:
                neighbors = self.nebula_client.get_entity_neighbors(
                    entity_name=entity,
                    doc_id=doc_id or "",
                    hops=2,
                )
                for nb in neighbors:
                    result_entities.append({
                        "entity": entity,
                        "related": nb.get("dst_name", ""),
                        "relation_type": nb.get("rel_type", ""),
                        "description": nb.get("rel_desc", ""),
                        "source": "entity_relation",
                    })
        except Exception as e:
            logger.warning(f"LazyEnhancer 图谱实体查询失败: {e}")

        return result_entities

    def _fetch_chunks_by_graph_entities(
        self,
        entities: List[str],
        graph_entities: List[dict],
        doc_id: Optional[str],
        existing_chunk_ids: set,
    ) -> List[dict]:
        """用子图遍历得到的实体名，从 Nebula 查 chunk_ids，再从 Milvus 拉取 chunks（plane 设计）"""
        if not self.nebula_client or not self.vector_client:
            return []

        # 收集所有相关实体名：查询实体 + 邻居实体
        entity_names = list(entities[:4])
        for ge in graph_entities:
            related = ge.get("related", "").strip()
            if related and related not in entity_names:
                entity_names.append(related)

        chunk_ids = self.nebula_client.get_entity_chunk_ids(
            entity_names=entity_names,
            doc_id=doc_id or "",
        )
        logger.info(
            f"LazyEnhancer 子图增强: entity_names={entity_names[:6]}, "
            f"chunk_ids_from_kg={len(chunk_ids) if chunk_ids else 0}, existing={len(existing_chunk_ids)}"
        )
        if not chunk_ids:
            return []

        # 过滤已存在的
        needed = [cid for cid in chunk_ids if cid and cid not in existing_chunk_ids]
        if not needed:
            logger.info("LazyEnhancer 子图增强: 所有 chunk_ids 已在 existing 中，无新增")
            return []

        chunks = self.vector_client.query_by_chunk_ids(needed)
        logger.info(f"LazyEnhancer 子图增强: needed={len(needed)}, milvus_hits={len(chunks)}")
        for c in chunks:
            existing_chunk_ids.add(c.get("chunk_id", ""))
        return chunks

    def extract_keywords(self, query: str) -> list:
        """提取关键词（公共接口）"""
        result = self._fallback_extract(query)
        return result["keywords"]

    def extract_entities(self, query: str) -> list:
        """提取实体（公共接口，需 LLM）"""
        result = self._extract_entities(query, [])
        return result.get("entities", [])
