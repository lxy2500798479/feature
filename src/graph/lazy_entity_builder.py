"""
懒加载实体构建器 (Lazy Entity Builder)

按 plane 设计：渐进式演进、越用越快

设计目标：
  - 每次关联型查询：基于本次召回的 chunks 抽取实体
  - 增量扩展：仅处理尚未抽取过的 chunks，合并新实体/关系到 NebulaGraph
  - 子图缓存：相同/相似 chunks 的抽取结果 Hash 缓存，避免重复 LLM 调用
  - 越用越快：图谱随查询不断丰富，高频子图被固化

架构：
  Query Time
      ↓
  获取已存储实体 → 计算已处理的 chunk_ids
      ↓
  过滤出本次新增的 chunks
      ↓
  ┌─ 无新增 ──→ 直接返回已存储图谱
  │
  └─ 有新增 ──→ 子图缓存命中? ─┬─ 是 ──→ 用缓存抽取结果
               │
               └─ 否 ──→ LLM 抽取 ──→ 写入缓存
                              ↓
                    合并到 NebulaGraph（INSERT 新实体 / UPDATE 已有实体 chunk_ids）
"""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any, Tuple

from src.utils.logger import logger

# ── Prompt ────────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """从下面文本中提取实体和关系，输出纯 JSON，不要其他内容。

要求：
1. 实体：人名、组织、产品、地点、事件等，每段文本尽量抽 3～8 个实体。
2. 关系：relations 里的 src、dst 必须与上面 entities 里的 name 完全一致（复制粘贴），这样关系才能连上。每段文本尽量抽 2～6 条关系。
3. 类型用英文：{entity_types}。

文本：{text}

输出格式（严格遵循）：{{"entities":[{"name":"实体名","type":"类型","description":"简短描述"}],"relations":[{"src":"实体A名","dst":"实体B名","type":"关系类型","strength":0.9}]}}"""


def _try_extract_json(text: str) -> dict:
    """从任意文本中提取 JSON（健壮提取）"""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text)
    text = re.sub(r"^\d+\.\s+\*\*.*?\*\*:?", "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        parsed = json.loads(text)
        if "entities" in parsed:
            return parsed
    except:
        pass

    entities = []
    relations = []
    for line in text.split('\n'):
        line = line.strip().lstrip('*-').strip()
        line = re.sub(r"^(Entities|Relations|Entity|Relation)\s*:?\s*", "", line, flags=re.IGNORECASE)
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "name" in obj:
                entities.append(obj)
            elif "src" in obj and "dst" in obj:
                relations.append(obj)
        except:
            pass

    if entities or relations:
        return {"entities": entities, "relations": relations}

    return {"entities": [], "relations": []}


class LazyEntityBuilder:
    """懒加载实体构建器 - 查询时按需构建 + 持久化"""

    def __init__(
        self,
        llm_client=None,
        nebula_client=None,
        entity_types: List[str] = None,
        max_chunks_per_query: int = 20,
    ):
        """
        Args:
            llm_client: LLM 客户端（用于抽取实体）
            nebula_client: NebulaGraph 客户端
            entity_types: 实体类型列表
            max_chunks_per_query: 每次查询最多处理多少个 chunks
        """
        self.llm_client = llm_client
        self.nebula_client = nebula_client
        self.entity_types = entity_types or ["PERSON", "ORG", "PRODUCT", "LOCATION", "EVENT", "CONCEPT"]
        self.max_chunks_per_query = max_chunks_per_query
        # 子图 Hash 缓存：key=doc_id:hash(chunk_ids), value={entities, relations}
        self._subgraph_cache: Dict[str, Dict] = {}

    def build(
        self,
        doc_id: str,
        chunks: List[Dict],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        构建/查询实体图谱（按 plane 设计：增量扩展 + 子图缓存）

        Args:
            doc_id: 文档 ID
            chunks: 当前检索到的 chunks（用于抽取新实体）
            force_rebuild: 是否强制重建（清空后重新构建）

        Returns:
            {
                "entities": [...],
                "relations": [...],
                "from_cache": bool,    # 是否无新增（直接复用）
                "new_entities": int,
                "new_relations": int,
            }
        """
        if not self.nebula_client:
            logger.warning("LazyEntityBuilder: 无 NebulaGraph 客户端，跳过")
            return {"entities": [], "relations": [], "from_cache": False, "new_entities": 0, "new_relations": 0}

        target_chunks = chunks[:self.max_chunks_per_query]
        if not target_chunks:
            return {"entities": [], "relations": [], "from_cache": False, "new_entities": 0, "new_relations": 0}

        # force_rebuild: 清空该文档的实体图谱后重建
        if force_rebuild:
            self.nebula_client.clear_entities_by_doc_id(doc_id)
            existing_entities = []
        else:
            existing_entities = self._get_existing_entities(doc_id)

        processed_cids = set()
        for e in existing_entities:
            try:
                cids = json.loads(e.get("chunk_ids", "[]"))
                if isinstance(cids, list):
                    processed_cids.update(c for c in cids if c)
            except Exception:
                pass

        # Step 2: 过滤出本次新增的 chunks（plane：渐进式演进、只处理新 chunks）
        new_chunks = [
            c for c in target_chunks
            if c.get("chunk_id") and c.get("chunk_id") not in processed_cids
        ]

        if not new_chunks:
            # 无新增，直接返回已存储图谱
            existing_relations = self._get_existing_relations(doc_id, existing_entities)
            logger.info(
                f"LazyEntityBuilder: 无新增 chunks，复用图谱 {len(existing_entities)} 实体 (doc={doc_id})"
            )
            return {
                "entities": existing_entities,
                "relations": existing_relations,
                "from_cache": True,
                "new_entities": 0,
                "new_relations": 0,
            }

        # Step 3: 子图缓存（plane：相同/相似 chunks 直接复用）—— 避免重复 LLM 调用
        cache_key = f"{doc_id}:{hashlib.md5(json.dumps(sorted(c.get('chunk_id','') for c in new_chunks), sort_keys=True).encode()).hexdigest()[:16]}"
        cached = self._subgraph_cache.get(cache_key)
        if cached:
            extracted = cached
            logger.info(f"LazyEntityBuilder: 子图缓存命中，复用 {len(new_chunks)} chunks 抽取结果 (doc={doc_id})")
        else:
            extracted = self._extract_from_chunks(new_chunks, doc_id)
            if extracted.get("entities") or extracted.get("relations"):
                self._subgraph_cache[cache_key] = extracted

        if not extracted.get("entities") and not extracted.get("relations"):
            existing_relations = self._get_existing_relations(doc_id, existing_entities)
            return {
                "entities": existing_entities,
                "relations": existing_relations,
                "from_cache": True,
                "new_entities": 0,
                "new_relations": 0,
            }

        # Step 4: 合并到已有图谱（新实体 INSERT，已有实体 UPDATE chunk_ids）
        existing_relations = self._get_existing_relations(doc_id, existing_entities) if existing_entities else []
        merged_entities, merged_relations, new_ent_count, new_rel_count, new_relations = self._merge_into_existing(
            doc_id=doc_id,
            existing_entities=existing_entities,
            existing_relations=existing_relations,
            extracted=extracted,
        )

        # Step 5: 持久化到 NebulaGraph（INSERT 新实体/关系，UPDATE 已有实体 chunk_ids）
        self._persist_incremental(
            doc_id=doc_id,
            existing_entities=existing_entities,
            entities_to_insert=[e for e in merged_entities if e.get("name", "") not in {x.get("name", "") for x in existing_entities}],
            entities_to_update=[e for e in merged_entities if e.get("name", "") in {x.get("name", "") for x in existing_entities}],
            new_relations=new_relations,
        )

        logger.info(
            f"LazyEntityBuilder: 增量扩展完成: +{new_ent_count} 实体, +{new_rel_count} 关系 "
            f"(本次 {len(new_chunks)} chunks, 总 {len(merged_entities)} 实体 (doc={doc_id})"
        )

        return {
            "entities": merged_entities,
            "relations": merged_relations,
            "from_cache": False,
            "new_entities": new_ent_count,
            "new_relations": new_rel_count,
        }

    def _get_existing_entities(self, doc_id: str) -> List[Dict]:
        """从 NebulaGraph 获取已存储的实体"""
        if not self.nebula_client:
            return []

        try:
            with self.nebula_client.get_session() as session:
                # 先切换 space 并检查结果
                r = session.execute(f"USE {self.nebula_client.space_name};")
                if not r.is_succeeded():
                    logger.warning(f"切换 space 失败: {r.error_msg()}")
                    return []

                # 修复：LOOKUP 不支持直接接 LIMIT，需通过管道符 | 传递
                query = (
                    f'LOOKUP ON Entity WHERE Entity.doc_id == "{doc_id}" '
                    f'YIELD Entity.id AS id, Entity.name AS name, '
                    f'Entity.entity_type AS entity_type, Entity.description AS description, '
                    f'Entity.chunk_ids AS chunk_ids '
                    f'| LIMIT 1000;'
                )
                result = session.execute(query)
                
                if not result.is_succeeded():
                    logger.warning(f"查询实体失败: {result.error_msg()}")
                    return []

                entities = []
                for row in result.rows():
                    try:
                        def get_str_val(val):
                            if val is None:
                                return ""
                            if val.getType() == 5:  # STRING
                                s = val.get_sVal()
                                return s.decode('utf-8') if isinstance(s, bytes) else s
                            return str(val)

                        entities.append({
                            "id": get_str_val(row.values[0]),
                            "name": get_str_val(row.values[1]),
                            "entity_type": get_str_val(row.values[2]),
                            "description": get_str_val(row.values[3]),
                            "chunk_ids": get_str_val(row.values[4]),
                        })
                    except Exception as e:
                        logger.debug(f"解析实体行失败: {e}")
                        continue

                return entities

        except Exception as e:
            logger.warning(f"获取已存在实体失败: {e}")
            return []

    def _get_existing_relations(self, doc_id: str, entities: List[Dict]) -> List[Dict]:
        """从 NebulaGraph 获取已存储的关系"""
        if not self.nebula_client or not entities:
            return []

        # 获取所有实体名称
        entity_names = {e.get("name", "") for e in entities if e.get("name")}
        entity_ids = {e.get("id", "") for e in entities if e.get("id")}

        try:
            with self.nebula_client.get_session() as session:
                # 先切换 space 并检查结果
                r = session.execute(f"USE {self.nebula_client.space_name};")
                if not r.is_succeeded():
                    return []

                # 构造 IN 列表
                ids_str = ", ".join(f'"{eid}"' for eid in entity_ids if eid)
                if not ids_str:
                    return []

                # 修复：GO 语句的 LIMIT 必须通过管道符 | 传递；src/dst 函数参数用小写 edge
                query = (
                    f'GO FROM {ids_str} OVER RELATION '
                    f'YIELD src(edge) AS src_id, dst(edge) AS dst_id, '
                    f'RELATION.relation_type AS relation_type, '
                    f'RELATION.description AS description, '
                    f'RELATION.strength AS strength '
                    f'| LIMIT 500;'
                )
                result = session.execute(query)

                if not result.is_succeeded():
                    return []

                relations = []
                for row in result.rows():
                    try:
                        def get_str_val(val):
                            if val is None:
                                return ""
                            if val.getType() == 5:
                                s = val.get_sVal()
                                return s.decode('utf-8') if isinstance(s, bytes) else s
                            return str(val)

                        relations.append({
                            "src_id": get_str_val(row.values[0]),
                            "dst_id": get_str_val(row.values[1]),
                            "relation_type": get_str_val(row.values[2]),
                            "description": get_str_val(row.values[3]),
                            "strength": float(row.values[4].get_fVal() or 1.0) if row.values[4] else 1.0,
                        })
                    except:
                        continue

                return relations

        except Exception as e:
            logger.warning(f"获取已存在关系失败: {e}")
            return []

    def _extract_single_chunk(
        self, chunk: Dict, doc_id: str, entity_types_str: str
    ) -> Tuple[str, Dict]:
        """单 chunk LLM 抽取，返回 (chunk_id, parsed)"""
        text = chunk.get("text", "").strip()
        if not text or len(text) < 30:
            return chunk.get("chunk_id", ""), {"entities": [], "relations": []}

        text_input = text[:800]
        prompt = _EXTRACT_PROMPT.replace("{text}", text_input).replace(
            "{entity_types}", entity_types_str
        )
        try:
            raw = self.llm_client.chat(
                prompt=prompt,
                system_prompt="你是一个严格的信息抽取助手，只输出 JSON。",
                max_tokens=1024,
                temperature=0.0,
                no_think=True,
            )
            parsed = _try_extract_json(raw)
            return chunk.get("chunk_id", ""), parsed
        except Exception as e:
            logger.debug(f"LazyEntityBuilder LLM 解析失败: {e}")
            return chunk.get("chunk_id", ""), {"entities": [], "relations": []}

    def _extract_from_chunks(self, chunks: List[Dict], doc_id: str) -> Dict:
        """从 chunks 抽取实体（并行 LLM 调用，显著降低首次构建耗时）"""
        if not self.llm_client:
            logger.warning("LazyEntityBuilder: 无 LLM 客户端，跳过抽取")
            return {"entities": [], "relations": []}

        entity_map: Dict[str, Dict] = {}
        rel_map: Dict[tuple, Dict] = {}
        entity_types_str = "/".join(self.entity_types)

        # 并行调用 LLM（max_workers=4 避免 API 限流）
        max_workers = min(4, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._extract_single_chunk, ch, doc_id, entity_types_str
                ): ch
                for ch in chunks
            }
            for future in as_completed(futures):
                try:
                    chunk_id, parsed = future.result()
                except Exception as e:
                    logger.debug(f"LazyEntityBuilder 并行抽取异常: {e}")
                    continue

                for ent in parsed.get("entities", []):
                    name = ent.get("name", "").strip()
                    if not name:
                        continue
                    if name not in entity_map:
                        eid = self._stable_id(doc_id, name)
                        entity_map[name] = {
                            "id": eid,
                            "name": name,
                            "entity_type": ent.get("type", "CONCEPT"),
                            "description": ent.get("description", ""),
                            "doc_id": doc_id,
                            "chunk_ids": [chunk_id] if chunk_id else [],
                        }
                    else:
                        existing = entity_map[name]
                        new_desc = ent.get("description", "")
                        if len(new_desc) > len(existing.get("description", "")):
                            existing["description"] = new_desc
                        if chunk_id and chunk_id not in existing["chunk_ids"]:
                            existing["chunk_ids"].append(chunk_id)

                for rel in parsed.get("relations", []):
                    src_name = rel.get("src", "").strip()
                    dst_name = rel.get("dst", "").strip()
                    rtype = rel.get("type", "RELATED").strip()
                    if not src_name or not dst_name:
                        continue
                    key = (src_name, dst_name, rtype)
                    if key not in rel_map:
                        rel_map[key] = {
                            "src_name": src_name,
                            "dst_name": dst_name,
                            "relation_type": rtype,
                            "description": rel.get("description", ""),
                            "strength": float(rel.get("strength", 1.0)),
                            "chunk_id": chunk_id,
                        }

        final_entities = []
        for ent in entity_map.values():
            ent["chunk_ids"] = json.dumps(ent["chunk_ids"], ensure_ascii=False)
            final_entities.append(ent)

        final_relations = []
        for rel in rel_map.values():
            src_ent = entity_map.get(rel["src_name"])
            dst_ent = entity_map.get(rel["dst_name"])
            if src_ent and dst_ent:
                final_relations.append({
                    "src_id": src_ent["id"],
                    "dst_id": dst_ent["id"],
                    "relation_type": rel["relation_type"],
                    "description": rel["description"],
                    "strength": rel["strength"],
                    "chunk_id": rel.get("chunk_id", ""),
                })

        return {"entities": final_entities, "relations": final_relations}

    def _persist_to_nebula(self, doc_id: str, extracted: Dict):
        """持久化到 NebulaGraph（全量插入）"""
        if not self.nebula_client:
            return

        try:
            self.nebula_client.insert_entity_graph(
                doc_id=doc_id,
                entities=extracted.get("entities", []),
                relations=extracted.get("relations", []),
            )
        except Exception as e:
            logger.warning(f"持久化实体失败: {e}")

    def _merge_into_existing(
        self,
        doc_id: str,
        existing_entities: List[Dict],
        existing_relations: List[Dict],
        extracted: Dict,
    ) -> Tuple[List[Dict], List[Dict], int, int, List[Dict]]:
        """合并抽取结果到已有图谱，返回 (merged_entities, merged_relations, new_ent_count, new_rel_count, new_relations)"""
        existing_rel_keys = {
            (r.get("src_id", ""), r.get("dst_id", ""), r.get("relation_type", ""))
            for r in existing_relations
        }

        entity_map: Dict[str, Dict] = {}
        for e in existing_entities:
            name = e.get("name", "")
            if name:
                try:
                    cids = json.loads(e.get("chunk_ids", "[]"))
                except Exception:
                    cids = []
                entity_map[name] = {**e, "chunk_ids": cids}

        new_ent_count = 0
        new_rel_count = 0
        new_relations: List[Dict] = []

        for ent in extracted.get("entities", []):
            name = ent.get("name", "").strip()
            if not name:
                continue
            eid = self._stable_id(doc_id, name)
            chunk_ids = ent.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = json.loads(chunk_ids)
                except Exception:
                    chunk_ids = []

            if name in entity_map:
                existing = entity_map[name]
                merged_cids = list(set(existing.get("chunk_ids", []) + chunk_ids))
                existing["chunk_ids"] = merged_cids
                entity_map[name] = existing
            else:
                entity_map[name] = {
                    "id": eid,
                    "name": name,
                    "entity_type": ent.get("entity_type", "CONCEPT"),
                    "description": ent.get("description", ""),
                    "doc_id": doc_id,
                    "chunk_ids": chunk_ids,
                }
                new_ent_count += 1

        for rel in extracted.get("relations", []):
            src_id = rel.get("src_id", "")
            dst_id = rel.get("dst_id", "")
            if not src_id or not dst_id:
                continue
            rtype = rel.get("relation_type", "RELATED").strip()
            key = (src_id, dst_id, rtype)
            if key not in existing_rel_keys:
                existing_rel_keys.add(key)
                new_relations.append({
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "relation_type": rtype,
                    "description": rel.get("description", ""),
                    "strength": float(rel.get("strength", 1.0)),
                    "chunk_id": rel.get("chunk_id", ""),
                })
                new_rel_count += 1

        rel_list = existing_relations + new_relations
        final_entities = []
        for ent in entity_map.values():
            cids = ent.get("chunk_ids", [])
            ent["chunk_ids"] = json.dumps(cids, ensure_ascii=False)
            final_entities.append(ent)

        return final_entities, rel_list, new_ent_count, new_rel_count, new_relations

    def _persist_incremental(
        self,
        doc_id: str,
        existing_entities: List[Dict],
        entities_to_insert: List[Dict],
        entities_to_update: List[Dict],
        new_relations: List[Dict],
    ):
        """增量持久化：INSERT 新实体/关系，UPDATE 已有实体的 chunk_ids"""
        if not self.nebula_client:
            return

        # INSERT 新实体
        if entities_to_insert:
            self.nebula_client.insert_entity_graph(
                doc_id=doc_id,
                entities=entities_to_insert,
                relations=[],
            )

        # UPDATE 已有实体的 chunk_ids（合并后的 chunk_ids）
        for ent in entities_to_update:
            self.nebula_client.update_entity_chunk_ids(
                entity_id=ent.get("id", ""),
                chunk_ids_json=ent.get("chunk_ids", "[]"),
            )

        # INSERT 新关系
        if new_relations:
            self.nebula_client.insert_entity_graph(
                doc_id=doc_id,
                entities=[],
                relations=new_relations,
            )

    @staticmethod
    def _stable_id(doc_id: str, name: str) -> str:
        """生成稳定的实体 VID"""
        raw = f"{doc_id}::{name}"
        return hashlib.md5(raw.encode()).hexdigest()[:20]

    def get_entity_neighbors(self, entity_name: str, doc_id: str = "", hops: int = 2) -> List[Dict]:
        """查询实体的 N 跳邻居（公共接口）"""
        if not self.nebula_client:
            return []

        try:
            return self.nebula_client.get_entity_neighbors(
                entity_name=entity_name,
                doc_id=doc_id,
                hops=hops,
            )
        except Exception as e:
            logger.warning(f"查询实体邻居失败: {e}")
            return []