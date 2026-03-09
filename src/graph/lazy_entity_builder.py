"""
懒加载实体构建器 (Lazy Entity Builder)

设计目标：
  - 首次查询时：从检索到的 chunks 抽取实体，存入 NebulaGraph
  - 后续查询：直接从 NebulaGraph 读取已构建的实体图谱
  - 越用越快：每次查询都会增量扩展实体图谱

架构：
  Query Time
      ↓
  检查 NebulaGraph 是否有该文档的实体
      ↓
  ┌─ 无 ──→ LLM 抽取实体 ──→ 存入 NebulaGraph
  │
  └─ 有 ──→ 直接使用已存储的实体图谱
"""
import hashlib
import json
import re
from typing import List, Dict, Optional, Any

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

    def build(
        self,
        doc_id: str,
        chunks: List[Dict],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        构建/查询实体图谱

        Args:
            doc_id: 文档 ID
            chunks: 当前检索到的 chunks（用于抽取新实体）
            force_rebuild: 是否强制重建（忽略已存储的实体）

        Returns:
            {
                "entities": [...],      # 实体列表
                "relations": [...],     # 关系列表
                "from_cache": bool,    # 是否从缓存读取
                "new_entities": int,   # 本次新增实体数
                "new_relations": int,   # 本次新增关系数
            }
        """
        if not self.nebula_client:
            logger.warning("LazyEntityBuilder: 无 NebulaGraph 客户端，跳过")
            return {"entities": [], "relations": [], "from_cache": False, "new_entities": 0, "new_relations": 0}

        # Step 1: 检查是否已有实体存储
        existing_entities = []
        if not force_rebuild:
            existing_entities = self._get_existing_entities(doc_id)

        if existing_entities:
            logger.info(f"LazyEntityBuilder: 从缓存读取 {len(existing_entities)} 实体 (doc={doc_id})")
            # 获取已存储的关系
            existing_relations = self._get_existing_relations(doc_id, existing_entities)
            return {
                "entities": existing_entities,
                "relations": existing_relations,
                "from_cache": True,
                "new_entities": 0,
                "new_relations": 0,
            }

        # Step 2: 首次查询，需要从 chunks 抽取实体
        logger.info(f"LazyEntityBuilder: 首次构建，抽取 {len(chunks)} 个 chunks (doc={doc_id})")
        
        # 限制处理的 chunks 数量
        target_chunks = chunks[:self.max_chunks_per_query]
        
        # 抽取实体
        extracted = self._extract_from_chunks(target_chunks, doc_id)
        
        if not extracted.get("entities"):
            logger.info(f"LazyEntityBuilder: 未抽取到实体 (doc={doc_id})")
            return {"entities": [], "relations": [], "from_cache": False, "new_entities": 0, "new_relations": 0}

        # Step 3: 持久化到 NebulaGraph
        self._persist_to_nebula(doc_id, extracted)

        new_entities = len(extracted.get("entities", []))
        new_relations = len(extracted.get("relations", []))

        logger.info(
            f"LazyEntityBuilder: 持久化完成: {new_entities} 实体, {new_relations} 关系 (doc={doc_id})"
        )

        return {
            "entities": extracted.get("entities", []),
            "relations": extracted.get("relations", []),
            "from_cache": False,
            "new_entities": new_entities,
            "new_relations": new_relations,
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
                
                query = f'LOOKUP ON Entity WHERE Entity.doc_id == "{doc_id}" YIELD Entity.id AS id, Entity.name AS name, Entity.entity_type AS entity_type, Entity.description AS description, Entity.chunk_ids AS chunk_ids LIMIT 1000;'
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

                query = f'GO FROM {ids_str} OVER RELATION YIELD src(EDGE) AS src_id, dst(EDGE) AS dst_id, RELATION.relation_type AS relation_type, RELATION.description AS description, RELATION.strength AS strength LIMIT 500;'
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

    def _extract_from_chunks(self, chunks: List[Dict], doc_id: str) -> Dict:
        """从 chunks 抽取实体（简化版，单线程）"""
        if not self.llm_client:
            logger.warning("LazyEntityBuilder: 无 LLM 客户端，跳过抽取")
            return {"entities": [], "relations": []}

        entity_map: Dict[str, Dict] = {}
        rel_map: Dict[tuple, Dict] = {}

        entity_types_str = "/".join(self.entity_types)

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text or len(text) < 30:
                continue

            text_input = text[:800]
            prompt = _EXTRACT_PROMPT.replace("{text}", text_input).replace("{entity_types}", entity_types_str)

            try:
                raw = self.llm_client.chat(
                    prompt=prompt,
                    system_prompt="你是一个严格的信息抽取助手，只输出 JSON。",
                    max_tokens=1024,
                    temperature=0.0,
                    no_think=True,
                )
                parsed = _try_extract_json(raw)
            except Exception as e:
                logger.debug(f"LazyEntityBuilder LLM 解析失败: {e}")
                continue

            chunk_id = chunk.get("chunk_id", "")

            # 处理实体
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
                    # 合并 description
                    new_desc = ent.get("description", "")
                    if len(new_desc) > len(existing.get("description", "")):
                        existing["description"] = new_desc
                    # 追加 chunk_id
                    if chunk_id and chunk_id not in existing["chunk_ids"]:
                        existing["chunk_ids"].append(chunk_id)

            # 处理关系
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

        # 构建最终结果
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
        """持久化到 NebulaGraph"""
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
