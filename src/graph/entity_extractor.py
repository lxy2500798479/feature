"""
LLM 驱动的实体关系抽取器 (Enhanced-KG)

设计目标：
  用 Qwen3.5-27b-fp8 从每个 Chunk 抽取带类型的实体 + 语义关系，
  写入 NebulaGraph 的 Entity 节点和 RELATION 边。

抽取粒度：每个 Chunk 独立抽取（并发执行），保留 chunk_id 溯源。
成本控制：异步后台执行，不阻塞文档上传响应。
"""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from src.utils.logger import logger

# ── Prompt ────────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """直接从下面文本中提取实体和关系，输出纯JSON，不要其他内容：

文本：{text}

输出格式：{{"entities":[{"name":"名","type":"类型","description":"描述"}],"relations":[{"src":"A","dst":"B","type":"关系","strength":0.9}]}}"""


def _try_extract_json(text: str) -> dict:
    """从任意文本中提取 JSON（健壮提取）"""
    import re
    # 去掉 markdown 代码块、编号列表头
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text)
    text = re.sub(r"^\d+\.\s+\*\*.*?\*\*:?", "", text, flags=re.MULTILINE)
    text = text.strip()

    # 1. 直接解析整段
    try:
        parsed = json.loads(text)
        if "entities" in parsed:
            return parsed
    except:
        pass

    # 2. 逐行找 {"name": ...} 或 {"src": ...} 格式
    entities = []
    relations = []
    for line in text.split('\n'):
        line = line.strip().lstrip('*-').strip()
        # 去掉前缀如 "Entities:"
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


class EntityExtractor:
    """LLM 驱动的实体关系抽取器"""

    def __init__(self, llm_client=None, max_workers: int = 3):
        self.llm_client = llm_client
        self.max_workers = max_workers

    def extract_from_chunks(
        self,
        chunks: List,          # List[ChunkNode]
        doc_id: str,
        max_chunks: int = 60,  # 限制处理 chunk 数，避免耗时过长
    ) -> Dict[str, List[Dict]]:
        """并发从多个 chunk 抽取实体关系，合并去重后返回

        Returns:
            {
                "entities": [{id, name, entity_type, description, doc_id, chunk_id}],
                "relations": [{src_id, dst_id, relation_type, description, strength}]
            }
        """
        if not self.llm_client:
            logger.warning("EntityExtractor: 未配置 LLM，跳过实体抽取")
            return {"entities": [], "relations": []}

        target_chunks = chunks[:max_chunks]
        logger.info(f"EntityExtractor: 开始处理 {len(target_chunks)} 个 chunk (doc={doc_id})")

        # 并发抽取
        chunk_results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(self._extract_from_one_chunk, chunk, doc_id): chunk
                for chunk in target_chunks
            }
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if res:
                        chunk_results.append(res)
                except Exception as e:
                    logger.warning(f"EntityExtractor chunk 抽取失败: {e}")

        # 合并 + 去重
        return self._merge(chunk_results, doc_id)

    def _extract_from_one_chunk(self, chunk, doc_id: str) -> Optional[Dict]:
        """从单个 chunk 调用 LLM 抽取"""
        text = chunk.text.strip()
        if not text or len(text) < 30:
            return None

        # 限制送入 LLM 的文本长度
        text_input = text[:800]
        prompt = _EXTRACT_PROMPT.replace("{text}", text_input)

        try:
            raw = self.llm_client.chat(
                prompt=prompt,
                system_prompt="你是一个严格的信息抽取助手，只输出 JSON。",
                max_tokens=1024,
                temperature=0.0,
                no_think=True,
            )
            parsed = _try_extract_json(raw)
            return {
                "chunk_id": chunk.chunk_id,
                "entities": parsed.get("entities", []),
                "relations": parsed.get("relations", []),
            }
        except Exception as e:
            logger.debug(f"EntityExtractor LLM 解析失败 (chunk={chunk.chunk_id}): {e}")
            return None

    def _merge(self, chunk_results: List[Dict], doc_id: str) -> Dict[str, List[Dict]]:
        """合并多个 chunk 的结果，对同名实体去重，生成稳定 ID"""
        # name → {id, entity_type, description, doc_id, chunk_id}
        entity_map: Dict[str, Dict] = {}
        # (src_name, dst_name, rel_type) → relation
        rel_map: Dict[tuple, Dict] = {}

        for cr in chunk_results:
            chunk_id = cr.get("chunk_id", "")

            for ent in cr.get("entities", []):
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
                        "chunk_id": chunk_id,
                    }
                else:
                    # 补充 description（取更长的那个）
                    existing = entity_map[name]
                    new_desc = ent.get("description", "")
                    if len(new_desc) > len(existing.get("description", "")):
                        existing["description"] = new_desc

            for rel in cr.get("relations", []):
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
                    }

        # 解析 relation 的 src_id / dst_id（只保留两端实体都存在的关系）
        final_entities = list(entity_map.values())
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
                })

        logger.info(
            f"EntityExtractor 合并完成: {len(final_entities)} 实体, "
            f"{len(final_relations)} 关系 (doc={doc_id})"
        )
        return {"entities": final_entities, "relations": final_relations}

    @staticmethod
    def _stable_id(doc_id: str, name: str) -> str:
        """生成稳定的实体 VID（截断 MD5）"""
        raw = f"{doc_id}::{name}"
        return hashlib.md5(raw.encode()).hexdigest()[:20]
