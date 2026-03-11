"""
答案合成器 - 将检索结果合成最终答案，支持流式和非流式输出

修复内容（对比原版）：
===========================================================================
【问题2-A】context 截断太粗暴（硬截 600 字）
  旧：text = item.get("text", "")[:600]
  新：按 score 排序后，动态分配 token 预算，高分 chunk 获得更多字符

【问题2-B】没有来源标注，LLM 无法引用章节出处
  旧：f"{i}. {text}"
  新：f"[{i}] 来源：{section_title}（{doc_title}）\n{text}"
      prompt 要求 LLM 用 [编号] 形式标注引用

【问题2-C】vector_context 未按相关性排序
  旧：直接取 vector_context[:8]，顺序不定
  新：先按 distance（余弦相似度）降序排序，最相关的排最前面
===========================================================================
"""
from typing import Optional, List, Dict, Any, Generator
from src.utils.logger import logger

# 单条 chunk 最大字符数（动态分配，高分 chunk 可用更多）
_MAX_CHARS_PER_CHUNK = 800
_MIN_CHARS_PER_CHUNK = 200
# 总 context 最大字符数（避免超出 LLM context window）
_MAX_TOTAL_CONTEXT_CHARS = 6000
# 参与合成的最大 chunk 数
_MAX_CHUNKS = 8


def _sort_chunks_by_score(chunks: List[dict]) -> List[dict]:
    """
    按相关性分数降序排列 chunks

    Milvus 返回的字段：distance（COSINE 相似度，越大越相关）
    graph_traversal 来的 chunk 没有 distance，统一给 0.5 中等分
    """
    def _score(c: dict) -> float:
        # distance 字段：Milvus 向量检索结果
        if "distance" in c:
            return float(c["distance"])
        # score 字段：部分路径使用
        if "score" in c:
            return float(c["score"])
        # 图谱邻居补充的 chunk，给中等分（不应排在向量命中之前）
        return 0.5

    return sorted(chunks, key=_score, reverse=True)


def _get_chunk_source(chunk: dict) -> str:
    """
    从 chunk 中提取来源描述（章节名 + 文档名）

    支持两种数据来源：
    1. Milvus 检索结果（section_id 形如 doc_xxx_sec_3）
    2. 图谱遍历结果（直接带 section_title 字段）
    """
    # 优先使用显式字段（问题5 Milvus schema 扩展后可用）
    section_title = chunk.get("section_title", "")
    doc_title = chunk.get("doc_title", "")

    # fallback：从 section_id 解析（格式 doc_xxx_sec_N）
    if not section_title:
        section_id = chunk.get("section_id", "")
        if section_id:
            # 取最后的 sec_N 部分作为简易标识
            parts = section_id.rsplit("_sec_", 1)
            if len(parts) == 2:
                section_title = f"第{parts[1]}节"

    if section_title and doc_title:
        return f"{doc_title} · {section_title}"
    elif section_title:
        return section_title
    elif doc_title:
        return doc_title
    else:
        return "参考资料"


def _build_context_block(chunks: List[dict]) -> str:
    """
    将 chunks 组装成带编号和来源的 context 字符串

    策略：
    - 高分 chunk（前3个）分配较多字符（最多 _MAX_CHARS_PER_CHUNK）
    - 低分 chunk 分配较少字符（最少 _MIN_CHARS_PER_CHUNK）
    - 总字符超过 _MAX_TOTAL_CONTEXT_CHARS 时停止添加
    """
    lines = ["## 参考资料：\n"]
    total_chars = 0
    n = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "").strip()
        if not text:
            continue

        # 动态字符预算：前 3 个 chunk 给满额，其余递减
        if i <= 3:
            char_limit = _MAX_CHARS_PER_CHUNK
        elif i <= 6:
            char_limit = (_MAX_CHARS_PER_CHUNK + _MIN_CHARS_PER_CHUNK) // 2
        else:
            char_limit = _MIN_CHARS_PER_CHUNK

        # 总量控制
        remaining = _MAX_TOTAL_CONTEXT_CHARS - total_chars
        if remaining <= _MIN_CHARS_PER_CHUNK:
            break
        char_limit = min(char_limit, remaining)

        truncated = text[:char_limit]
        source = _get_chunk_source(chunk)

        lines.append(f"[{i}] 来源：{source}")
        lines.append(truncated)
        lines.append("")  # 空行分隔

        total_chars += len(truncated)

    return "\n".join(lines)


def _build_prompt(
    query: str,
    vector_context: Optional[List[dict]],
    graph_context: Optional[Dict],
) -> tuple:
    """
    构建 system_prompt 和 user_prompt

    修复：
    1. 合并前先按 score 降序排列
    2. 每条 chunk 带编号和来源
    3. 要求 LLM 用 [编号] 引用来源
    """
    context_parts = []

    # ── 向量检索结果（主体，已按分数排序）────────────────────────────────
    if vector_context:
        sorted_chunks = _sort_chunks_by_score(vector_context)
        top_chunks = sorted_chunks[:_MAX_CHUNKS]
        context_parts.append(_build_context_block(top_chunks))

    # ── 图谱关联内容 ──────────────────────────────────────────────────────
    if graph_context:
        graph_chunks = graph_context.get("chunks", [])
        communities = graph_context.get("communities", [])
        graph_entities = graph_context.get("graph_entities", [])

        if graph_chunks:
            gc_lines = ["## 图谱关联内容：\n"]
            for item in graph_chunks[:3]:
                text = item.get("text", "").strip()[:300]
                source = _get_chunk_source(item)
                if text:
                    gc_lines.append(f"- [{source}] {text}")
            context_parts.append("\n".join(gc_lines))

        if graph_entities:
            ge_lines = ["## 实体关系（子图遍历）：\n"]
            for ge in graph_entities[:6]:
                entity = ge.get("entity", "")
                related = ge.get("related", "")
                rel_type = ge.get("relation_type", "")
                desc = ge.get("description", "")[:100]
                if entity and related:
                    ge_lines.append(f"- {entity} --[{rel_type}]--> {related}: {desc}")
            context_parts.append("\n".join(ge_lines))

        if communities:
            cm_lines = ["## 社区摘要：\n"]
            for c in communities[:3]:
                cm_lines.append(f"- {c.get('summary', '')[:300]}")
            context_parts.append("\n".join(cm_lines))

    context = "\n\n".join(context_parts)

    system_prompt = (
        "你是一个专业的知识库问答助手。"
        "根据用户问题和检索到的上下文信息，给出准确、详细的回答。\n"
        "要求：\n"
        "1. 回答时用 [编号] 标注信息来源，例如：「唐三觉醒了蓝银草 [1]」\n"
        "2. 如果上下文中没有相关信息，如实告知用户，不要编造\n"
        "3. 回答使用中文，条理清晰\n"
        "4. 优先使用编号靠前（相关性更高）的参考资料"
    )
    user_prompt = (
        f"用户问题：{query}\n\n"
        f"{context}\n\n"
        "请根据以上检索结果回答用户问题，并用 [编号] 标注引用来源。"
    )
    return system_prompt, user_prompt


class AnswerSynthesizer:
    """答案合成器 - 将检索结果合成最终答案"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def synthesize(
        self,
        query: str,
        graph_context: Optional[Dict] = None,
        vector_context: Optional[List[dict]] = None,
    ) -> str:
        """非流式合成答案"""
        if not vector_context and not graph_context:
            return "抱歉，我没有找到相关信息来回答这个问题。"

        system_prompt, user_prompt = _build_prompt(query, vector_context, graph_context)

        if self.llm_client:
            try:
                answer = self.llm_client.chat(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=2048,
                    temperature=0.3,
                )
                if answer:
                    return answer
            except Exception as e:
                logger.warning(f"LLM 调用失败: {e}")

        return f"根据检索到的 {len(vector_context or [])} 条相关文档内容，请参考上述信息回答您的问题。"

    def synthesize_stream(
        self,
        query: str,
        graph_context: Optional[Dict] = None,
        vector_context: Optional[List[dict]] = None,
    ) -> Generator[str, None, None]:
        """流式合成答案，逐 token yield"""
        if not vector_context and not graph_context:
            yield "抱歉，我没有找到相关信息来回答这个问题。"
            return

        system_prompt, user_prompt = _build_prompt(query, vector_context, graph_context)

        if self.llm_client:
            try:
                gen = self.llm_client.chat_stream(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=2048,
                    temperature=0.3,
                )
                for token in gen:
                    if token:
                        yield token
                return
            except Exception as e:
                logger.warning(f"LLM 流式调用失败，降级为非流式: {e}")

        # 降级：把整段答案作为单个 token 返回
        fallback = self.synthesize(query, graph_context, vector_context)
        yield fallback