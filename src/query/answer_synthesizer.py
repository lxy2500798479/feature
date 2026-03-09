"""
答案合成器 - 将检索结果合成最终答案，支持流式和非流式输出
"""
from typing import Optional, List, Dict, Any, Generator
from src.utils.logger import logger


def _build_prompt(query: str, vector_context: Optional[List[dict]], graph_context: Optional[Dict]) -> tuple:
    """构建 system_prompt 和 user_prompt"""
    context_parts = []

    if vector_context:
        context_parts.append("## 参考资料:\n")
        for i, item in enumerate(vector_context[:8], 1):
            text = item.get("text", "")[:600]
            context_parts.append(f"{i}. {text}")

    if graph_context:
        graph_chunks = graph_context.get("chunks", [])
        communities = graph_context.get("communities", [])
        graph_entities = graph_context.get("graph_entities", [])
        if graph_chunks:
            context_parts.append("\n## 图谱关联内容:\n")
            for item in graph_chunks[:3]:
                context_parts.append(f"- {item.get('text', '')[:300]}")
        if graph_entities:
            context_parts.append("\n## 实体关系（子图遍历）:\n")
            for ge in graph_entities[:6]:
                entity = ge.get("entity", "")
                related = ge.get("related", "")
                rel_type = ge.get("relation_type", "")
                desc = ge.get("description", "")[:100]
                if entity and related:
                    context_parts.append(f"- {entity} --[{rel_type}]--> {related}: {desc}")
        if communities:
            context_parts.append("\n## 社区摘要:\n")
            for c in communities[:3]:
                context_parts.append(f"- {c.get('summary', '')[:300]}")

    context = "\n".join(context_parts)

    system_prompt = (
        "你是一个专业的知识库问答助手。"
        "根据用户问题和检索到的上下文信息，给出准确、详细的回答。"
        "如果上下文中没有相关信息，请如实告知用户。"
        "回答使用中文，条理清晰。"
    )
    user_prompt = f"用户问题: {query}\n\n{context}\n\n请根据以上检索结果回答用户问题。"
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
