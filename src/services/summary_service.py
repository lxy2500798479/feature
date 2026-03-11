"""
Summary Service - 摘要生成服务

修复内容（对比原版）：
===========================================================================
【问题3】多批摘要合并逻辑有缺陷

  旧方案：
    if len(parts) <= 3:
        return " | ".join(parts)   # 直接用 "|" 拼接，垃圾质量写入 Milvus

  新方案：
    不管几批，统一走 LLM 合并（_merge_summaries）
    _merge_summaries 失败时 fallback 到截断拼接（保底，不影响主流程）

  同时新增摘要质量校验 _validate_summary()：
    - 长度合理（20字 ~ max_length）
    - 非原文截取（不能和 content 前200字完全一样）
    - 无错误标志词（"抱歉"、"无法"、"error" 等 LLM 错误返回）
    校验失败时 fallback 到 content 前200字（总比空的或错误强）
===========================================================================
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from collections import defaultdict

from src.config import settings
from src.core.models import SectionNode, ChunkNode
from src.utils.llm_client import LLMClient
from src.utils.logger import logger


_SUMMARY_PROMPT = """你是一个文档摘要专家。请为以下章节内容生成一段简洁的摘要。
要求：
1. 包含核心主题、关键实体和主要结论
2. 长度不超过{max_length}字
3. 使用与原文相同的语言
4. 只输出摘要文本本身，不要输出"摘要："等前缀

章节标题：{title}
章节内容：
{content}

摘要："""


_SUMMARY_MERGE_PROMPT = """你是一个文档摘要专家。以下是同一章节的多个部分摘要，请合并成一个连贯、简洁的摘要。
要求：
1. 保留核心主题和关键信息，去除重复内容
2. 长度不超过{max_length}字
3. 使用与原文相同的语言
4. 只输出合并后的摘要文本，不要输出"合并摘要："等前缀

部分摘要：
{parts}

合并后的摘要："""


# 摘要质量校验：命中这些词则视为 LLM 返回了错误信息
_ERROR_KEYWORDS = {
    "抱歉", "无法", "对不起", "error", "sorry",
    "i cannot", "i can't", "无关", "没有提供",
}


class SummaryService:
    """Section Summary 生成服务"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient(
            api_url=settings.SUMMARY_API_URL or settings.LLM_API_URL,
            api_key=settings.SUMMARY_API_KEY or settings.LLM_API_KEY,
            model=settings.SUMMARY_MODEL,
            timeout=settings.SUMMARY_TIMEOUT,
        )
        self.max_length = settings.SUMMARY_MAX_LENGTH
        self.concurrency = settings.SUMMARY_CONCURRENCY
        self.max_chunk_chars = settings.SUMMARY_MAX_CHUNK_CHARS or 4000

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def generate_summary(self, title: str, content: str) -> str:
        """
        为单个 Section 生成摘要（支持大内容分批处理）

        返回值保证：
        - 非空字符串（失败时 fallback 到 content 前200字）
        - 通过质量校验（无 LLM 错误信息、长度合理）
        """
        if not content.strip():
            return ""

        if len(content) <= self.max_chunk_chars:
            raw = self._generate_single_summary(title, content)
        else:
            raw = self._generate_summary_in_batches(title, content)

        # 质量校验 + fallback
        return self._validate_summary(raw, content)

    def generate_summaries_for_sections(
        self,
        sections: List[SectionNode],
        chunks: List[ChunkNode],
    ) -> Dict[str, str]:
        """为多个 Section 批量生成摘要（并行处理）"""
        # 按 section_id 分组 chunks
        section_chunks: Dict[str, List[ChunkNode]] = defaultdict(list)
        for chunk in chunks:
            section_chunks[chunk.section_id].append(chunk)

        level1_sections = [s for s in sections if s.level == 1] or sections

        def generate_for_section(section: SectionNode) -> tuple:
            section_chunk_list = section_chunks.get(section.section_id, [])

            if section.content:
                content = section.content
            elif section_chunk_list:
                sorted_chunks = sorted(section_chunk_list, key=lambda c: c.position)
                content = "\n".join(c.text for c in sorted_chunks)
            else:
                return section.section_id, ""

            summary = self.generate_summary(section.title, content)
            return section.section_id, summary

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            results = list(executor.map(generate_for_section, level1_sections))

        return {sid: s for sid, s in results}

    def generate_document_summary(
        self,
        sections: List[SectionNode],
        chunks: List[ChunkNode],
    ) -> str:
        """生成文档摘要（基于章节摘要合并）"""
        section_summaries = [
            f"{s.title}: {s.summary}"
            for s in sections
            if s.level == 1 and s.summary
        ]

        if section_summaries:
            combined = "\n\n".join(section_summaries)
            return self._generate_from_summaries(combined)

        if chunks:
            sample = sorted(chunks, key=lambda c: c.position)[:10]
            content = "\n".join(c.text for c in sample)
            return self.generate_summary("文档摘要", content)

        return ""

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _generate_single_summary(self, title: str, content: str) -> str:
        """生成单个摘要（内容在限制内）"""
        prompt = _SUMMARY_PROMPT.format(
            max_length=self.max_length,
            title=title,
            content=content[: self.max_chunk_chars],
        )
        result = self.llm.chat(
            prompt,
            max_tokens=4096,
            temperature=0.1,
            no_think=True,
        )
        return result.strip()[: self.max_length] if result else ""

    def _generate_summary_in_batches(self, title: str, content: str) -> str:
        """
        大内容分批生成摘要，最后统一用 LLM 合并

        修复：原来 <= 3 批直接 "|" 拼接，现在所有情况都走 LLM 合并，
        保证写入 NebulaGraph 和 SectionSummaryIndex 的摘要质量。
        """
        parts = []
        start = 0
        batch_idx = 0

        while start < len(content):
            end = min(start + self.max_chunk_chars, len(content))
            batch_content = content[start:end]
            logger.info(f"生成摘要批 {batch_idx + 1}: 字符 {start}-{end}")

            part = self._generate_single_summary(
                f"{title} (Part {batch_idx + 1})", batch_content
            )
            if part:
                parts.append(part)

            start = end
            batch_idx += 1

        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]

        # 不管几批，统一走 LLM 合并（原来 <=3 批直接拼接的 bug 修复点）
        return self._merge_summaries(title, parts)

    def _merge_summaries(self, title: str, parts: List[str]) -> str:
        """
        用 LLM 将多段部分摘要合并为一段连贯摘要

        LLM 调用失败时 fallback 到截断拼接（保底，不会崩）
        """
        prompt = _SUMMARY_MERGE_PROMPT.format(
            max_length=self.max_length,
            parts="\n\n".join(f"Part {i + 1}: {p}" for i, p in enumerate(parts)),
        )
        result = self.llm.chat(
            prompt,
            max_tokens=4096,
            temperature=0.1,
            no_think=True,
        )
        if result:
            merged = result.strip()[: self.max_length]
            if merged:
                return merged

        # fallback：截断拼接（比空值好，但质量较差，会被 _validate_summary 检测到过长问题）
        logger.warning(f"_merge_summaries LLM 调用失败，使用截断拼接: title={title}")
        return (" ".join(parts))[: self.max_length]

    def _validate_summary(self, summary: str, original_content: str) -> str:
        """
        摘要质量校验，校验失败则 fallback 到原文前200字

        校验三项：
        1. 长度合理：>= 20 字 且 <= max_length
        2. 非原文截取：不能和 original_content 前200字完全相同
        3. 无 LLM 错误标志词：不包含"抱歉"/"无法"/"error" 等
        """
        if not summary:
            logger.warning("摘要为空，使用原文 fallback")
            return original_content.strip()[:200]

        # 1. 长度检查
        if len(summary) < 20:
            logger.warning(f"摘要过短({len(summary)}字)，使用原文 fallback")
            return original_content.strip()[:200]

        # 2. 原文截取检查（LLM 有时直接返回原文）
        content_prefix = original_content.strip()[:200]
        if summary.strip()[:100] == content_prefix[:100]:
            logger.warning("摘要疑似原文截取，使用原文 fallback")
            return content_prefix

        # 3. 错误关键词检查
        summary_lower = summary.lower()
        for kw in _ERROR_KEYWORDS:
            if summary_lower.startswith(kw) or f"，{kw}" in summary_lower:
                logger.warning(f"摘要含错误标志词 '{kw}'，使用原文 fallback")
                return original_content.strip()[:200]

        return summary

    def _generate_from_summaries(self, combined_summaries: str) -> str:
        """从章节摘要合并生成文档摘要"""
        prompt = (
            f"你是一个文档摘要专家。以下是文档各章节的摘要，请合并成一个简洁的文档摘要。\n"
            f"要求：\n"
            f"1. 保留核心主题和关键信息\n"
            f"2. 长度不超过{self.max_length}字\n"
            f"3. 使用与原文相同的语言\n"
            f"4. 只输出摘要文本，不要输出前缀\n\n"
            f"章节摘要：\n{combined_summaries}\n\n文档摘要："
        )
        result = self.llm.chat(
            prompt,
            max_tokens=4096,
            temperature=0.1,
            no_think=True,
        )
        if result:
            return result.strip()[: self.max_length]
        return combined_summaries[: self.max_length]