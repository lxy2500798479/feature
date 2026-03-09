"""
Summary Service - 摘要生成服务
"""
import asyncio
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

章节标题：{title}
章节内容：
{content}

摘要："""


_SUMMARY_MERGE_PROMPT = """你是一个文档摘要专家。以下是同一章节的多个部分摘要，请合并成一个简洁的摘要。
要求：
1. 保留核心主题和关键信息
2. 长度不超过{max_length}字
3. 使用与原文相同的语言

部分摘要：
{parts}

合并后的摘要："""


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
        self.max_chunk_chars = settings.SUMMARY_MAX_CHUNK_CHARS or 4000  # 单次最大字符数

    def generate_summary(self, title: str, content: str) -> str:
        """为单个 Section 生成摘要（支持大内容分批处理）"""
        if not content.strip():
            return ""

        # 如果内容在限制内，直接生成
        if len(content) <= self.max_chunk_chars:
            return self._generate_single_summary(title, content)

        # 内容太长，分批处理
        return self._generate_summary_in_batches(title, content)

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

        if result:
            return result.strip()[: self.max_length]

        return content.strip()[:200]

    def _generate_summary_in_batches(self, title: str, content: str) -> str:
        """大内容分批生成摘要，最后合并"""
        # 将内容分成多批
        parts = []
        start = 0
        batch_idx = 0

        while start < len(content):
            end = min(start + self.max_chunk_chars, len(content))
            batch_content = content[start:end]

            logger.info(
                f"生成摘要批 {batch_idx + 1}: 字符 {start}-{end}"
            )

            part_summary = self._generate_single_summary(f"{title} (Part {batch_idx + 1})", batch_content)
            parts.append(part_summary)

            start = end
            batch_idx += 1

        # 如果只有一批，直接返回
        if len(parts) == 1:
            return parts[0]

        # 多批合并
        if len(parts) <= 3:
            # 少于等于3批，直接拼接
            return " | ".join(parts)

        # 超过3批，用 LLM 合并
        return self._merge_summaries(title, parts)

    def _merge_summaries(self, title: str, parts: List[str]) -> str:
        """合并多个部分的摘要"""
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
            return result.strip()[: self.max_length]

        # 合并失败，截断拼接
        merged = " | ".join(parts)
        return merged[: self.max_length]

    def generate_summaries_for_sections(
        self,
        sections: List[SectionNode],
        chunks: List[ChunkNode],
    ) -> Dict[str, str]:
        """为多个 Section 批量生成摘要（并行处理）"""
        # 按 section_id 分组 chunks
        section_chunks = defaultdict(list)
        for chunk in chunks:
            section_chunks[chunk.section_id].append(chunk)

        # 优先处理一级 section
        level1_sections = [s for s in sections if s.level == 1]

        if not level1_sections:
            level1_sections = sections

        # 并行生成摘要
        summaries = {}

        def generate_for_section(section: SectionNode) -> tuple:
            section_chunk_list = section_chunks.get(section.section_id, [])

            # 优先使用 section.content（已分配的）
            if section.content:
                content = section.content
            elif section_chunk_list:
                section_chunks_sorted = sorted(section_chunk_list, key=lambda c: c.position)
                content = "\n".join(c.text for c in section_chunks_sorted)
            else:
                return section.section_id, ""

            summary = self.generate_summary(section.title, content)
            return section.section_id, summary

        # 并行处理
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            results = list(executor.map(generate_for_section, level1_sections))

        for section_id, summary in results:
            summaries[section_id] = summary

        return summaries

    def generate_document_summary(
        self,
        sections: List[SectionNode],
        chunks: List[ChunkNode],
    ) -> str:
        """生成文档摘要（基于章节摘要合并）"""
        # 优先使用已生成的章节摘要
        section_summaries = []
        for section in sections:
            if section.level == 1 and section.summary:
                section_summaries.append(f"{section.title}: {section.summary}")

        if section_summaries:
            # 有章节摘要，合并生成文档摘要
            combined = "\n\n".join(section_summaries)
            return self._generate_from_summaries(combined)

        # 没有章节摘要，直接从 chunks 生成
        if chunks:
            # 取前 N 个 chunk 作为代表
            sample_chunks = sorted(chunks, key=lambda c: c.position)[:10]
            content = "\n".join(c.text for c in sample_chunks)
            return self.generate_summary("文档摘要", content)

        return ""

    def _generate_from_summaries(self, combined_summaries: str) -> str:
        """从章节摘要合并生成文档摘要"""
        prompt = f"""你是一个文档摘要专家。以下是文档各章节的摘要，请合并成一个简洁的文档摘要。
要求：
1. 保留核心主题和关键信息
2. 长度不超过{self.max_length}字
3. 使用与原文相同的语言

章节摘要：
{combined_summaries}

文档摘要："""

        result = self.llm.chat(
            prompt,
            max_tokens=4096,
            temperature=0.1,
            no_think=True,
        )

        if result:
            return result.strip()[: self.max_length]

        # 失败则截断
        return combined_summaries[: self.max_length]
