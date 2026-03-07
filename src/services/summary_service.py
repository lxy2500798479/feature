"""
Summary Service - 摘要生成服务
"""
import time
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
    
    def generate_summary(self, title: str, content: str) -> str:
        """为单个 Section 生成摘要"""
        if not content.strip():
            return ""
        
        prompt = _SUMMARY_PROMPT.format(
            max_length=self.max_length,
            title=title,
            content=content[:4000],
        )
        
        # no_think=True: 禁用 qwen3 思考模式，减少 token 消耗，直接输出摘要
        # max_tokens 设大一些（4096）避免推理链截断
        result = self.llm.chat(
            prompt,
            max_tokens=4096,
            temperature=0.1,
            no_think=True,
        )
        
        if result:
            return result.strip()[:self.max_length]
        
        return content.strip()[:200]
    
    def generate_summaries_for_sections(
        self,
        sections: List[SectionNode],
        chunks: List[ChunkNode],
    ) -> Dict[str, str]:
        """为多个 Section 批量生成摘要"""
        section_chunks = defaultdict(list)
        for chunk in chunks:
            section_chunks[chunk.section_id].append(chunk)
        
        summaries = {}
        for section in sections:
            section_chunk_list = section_chunks.get(section.section_id, [])
            
            if not section_chunk_list:
                summaries[section.section_id] = ""
                continue
            
            section_chunks_sorted = sorted(section_chunk_list, key=lambda c: c.position)
            content = "\n".join(c.text for c in section_chunks_sorted)
            
            summary = self.generate_summary(section.title, content)
            summaries[section.section_id] = summary
        
        return summaries
