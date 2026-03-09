"""
AI Outline Service - 为无结构文档生成二级大纲（Section 结构）
"""
import json
import re
from typing import List, Optional

from src.config import settings
from src.core.models import SectionNode
from src.utils.llm_client import LLMClient
from src.utils.logger import logger


_OUTLINE_PROMPT = """你是一个文档结构分析专家。请分析以下文档内容，生成一个便于检索的章节摘要大纲。

要求：
1. 大纲标题应该是**描述性的内容摘要**，而不是"第一章"这种编号标题
2. 每个标题要用一句话概括该章节的核心内容，方便用户快速了解文档结构
3. 二级大纲：小节用简短的关键词或短语，描述该小节的具体内容
4. 标题示例：
   - 好的例子："穿越的唐三"、"武魂觉醒仪式"、"史莱克七怪集结"
   - 不好的例子："第一章"、"第1节"、"斗罗大陆异界唐三"
5. 一级章节数量控制在 8-20 个之间，根据文档长度适当调整
6. 大纲应该覆盖文档的主要内容
7. 直接输出 JSON 格式，不要有其他内容

输出格式：
[
  {{"title": "一级章节摘要标题", "subsections": ["小节1", "小节2"]}},
  ...
]

文档内容预览：
{content}

请生成大纲："""


_OUTLINE_SYSTEM_PROMPT = """你是一个文档结构分析专家，擅长为文档生成**描述性的内容摘要大纲**，而非传统编号章节标题。

请生成便于检索的大纲：
- 标题应该用简短的一句话概括章节核心内容
- 例如："穿越的唐三"、"武魂觉醒仪式" 而不是 "第一章"
- 直接输出 JSON 格式的大纲，不要有其他内容。"""


class AIOutlineService:
    """AI 生成文档大纲服务"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient(
            api_url=settings.SUMMARY_API_URL or settings.LLM_API_URL,
            api_key=settings.SUMMARY_API_KEY or settings.LLM_API_KEY,
            model=settings.SUMMARY_MODEL,  # 用 summary 模型
            timeout=settings.SUMMARY_TIMEOUT,
        )
        self.max_sections = settings.OUTLINE_MAX_SECTIONS or 15
        self.min_sections = settings.OUTLINE_MIN_SECTIONS or 5

    def generate_outline(
        self,
        content: str,
        doc_id: str,
        level: int = 1,
        parent_section_id: Optional[str] = None,
    ) -> List[SectionNode]:
        """为文档内容生成二级大纲结构

        Args:
            content: 文档内容
            doc_id: 文档 ID
            level: 当前层级（1 或 2）
            parent_section_id: 父 Section ID（用于二级章节）

        Returns:
            SectionNode 列表
        """
        # 截取内容（避免 token 过多）
        max_chars = settings.OUTLINE_MAX_CHARS or 8000
        truncated_content = content[:max_chars]

        prompt = _OUTLINE_PROMPT.format(content=truncated_content)

        logger.info(f"开始为文档 {doc_id} 生成大纲（层级 {level}）...")

        result = self.llm.chat(
            prompt,
            system_prompt=_OUTLINE_SYSTEM_PROMPT,
            max_tokens=4096,
            temperature=0.3,
            no_think=True,
        )

        if not result:
            logger.warning(f"文档 {doc_id} 大纲生成失败，使用默认结构")
            return self._create_default_section(content, doc_id, level, parent_section_id)

        sections = self._parse_outline(result, doc_id, level, parent_section_id)

        if not sections:
            logger.warning(f"文档 {doc_id} 大纲解析失败，使用默认结构")
            return self._create_default_section(content, doc_id, level, parent_section_id)

        logger.info(f"文档 {doc_id} 大纲生成成功，共 {len(sections)} 个章节")
        return sections

    def _parse_outline(
        self,
        outline_text: str,
        doc_id: str,
        level: int,
        parent_section_id: Optional[str] = None,
    ) -> List[SectionNode]:
        """解析 LLM 返回的大纲文本"""
        sections = []

        # 尝试提取 JSON
        try:
            # 去除可能的 markdown 代码块标记
            clean_text = re.sub(r"^```json", "", outline_text)
            clean_text = re.sub(r"^```", "", clean_text)
            clean_text = re.sub(r"```$", "", clean_text)
            clean_text = clean_text.strip()

            outline_data = json.loads(clean_text)

            if not isinstance(outline_data, list):
                logger.warning("大纲格式错误：不是数组")
                return []

            order = 0
            for idx, item in enumerate(outline_data):
                if not isinstance(item, dict):
                    continue

                title = item.get("title", "").strip()
                if not title:
                    continue

                section_id = f"{doc_id}_sec_{idx}"
                hierarchy_path = str(idx) if level == 1 else f"{parent_section_id.split('_')[-1]}.{idx}"

                section = SectionNode(
                    section_id=section_id,
                    doc_id=doc_id,
                    title=title,
                    level=level,
                    hierarchy_path=hierarchy_path,
                    content="",
                    order=order,
                    parent_section_id=parent_section_id,
                )
                sections.append(section)
                order += 1

                # 处理二级章节
                if level == 1 and "subsections" in item:
                    subsections = item["subsections"]
                    if isinstance(subsections, list):
                        for sub_idx, sub_title in enumerate(subsections):
                            if not isinstance(sub_title, str):
                                continue
                            sub_title = sub_title.strip()
                            if not sub_title:
                                continue

                            sub_section_id = f"{doc_id}_sec_{idx}_{sub_idx}"
                            sub_hierarchy_path = f"{idx}.{sub_idx}"

                            sub_section = SectionNode(
                                section_id=sub_section_id,
                                doc_id=doc_id,
                                title=sub_title,
                                level=2,
                                hierarchy_path=sub_hierarchy_path,
                                content="",
                                order=sub_idx,
                                parent_section_id=section_id,
                            )
                            sections.append(sub_section)

            return sections

        except json.JSONDecodeError as e:
            logger.warning(f"大纲 JSON 解析失败: {e}, 原始内容: {outline_text[:200]}")
            return []

    def _create_default_section(
        self,
        content: str,
        doc_id: str,
        level: int,
        parent_section_id: Optional[str] = None,
    ) -> List[SectionNode]:
        """创建默认章节结构（当 AI 生成失败时使用）"""
        sections = []

        if level == 1:
            # 默认一个大章节包含全文
            section = SectionNode(
                section_id=f"{doc_id}_sec_0",
                doc_id=doc_id,
                title="全文",
                level=1,
                hierarchy_path="0",
                content=content[:10000],  # 截取部分内容
                order=0,
                parent_section_id=None,
            )
            sections.append(section)

        return sections


class LargeFileOutlineService:
    """大文件分批生成大纲服务

    用于处理超长文档，将其分批处理后合并大纲。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.outline_service = AIOutlineService(llm_client)
        self.batch_size = settings.OUTLINE_BATCH_CHARS or 50000  # 每次处理 5 万字

    def generate_outline_for_large_file(
        self,
        content: str,
        doc_id: str,
    ) -> List[SectionNode]:
        """为大型文档生成分层大纲

        处理流程：
        1. 将文档分成多个批次
        2. 每批次生成局部大纲
        3. 合并并去重

        Args:
            content: 完整文档内容
            doc_id: 文档 ID

        Returns:
            合并后的 SectionNode 列表
        """
        total_chars = len(content)
        logger.info(f"文档 {doc_id} 总字符数: {total_chars}，分批处理...")

        if total_chars <= self.batch_size:
            # 小文件直接处理
            return self.outline_service.generate_outline(content, doc_id)

        # 分批处理
        all_sections = []
        batch_count = (total_chars + self.batch_size - 1) // self.batch_size

        for batch_idx in range(batch_count):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, total_chars)
            batch_content = content[start:end]

            logger.info(
                f"处理批次 {batch_idx + 1}/{batch_count}: 字符 {start}-{end}"
            )

            # 每批次生成一级大纲
            batch_sections = self.outline_service.generate_outline(
                batch_content, doc_id, level=1
            )

            # 调整 section_id 避免冲突
            for section in batch_sections:
                section.section_id = f"{section.section_id}_batch{batch_idx}"

            all_sections.extend(batch_sections)

        # 合并大纲：按 order 排序后返回
        all_sections.sort(key=lambda s: (s.order, s.hierarchy_path))

        logger.info(
            f"文档 {doc_id} 大纲生成分批完成，共 {len(all_sections)} 个章节"
        )
        return all_sections
