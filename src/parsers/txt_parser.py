"""
TXT 文档解析器
"""
import hashlib
import re
import uuid
from pathlib import Path
from typing import List, Tuple

from src.config import settings
from src.core.models import (
    DocumentMetadata,
    SectionNode,
    ChunkNode,
    GraphEdge,
    ParsedDocument
)
from src.parsers.base import BaseParser
from src.services.outline_service import AIOutlineService, LargeFileOutlineService
from src.utils.logger import logger


STRUCTURE_PRESETS = {
    "none": {"enabled": False},
    "heading": {
        "enabled": True,
        "patterns": [
            r"^第[一二三四五六七八九十百千\d]+[章节卷部篇集]",
            r"^\d+\.\d+\.\d+",
            r"^\d+\.\d+",
            r"^\d+",
        ]
    },
}


class TxtParser(BaseParser):
    """TXT 文档解析器"""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", settings.CHUNK_SIZE)
        self.chunk_overlap = self.config.get("chunk_overlap", settings.CHUNK_OVERLAP)
        self.structure_mode = self.config.get("structure_mode", "auto")

        self.structure_config = STRUCTURE_PRESETS.get(self.structure_mode, STRUCTURE_PRESETS["none"])

        # AI 大纲服务（用于无结构文档）
        self.ai_outline_service = AIOutlineService()
        self.large_file_outline_service = LargeFileOutlineService()
    
    def supports(self, file_path: str) -> bool:
        """检查是否支持该文件"""
        ext = Path(file_path).suffix.lower()
        return ext in [".txt", ".text"]
    
    def parse(self, file_path: str) -> ParsedDocument:
        """解析 TXT 文件（优化大文件内存占用）"""
        logger.info(f"TxtParser 解析: {file_path}")
        
        file_path_obj = Path(file_path)
        
        # 生成 doc_id
        doc_id = self._generate_doc_id(file_path)
        
        # 检查文件大小
        file_size = file_path_obj.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"文件大小: {file_size_mb:.2f} MB")
        
        # 大文件阈值：50MB
        LARGE_FILE_THRESHOLD = 50 * 1024 * 1024
        
        if file_size > LARGE_FILE_THRESHOLD:
            logger.info(f"检测到大文件 ({file_size_mb:.2f} MB)，使用流式处理")
            return self._parse_large_file(file_path, doc_id)
        
        # 小文件：正常处理
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 生成文档元数据
        metadata = DocumentMetadata(
            doc_id=doc_id,
            title=file_path_obj.stem,
            file_path=str(file_path),
            file_type="txt",
        )
        
        # 解析章节结构
        sections = self._parse_structure(content, doc_id)
        
        # 分块（按 section 分块）
        chunks = self._chunk_content_by_sections(content, doc_id, sections)
        
        # 生成边关系
        edges = self._build_edges(sections, chunks)
        
        logger.info(f"✅ TXT 解析完成: {len(sections)} sections, {len(chunks)} chunks")
        
        return ParsedDocument(
            metadata=metadata,
            sections=sections,
            chunks=chunks,
            edges=edges,
        )
    
    def _generate_doc_id(self, file_path: str) -> str:
        """生成文档 ID"""
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"doc_{path_hash}"
    
    def _parse_structure(self, content: str, doc_id: str) -> List[SectionNode]:
        """解析文档结构（章节）

        策略：
        1. auto 模式：自动检测是否有章节标题
           - 有标题：解析标题层级
           - 无标题：调用 AI 生成二级大纲
        2. none 模式：创建一个默认章节（全文）
        3. heading 模式：用正则匹配章节标题
        """
        if self.structure_mode == "none":
            # 无结构模式：创建一个默认章节
            return [
                SectionNode(
                    section_id=f"{doc_id}_section_0",
                    doc_id=doc_id,
                    title="全文",
                    level=1,
                    hierarchy_path="1",
                    content=content,
                    order=0,
                )
            ]

        # 检测是否有章节标题
        if self._has_headings(content):
            # 有标题：用正则解析
            logger.info(f"文档 {doc_id} 检测到章节标题，使用正则解析")
            return self._parse_headings(content, doc_id)

        # 无标题：调用 AI 生成二级大纲
        logger.info(f"文档 {doc_id} 未检测到章节标题，调用 AI 生成大纲")
        return self._generate_ai_outline(content, doc_id)

    def _has_headings(self, content: str) -> bool:
        """检测文档是否包含章节标题"""
        import re

        # 章节标题 patterns
        heading_patterns = [
            r"^#{1,6}\s+",  # Markdown 标题
            r"^第[一二三四五六七八九十百千\d]+[章节卷部篇集]\s",  # 第一章、第一节
            r"^第[一二三四五六七八九十百千\d]+章\s",  # 第X章
            r"^第[一二三四五六七八九十]+[篇部集]\s",  # 第X篇/部/集
            r"^第[一二三四五六七八九十]+[^\n]{1,10}\n",  # 第X天、第X部分（后面是换行）
            r"^\d+\.\d+\.\d+",  # 1.1.1
            r"^\d+\.\d+",  # 1.1
            r"^[①②③④⑤⑥⑦⑧⑨⑩]",  # 序号
        ]

        lines = content.split("\n")
        heading_count = 0

        for line in lines[:100]:  # 只检查前 100 行
            line = line.strip()
            if not line:
                continue

            for pattern in heading_patterns:
                if re.match(pattern, line):
                    heading_count += 1
                    break

        # 有超过 3 个标题才算有结构
        return heading_count >= 3

    def _parse_headings(self, content: str, doc_id: str) -> List[SectionNode]:
        """用正则解析章节标题"""
        import re

        lines = content.split("\n")
        sections = []
        section_idx = 0

        # 章节标题 patterns（优先级从高到低）
        heading_patterns = [
            (r"^第([一二三四五六七八九十百千\d]+)[章节卷部篇集]", 1),  # 第一章
            (r"^#{1,6}\s+(.+)$", 6),  # Markdown 标题
            (r"^(\d+)\.\s+(.+)$", 2),  # 1. 标题
            (r"^(\d+\.\d+)\s+(.+)$", 2),  # 1.1 标题
            (r"^(\d+\.\d+\.\d+)\s+(.+)$", 2),  # 1.1.1 标题
        ]

        current_section = None
        current_level = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue

            matched = False
            for pattern, level in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    # 提取标题
                    if level == 1:
                        title = match.group(1) if match.lastindex else line
                    elif level == 6:
                        title = match.group(1).strip()
                    else:
                        title = match.group(2).strip() if match.lastindex >= 2 else line

                    section_id = f"{doc_id}_sec_{section_idx}"
                    hierarchy_path = str(section_idx)

                    section = SectionNode(
                        section_id=section_id,
                        doc_id=doc_id,
                        title=title,
                        level=level,
                        hierarchy_path=hierarchy_path,
                        content="",
                        order=section_idx,
                        parent_section_id=None,
                    )

                    sections.append(section)
                    section_idx += 1
                    current_section = section
                    current_level = level
                    matched = True
                    break

        # 如果没解析到任何章节，创建默认
        if not sections:
            sections.append(
                SectionNode(
                    section_id=f"{doc_id}_sec_0",
                    doc_id=doc_id,
                    title="全文",
                    level=1,
                    hierarchy_path="0",
                    content=content,
                    order=0,
                )
            )

        return sections

    def _generate_ai_outline(self, content: str, doc_id: str) -> List[SectionNode]:
        """调用 AI 生成二级大纲"""
        import re

        total_chars = len(content)
        logger.info(f"文档 {doc_id} 字符数: {total_chars}，调用 AI 生成大纲...")

        # 判断文件大小，选择处理方式
        if total_chars > settings.OUTLINE_BATCH_CHARS:
            # 大文件：分批处理
            sections = self.large_file_outline_service.generate_outline_for_large_file(
                content, doc_id
            )
        else:
            # 小文件：直接生成
            sections = self.ai_outline_service.generate_outline(content, doc_id)

        # 为每个一级 section 设置对应的内容范围
        if sections:
            sections = self._assign_content_to_sections(content, sections)

        return sections

    def _assign_content_to_sections(
        self, content: str, sections: List[SectionNode]
    ) -> List[SectionNode]:
        """为每个 Section 分配对应的内容（用于后续摘要生成）"""
        if not sections:
            return sections

        # 找出所有一级 section（level=1）
        level1_sections = [s for s in sections if s.level == 1]

        if not level1_sections:
            return sections

        # 计算每个 section 的内容范围
        total_len = len(content)

        for i, section in enumerate(level1_sections):
            start_pos = int(i * total_len / len(level1_sections))
            next_start = (
                int((i + 1) * total_len / len(level1_sections))
                if i < len(level1_sections) - 1
                else total_len
            )

            section.content = content[start_pos:next_start]

        return sections
    
    def _chunk_content(
        self,
        content: str,
        doc_id: str,
        section_id: str = None
    ) -> List[ChunkNode]:
        """将内容分块"""
        if not section_id:
            section_id = f"{doc_id}_section_0"

        chunks = []
        start = 0
        position = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))

            # 尝试在句子边界切分
            if end < len(content):
                for sep in ["。", "！", "？", "，", "。", "\n"]:
                    last_sep = content.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break

            text = content[start:end]
            if not text.strip():
                start = end
                continue

            chunk_id = f"{section_id}_chunk_{position}" if section_id else f"{doc_id}_chunk_{position}"

            chunk = ChunkNode(
                chunk_id=chunk_id,
                section_id=section_id,
                doc_id=doc_id,
                text=text,
                token_count=len(text),
                position=position,
                start_char=start,
                end_char=end,
            )
            chunks.append(chunk)

            start = end - self.chunk_overlap
            position += 1

        return chunks

    def _chunk_content_by_sections(
        self, content: str, doc_id: str, sections: List[SectionNode]
    ) -> List[ChunkNode]:
        """按 section 分块

        为每个 section 的内容分别进行分块，确保 chunk 正确关联到 section。
        """
        if not sections:
            return self._chunk_content(content, doc_id, None)

        all_chunks = []
        global_position = 0

        # 找出所有一级 section（level=1）
        level1_sections = [s for s in sections if s.level == 1]

        for section in level1_sections:
            # 使用 section.content（如果已分配）或按范围计算
            section_content = section.content if section.content else ""

            if not section_content:
                # 如果 section.content 为空，按位置分配内容
                start_pos = int(section.order * len(content) / max(len(level1_sections), 1))
                next_pos = (
                    int((section.order + 1) * len(content) / max(len(level1_sections), 1))
                    if section.order < len(level1_sections) - 1
                    else len(content)
                )
                section_content = content[start_pos:next_pos]

            if not section_content.strip():
                continue

            # 为该 section 分块
            section_chunks = self._chunk_content(
                section_content, doc_id, section.section_id
            )

            # 调整 chunk 的位置和字符范围
            base_position = global_position
            for chunk in section_chunks:
                chunk.position = base_position + chunk.position
                chunk.start_char = base_position + chunk.start_char
                chunk.end_char = base_position + chunk.end_char

            all_chunks.extend(section_chunks)
            global_position = (
                section_chunks[-1].position + 1 if section_chunks else global_position
            )

        # 如果没有一级 section，退回到默认分块
        if not all_chunks:
            return self._chunk_content(content, doc_id, sections[0].section_id)

        return all_chunks
    
    def _build_edges(self, sections: List[SectionNode], chunks: List[ChunkNode]) -> List[GraphEdge]:
        """构建边关系"""
        edges = []
        
        # Section -> Chunk 关系
        for chunk in chunks:
            edges.append(
                GraphEdge(
                    src_id=chunk.section_id,
                    dst_id=chunk.chunk_id,
                    edge_type="HAS_CHUNK",
                )
            )
        
        return edges

    def _parse_large_file(self, file_path: str, doc_id: str) -> ParsedDocument:
        """流式处理大文件（优化：单次顺序读取，边读边分块）"""
        file_path_obj = Path(file_path)

        metadata = DocumentMetadata(
            doc_id=doc_id,
            title=file_path_obj.stem,
            file_path=str(file_path),
            file_type="txt",
        )

        logger.info(f"开始流式处理大文件: {file_path}")

        # 预编译章节标题正则
        _HEADING_PATTERNS = [
            (re.compile(r"^第([一二三四五六七八九十百千\d]+)[章节卷部篇集]"), 1),
            (re.compile(r"^#{1,6}\s+(.+)$"), 6),
            (re.compile(r"^(\d+)\.\s+(.+)$"), 2),
        ]

        sections = []
        chunks = []
        global_position = 0
        section_idx = 0

        # 当前章节状态
        current_section_title = "开篇"
        current_section_content = []  # 使用列表累积
        current_section_lines = 0
        line_count = 0

        # 单次顺序读取：逐行处理，边读边分块
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_count += 1
                line_stripped = line.strip()

                # 每 10000 行打印进度
                if line_count % 10000 == 0:
                    logger.info(f"  读取进度: {line_count} 行, {section_idx} 章节, {len(chunks)} chunks")

                # 检测章节标题
                is_heading = False
                heading_title = None
                for pattern, level in _HEADING_PATTERNS:
                    match = pattern.match(line_stripped)
                    if match:
                        if level == 1:
                            heading_title = match.group(1) if match.lastindex else line_stripped[:20]
                        elif level == 6:
                            heading_title = match.group(1).strip()[:50]
                        else:
                            heading_title = (match.group(2).strip() if match.lastindex >= 2 else line_stripped)[:50]
                        is_heading = True
                        break

                # 遇到新章节且当前章节有足够内容
                if is_heading and heading_title and current_section_lines > 10:
                    # 处理上一章节
                    section_content = "".join(current_section_content)
                    if section_content.strip():
                        section_id = f"{doc_id}_sec_{section_idx}"
                        section = SectionNode(
                            section_id=section_id,
                            doc_id=doc_id,
                            title=current_section_title,
                            level=1,
                            hierarchy_path=str(section_idx),
                            content="",
                            order=section_idx,
                        )
                        sections.append(section)

                        # 分块
                        section_chunks = self._chunk_content(
                            section_content, doc_id, section_id
                        )
                        for chunk in section_chunks:
                            chunk.position = global_position
                            global_position += 1
                        chunks.extend(section_chunks)

                        section_idx += 1

                    # 开始新章节
                    current_section_title = heading_title
                    current_section_content = []
                    current_section_lines = 0
                else:
                    # 累积章节内容
                    current_section_content.append(line)
                    current_section_lines += 1

        # 处理最后一个章节
        section_content = "".join(current_section_content)
        if section_content.strip():
            section_id = f"{doc_id}_sec_{section_idx}"
            section = SectionNode(
                section_id=section_id,
                doc_id=doc_id,
                title=current_section_title,
                level=1,
                hierarchy_path=str(section_idx),
                content="",
                order=section_idx,
            )
            sections.append(section)

            section_chunks = self._chunk_content(
                section_content, doc_id, section_id
            )
            for chunk in section_chunks:
                chunk.position = global_position
                global_position += 1
            chunks.extend(section_chunks)

        # 如果没有检测到章节，创建默认章节
        if not sections:
            logger.warning("未检测到章节标题，回退到简单分块")
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            section_id = f"{doc_id}_sec_0"
            section = SectionNode(
                section_id=section_id,
                doc_id=doc_id,
                title="全文",
                level=1,
                hierarchy_path="0",
                content="",
                order=0,
            )
            sections.append(section)
            chunks = self._chunk_content(content, doc_id, section_id)
            for i, chunk in enumerate(chunks):
                chunk.position = i

        edges = self._build_edges(sections, chunks)
        logger.info(f"✅ 大文件流式解析完成: {len(sections)} sections, {len(chunks)} chunks")

        return ParsedDocument(
            metadata=metadata,
            sections=sections,
            chunks=chunks,
            edges=edges,
        )

