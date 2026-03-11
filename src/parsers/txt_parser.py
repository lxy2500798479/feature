"""
TXT 文档解析器 - 语义分块优化版

核心改进（对比上一版）：
===========================================================================
【问题1：分块策略改为语义边界切割】

旧方案：固定 512 字符切割，在句子中间截断，chunk 末尾是半句话
  → 向量语义不完整，LLM 拿到残句，回答质量差

新方案：SemanticChunker，严格按句子边界切割
  策略：
  1. 先将文本拆成完整句子（按句号/感叹号/问号分句）
  2. 将句子累积到 target_chars（默认 350 字）附近，在句子边界处切断
  3. overlap 保留上一 chunk 最后 N 个完整句子（而非固定字符数）
  4. 单句超长时退化为字符切割（兜底）

效果：
  - 每个 chunk 都是完整语义单元（完整句子）
  - LLM 收到的上下文不再有残句
  - 对小说、长段落文本尤其显著（斗罗大陆每章约 2000 字 → 5-6 个干净 chunk）

其他保留优化（上一版已有）：
  - 大文件阈值 2MB，流式逐行读取
  - 正则预编译到模块级别
  - 死循环修复（new_start 必须 > start）
  - 每 1000 行打进度日志
===========================================================================
"""
import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

from src.config import settings
from src.core.models import (
    DocumentMetadata,
    SectionNode,
    ChunkNode,
    GraphEdge,
    ParsedDocument,
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

# ── 模块级预编译正则（只编译一次）─────────────────────────────────────────

_HEADING_PATTERNS_COMPILED: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"^第([一二三四五六七八九十百千\d]+)[章节卷部篇集]"), 1),
    (re.compile(r"^(#{1,6})\s+(.+)$"), 6),
    (re.compile(r"^(\d+)\.\s+(.+)$"), 2),
    (re.compile(r"^(\d+\.\d+)\s+(.+)$"), 2),
    (re.compile(r"^(\d+\.\d+\.\d+)\s+(.+)$"), 2),
]

_HAS_HEADING_PATTERNS_COMPILED: List[re.Pattern] = [
    re.compile(r"^#{1,6}\s+"),
    re.compile(r"^第[一二三四五六七八九十百千\d]+[章节卷部篇集][\s　]"),
    re.compile(r"^第[一二三四五六七八九十百千\d]+章[\s　]"),
    re.compile(r"^第[一二三四五六七八九十]+[篇部集][\s　]"),
    re.compile(r"^\d+\.\d+\.\d+"),
    re.compile(r"^\d+\.\d+"),
    re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]"),
]

# 句子分隔符：匹配句末标点本身（含标点，用于 split 后重组）
# 注意：Python re lookbehind 不支持变长 pattern，所以直接匹配标点符号本身
# split 后奇数位是标点，偶数位是文本内容，重组时合并即可
_SENTENCE_SPLIT_RE = re.compile(r"([。！？…]+|[!?]+|\n)")



# ── 语义分块器 ─────────────────────────────────────────────────────────────

@dataclass
class SemanticChunkerConfig:
    """语义分块器配置"""
    target_chars: int = 350       # 目标 chunk 大小（字符数）
    max_chars: int = 600          # 单 chunk 最大字符数（硬上限）
    min_chars: int = 30           # 单 chunk 最小字符数（过滤噪声）
    overlap_sentences: int = 1    # overlap 保留上一 chunk 末尾 N 句
    hard_split_chars: int = 800   # 超长单句强制切割阈值


class SemanticChunker:
    """
    语义边界分块器

    核心思路：
    1. 用标点将文本拆成完整句子
    2. 按 target_chars 累积句子，在句子边界处切 chunk
    3. overlap 用"保留最后 N 句"而非固定字符数
    4. 超长单句（> hard_split_chars）才退化到字符切割
    """

    def __init__(self, cfg: Optional[SemanticChunkerConfig] = None):
        self.cfg = cfg or SemanticChunkerConfig()

    def split_sentences(self, text: str) -> List[str]:
        """将文本拆成句子列表，每句保留末尾标点"""
        if not text:
            return []

        # 按句末标点分割，保留分隔符
        parts = _SENTENCE_SPLIT_RE.split(text)

        sentences: List[str] = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if not part:
                i += 1
                continue

            # 若下一个 part 是分隔符，合并
            if i + 1 < len(parts) and _SENTENCE_SPLIT_RE.fullmatch(parts[i + 1] or ""):
                combined = part + parts[i + 1]
                i += 2
            else:
                combined = part
                i += 1

            combined = combined.strip()
            if combined:
                sentences.append(combined)

        return sentences

    def chunk(self, text: str, doc_id: str, section_id: str, base_position: int = 0) -> List[ChunkNode]:
        """
        将 section 内容切成语义 chunk

        Args:
            text: section 完整文本
            doc_id: 文档 ID
            section_id: section ID
            base_position: 在整个文档中的起始位置偏移

        Returns:
            ChunkNode 列表
        """
        if not text or not text.strip():
            return []

        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks: List[ChunkNode] = []
        position = 0
        char_cursor = 0  # 在 text 中的字符偏移

        # 用句子列表游标来做 overlap
        pending: List[str] = []          # 当前 chunk 待累积句子
        pending_chars = 0                 # 当前 chunk 累积字符数
        overlap_tail: List[str] = []     # 上一 chunk 末尾 N 句（用于 overlap）

        def _flush_chunk(sents: List[str]) -> Optional[ChunkNode]:
            nonlocal position, char_cursor
            if not sents:
                return None
            chunk_text = "".join(sents).strip()
            if len(chunk_text) < self.cfg.min_chars:
                return None

            # 计算字符偏移（在 text 中找到 chunk_text 的位置）
            # 使用简单的顺序查找，因为 chunk 是顺序生成的
            idx = text.find(chunk_text[:min(20, len(chunk_text))], char_cursor)
            if idx == -1:
                idx = char_cursor
            start_char = idx
            end_char = start_char + len(chunk_text)
            char_cursor = max(char_cursor, end_char - len("".join(overlap_tail)))

            node = ChunkNode(
                chunk_id=f"{section_id}_chunk_{position}",
                section_id=section_id,
                doc_id=doc_id,
                text=chunk_text,
                token_count=len(chunk_text),
                position=base_position + position,
                start_char=start_char,
                end_char=end_char,
            )
            position += 1
            return node

        for sent in sentences:
            sent_len = len(sent)

            # 超长单句：先 flush 当前，再强制字符切割该句
            if sent_len > self.cfg.hard_split_chars:
                if pending:
                    node = _flush_chunk(pending)
                    if node:
                        chunks.append(node)
                        overlap_tail = pending[-self.cfg.overlap_sentences:]
                    pending = list(overlap_tail)
                    pending_chars = sum(len(s) for s in pending)

                # 对超长句字符切割
                sub_start = 0
                while sub_start < sent_len:
                    sub_end = min(sub_start + self.cfg.max_chars, sent_len)
                    sub_text = sent[sub_start:sub_end].strip()
                    if sub_text and len(sub_text) >= self.cfg.min_chars:
                        node = ChunkNode(
                            chunk_id=f"{section_id}_chunk_{position}",
                            section_id=section_id,
                            doc_id=doc_id,
                            text=sub_text,
                            token_count=len(sub_text),
                            position=base_position + position,
                            start_char=char_cursor,
                            end_char=char_cursor + len(sub_text),
                        )
                        chunks.append(node)
                        position += 1
                    char_cursor += len(sub_text)
                    # 无 overlap（超长句切割的 sub chunk 不做 overlap）
                    sub_start = sub_end
                overlap_tail = []
                pending = []
                pending_chars = 0
                continue

            # 累积到 target_chars 附近
            if pending_chars + sent_len > self.cfg.max_chars and pending:
                # 超出硬上限，立刻 flush
                node = _flush_chunk(pending)
                if node:
                    chunks.append(node)
                    overlap_tail = pending[-self.cfg.overlap_sentences:]
                pending = list(overlap_tail)
                pending_chars = sum(len(s) for s in pending)

            pending.append(sent)
            pending_chars += sent_len

            # 达到目标大小，在句子边界切割
            if pending_chars >= self.cfg.target_chars:
                node = _flush_chunk(pending)
                if node:
                    chunks.append(node)
                    overlap_tail = pending[-self.cfg.overlap_sentences:]
                pending = list(overlap_tail)
                pending_chars = sum(len(s) for s in pending)

        # flush 剩余
        if pending:
            # 如果剩余内容太短，合并到上一个 chunk（避免碎片）
            if chunks and pending_chars < self.cfg.min_chars * 2:
                last = chunks[-1]
                extra = "".join(pending).strip()
                merged_text = last.text + extra
                chunks[-1] = ChunkNode(
                    chunk_id=last.chunk_id,
                    section_id=last.section_id,
                    doc_id=last.doc_id,
                    text=merged_text,
                    token_count=len(merged_text),
                    position=last.position,
                    start_char=last.start_char,
                    end_char=last.end_char + len(extra),
                )
            else:
                node = _flush_chunk(pending)
                if node:
                    chunks.append(node)

        return chunks


# ── TxtParser ──────────────────────────────────────────────────────────────

class TxtParser(BaseParser):
    """
    TXT 文档解析器

    分块策略：SemanticChunker（语义边界切割）
      - target_chars 从 config 或 settings 读取，默认 350
      - max_chars 默认 600（约为 target 的 1.7 倍，保留弹性）
      - overlap 保留上一 chunk 最后 1 句

    大文件（>2MB）：流式逐行读取，按章节标题分 section 后分块
    小文件（≤2MB）：一次性读入，走正则/AI大纲路径
    """

    def __init__(self, config: dict = None):
        super().__init__(config)

        # 语义分块参数（优先从 config 读，fallback 到 settings）
        target_chars = self.config.get(
            "semantic_target_chars",
            getattr(settings, "SEMANTIC_TARGET_CHARS", 350)
        )
        max_chars = self.config.get(
            "semantic_max_chars",
            getattr(settings, "SEMANTIC_MAX_CHARS", 600)
        )
        overlap_sentences = self.config.get(
            "semantic_overlap_sentences",
            getattr(settings, "SEMANTIC_OVERLAP_SENTENCES", 1)
        )
        min_chars = self.config.get(
            "semantic_min_chars",
            getattr(settings, "SEMANTIC_MIN_CHARS", 30)
        )

        self.chunker_cfg = SemanticChunkerConfig(
            target_chars=target_chars,
            max_chars=max_chars,
            overlap_sentences=overlap_sentences,
            min_chars=min_chars,
        )
        self.chunker = SemanticChunker(self.chunker_cfg)

        self.structure_mode = self.config.get("structure_mode", "auto")
        self.ai_outline_service = AIOutlineService()
        self.large_file_outline_service = LargeFileOutlineService()

        logger.info(
            f"TxtParser 初始化: SemanticChunker("
            f"target={target_chars}, max={max_chars}, "
            f"overlap_sentences={overlap_sentences})"
        )

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in [".txt", ".text"]

    # ── 主入口 ──────────────────────────────────────────────────────────────

    def parse(self, file_path: str) -> ParsedDocument:
        logger.info(f"TxtParser 解析: {file_path}")
        file_path_obj = Path(file_path)
        doc_id = self._generate_doc_id(file_path)

        file_size = file_path_obj.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"文件大小: {file_size_mb:.2f} MB")

        LARGE_FILE_THRESHOLD = 2 * 1024 * 1024  # 2MB

        if file_size > LARGE_FILE_THRESHOLD:
            logger.info(f"文件超过 2MB，使用流式处理")
            return self._parse_large_file(file_path, doc_id)

        logger.info("小文件，一次性读取解析")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        metadata = self._build_metadata(doc_id, file_path_obj)
        sections = self._parse_structure(content, doc_id)
        chunks, edges = self._build_chunks_and_edges(content, doc_id, sections)

        logger.info(
            f"✅ TXT 解析完成: {len(sections)} sections, {len(chunks)} chunks "
            f"| avg_chunk_chars={int(sum(len(c.text) for c in chunks)/max(len(chunks),1))}"
        )
        return ParsedDocument(metadata=metadata, sections=sections, chunks=chunks, edges=edges)

    # ── 元数据 ───────────────────────────────────────────────────────────────

    def _build_metadata(self, doc_id: str, file_path_obj: Path) -> DocumentMetadata:
        return DocumentMetadata(
            doc_id=doc_id,
            title=file_path_obj.stem,
            file_path=str(file_path_obj),
            file_type="txt",
        )

    def _generate_doc_id(self, file_path: str) -> str:
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"doc_{path_hash}"

    # ── 结构解析 ─────────────────────────────────────────────────────────────

    def _parse_structure(self, content: str, doc_id: str) -> List[SectionNode]:
        if self.structure_mode == "none":
            return [SectionNode(
                section_id=f"{doc_id}_sec_0",
                doc_id=doc_id,
                title="全文",
                level=1,
                hierarchy_path="0",
                content=content,
                order=0,
            )]

        if self._has_headings(content):
            logger.info("检测到章节标题，使用正则解析")
            return self._parse_headings(content, doc_id)

        logger.info("未检测到章节标题，调用 AI 生成大纲")
        return self._generate_ai_outline(content, doc_id)

    def _has_headings(self, content: str) -> bool:
        """扫前 100 个非空行，检测章节标题"""
        heading_count = 0
        scanned = 0
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            scanned += 1
            if scanned > 100:
                break
            for pattern in _HAS_HEADING_PATTERNS_COMPILED:
                if pattern.match(line):
                    heading_count += 1
                    break
        return heading_count >= 3

    def _parse_headings(self, content: str, doc_id: str) -> List[SectionNode]:
        """
        单次遍历识别标题并直接收集 section.content。
        不做二次全文切割。
        """
        sections: List[SectionNode] = []
        section_idx = 0
        current_title: Optional[str] = None
        current_level = 1
        current_lines: List[str] = []

        def _flush(title: str, level: int, lines: List[str]) -> None:
            nonlocal section_idx
            sec_content = "".join(lines).strip()
            if not sec_content and not title:
                return
            sections.append(SectionNode(
                section_id=f"{doc_id}_sec_{section_idx}",
                doc_id=doc_id,
                title=title or "无标题",
                level=level,
                hierarchy_path=str(section_idx),
                content=sec_content,
                order=section_idx,
                parent_section_id=None,
            ))
            section_idx += 1
            if section_idx % 100 == 0:
                logger.info(f"  正则解析进度: {section_idx} 章节...")

        for raw_line in content.splitlines(keepends=True):
            stripped = raw_line.strip()
            if not stripped:
                if current_title is not None:
                    current_lines.append(raw_line)
                continue

            matched_title: Optional[str] = None
            matched_level = 1
            for pattern, level in _HEADING_PATTERNS_COMPILED:
                m = pattern.match(stripped)
                if m:
                    if level == 1:
                        matched_title = (m.group(1) if m.lastindex else stripped)[:80]
                    elif level == 6:
                        matched_title = (m.group(2).strip() if m.lastindex >= 2 else m.group(1).strip())[:80]
                    else:
                        matched_title = (m.group(2).strip() if m.lastindex >= 2 else stripped)[:80]
                    matched_level = level
                    break

            if matched_title is not None:
                if current_title is not None:
                    _flush(current_title, current_level, current_lines)
                elif current_lines:
                    _flush("前言", 1, current_lines)
                current_title = matched_title
                current_level = matched_level
                current_lines = []
            else:
                current_lines.append(raw_line)

        # 最后一节
        if current_title is not None:
            _flush(current_title, current_level, current_lines)
        elif current_lines:
            _flush("全文", 1, current_lines)

        if not sections:
            sections.append(SectionNode(
                section_id=f"{doc_id}_sec_0",
                doc_id=doc_id,
                title="全文",
                level=1,
                hierarchy_path="0",
                content=content,
                order=0,
            ))

        logger.info(f"  正则解析完成: {len(sections)} 个章节")
        return sections

    def _generate_ai_outline(self, content: str, doc_id: str) -> List[SectionNode]:
        total_chars = len(content)
        logger.info(f"字符数: {total_chars}，调用 AI 生成大纲...")
        if total_chars > settings.OUTLINE_BATCH_CHARS:
            sections = self.large_file_outline_service.generate_outline_for_large_file(content, doc_id)
        else:
            sections = self.ai_outline_service.generate_outline(content, doc_id)
        if sections:
            sections = self._assign_content_to_sections(content, sections)
        return sections

    def _assign_content_to_sections(self, content: str, sections: List[SectionNode]) -> List[SectionNode]:
        """AI 大纲路径：按比例分配文本内容（兜底方案）"""
        level1 = [s for s in sections if s.level == 1]
        if not level1:
            return sections
        total_len = len(content)
        n = len(level1)
        for i, section in enumerate(level1):
            start_pos = int(i * total_len / n)
            end_pos = int((i + 1) * total_len / n) if i < n - 1 else total_len
            section.content = content[start_pos:end_pos]
        return sections

    # ── 分块 & 边 ─────────────────────────────────────────────────────────

    def _build_chunks_and_edges(
        self,
        content: str,
        doc_id: str,
        sections: List[SectionNode],
    ) -> Tuple[List[ChunkNode], List[GraphEdge]]:
        """
        对所有 section 调用 SemanticChunker 生成 chunks。
        同时构建 HAS_CHUNK 边。
        """
        if not sections:
            # fallback：整文件做一个 section
            fallback_section = SectionNode(
                section_id=f"{doc_id}_sec_0",
                doc_id=doc_id,
                title="全文",
                level=1,
                hierarchy_path="0",
                content=content,
                order=0,
            )
            sections = [fallback_section]

        all_chunks: List[ChunkNode] = []
        global_position = 0
        level1_sections = [s for s in sections if s.level == 1]

        for section in level1_sections:
            sec_content = section.content or ""

            # AI 大纲路径：content 可能为空，按比例截取兜底
            if not sec_content and content:
                n = len(level1_sections)
                i = section.order
                start_pos = int(i * len(content) / n)
                end_pos = int((i + 1) * len(content) / n) if i < n - 1 else len(content)
                sec_content = content[start_pos:end_pos]

            if not sec_content.strip():
                continue

            sec_chunks = self.chunker.chunk(
                text=sec_content,
                doc_id=doc_id,
                section_id=section.section_id,
                base_position=global_position,
            )

            if sec_chunks:
                all_chunks.extend(sec_chunks)
                global_position = sec_chunks[-1].position + 1

        if not all_chunks and content.strip():
            # 极端兜底：整文件直接分块
            fallback_sid = f"{doc_id}_sec_0"
            all_chunks = self.chunker.chunk(
                text=content,
                doc_id=doc_id,
                section_id=fallback_sid,
                base_position=0,
            )

        edges = [
            GraphEdge(src_id=c.section_id, dst_id=c.chunk_id, edge_type="HAS_CHUNK")
            for c in all_chunks
        ]
        return all_chunks, edges

    # ── 大文件流式路径 ────────────────────────────────────────────────────

    def _parse_large_file(self, file_path: str, doc_id: str) -> ParsedDocument:
        """
        流式处理大文件：
        - 单次顺序读取，边读边识别章节标题
        - 每当检测到新章节标题，立刻对已累积内容做语义分块
        - 内存占用恒定（单个最大章节大小），不随文件增长

        保留：每 1000 行打进度日志，杜绝假卡死
        """
        file_path_obj = Path(file_path)
        metadata = self._build_metadata(doc_id, file_path_obj)

        logger.info(f"开始流式处理: {file_path}")
        t_start = time.time()

        sections: List[SectionNode] = []
        chunks: List[ChunkNode] = []
        global_position = 0
        section_idx = 0
        current_title = "开篇"
        current_lines: List[str] = []
        current_line_count = 0
        total_line_count = 0

        def _flush_section(title: str, lines: List[str]) -> None:
            nonlocal global_position, section_idx

            sec_content = "".join(lines).strip()
            if not sec_content:
                return

            section_id = f"{doc_id}_sec_{section_idx}"
            sections.append(SectionNode(
                section_id=section_id,
                doc_id=doc_id,
                title=title,
                level=1,
                hierarchy_path=str(section_idx),
                content="",   # 流式路径：不在内存中保留 content，节省内存
                order=section_idx,
            ))

            # 语义分块
            sec_chunks = self.chunker.chunk(
                text=sec_content,
                doc_id=doc_id,
                section_id=section_id,
                base_position=global_position,
            )
            chunks.extend(sec_chunks)
            if sec_chunks:
                global_position = sec_chunks[-1].position + 1

            section_idx += 1

            if section_idx % 50 == 0:
                logger.info(
                    f"  流式分块进度: {section_idx} 章节 | "
                    f"{len(chunks)} chunks | 耗时 {time.time()-t_start:.1f}s"
                )

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_line_count += 1
                stripped = line.strip()

                # 每 1000 行打进度日志
                if total_line_count % 1000 == 0:
                    elapsed = time.time() - t_start
                    logger.info(
                        f"  流式进度: {total_line_count} 行 | "
                        f"{section_idx} 章节 | {len(chunks)} chunks | "
                        f"耗时 {elapsed:.1f}s"
                    )

                # 标题识别
                heading_title: Optional[str] = None
                for pattern, level in _HEADING_PATTERNS_COMPILED:
                    m = pattern.match(stripped)
                    if m:
                        if level == 1:
                            heading_title = (m.group(1) if m.lastindex else stripped)[:80]
                        elif level == 6:
                            heading_title = (m.group(2).strip() if m.lastindex >= 2 else m.group(1).strip())[:80]
                        else:
                            heading_title = (m.group(2).strip() if m.lastindex >= 2 else stripped)[:80]
                        break

                if heading_title and current_line_count > 10:
                    _flush_section(current_title, current_lines)
                    current_title = heading_title
                    current_lines = []
                    current_line_count = 0
                else:
                    current_lines.append(line)
                    current_line_count += 1

        # 处理最后一章节
        if current_lines:
            _flush_section(current_title, current_lines)

        # 未检测到章节标题 → 回退：二次读文件整体分块
        if not sections:
            logger.warning("未检测到章节标题，回退到整文件语义分块")
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content_full = f.read()
            section_id = f"{doc_id}_sec_0"
            sections.append(SectionNode(
                section_id=section_id,
                doc_id=doc_id,
                title="全文",
                level=1,
                hierarchy_path="0",
                content="",
                order=0,
            ))
            chunks = self.chunker.chunk(
                text=content_full,
                doc_id=doc_id,
                section_id=section_id,
                base_position=0,
            )

        edges = [
            GraphEdge(src_id=c.section_id, dst_id=c.chunk_id, edge_type="HAS_CHUNK")
            for c in chunks
        ]

        elapsed = time.time() - t_start
        avg_chars = int(sum(len(c.text) for c in chunks) / max(len(chunks), 1))
        logger.info(
            f"✅ 流式解析完成: {len(sections)} sections, {len(chunks)} chunks "
            f"| avg_chunk_chars={avg_chars} | 耗时 {elapsed:.1f}s"
        )
        return ParsedDocument(metadata=metadata, sections=sections, chunks=chunks, edges=edges)