"""
Docling 文档解析器 - 本地文档解析
"""
from pathlib import Path
from typing import Optional, Dict, List

from src.config import settings
from src.core.models import (
    ParsedDocument,
    DocumentMetadata,
    SectionNode,
    ChunkNode,
    GraphEdge,
    EdgeType,
)
from .base import BaseParser
from src.utils.logger import logger


class DoclingParser(BaseParser):
    """
    Docling 文档解析器
    
    使用 Docling 库进行本地文档解析。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.logger = logger
    
    def supports(self, file_path: str) -> bool:
        """检查是否支持该文件类型"""
        ext = Path(file_path).suffix.lower()
        return ext in [".pdf", ".docx", ".doc", ".pptx", ".xlsx"]
    
    def parse(self, file_path: str) -> ParsedDocument:
        """解析文档"""
        import time
        total_start = time.time()
        self.logger.info(f"Docling 解析: {file_path}")
        
        # 使用 Docling 解析
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        
        # 转换为 Markdown
        md_content = result.document.export_to_markdown()
        
        # 提取元数据
        metadata = self._extract_metadata(file_path)
        
        # 解析 Markdown
        sections, chunks, edges = self._parse_markdown(md_content, metadata.doc_id)
        
        total_time = time.time() - total_start
        self.logger.info(f"Docling 解析完成: {total_time:.2f}秒")
        
        return ParsedDocument(
            metadata=metadata,
            sections=sections,
            chunks=chunks,
            edges=edges
        )
    
    def _parse_markdown(self, md_content: str, doc_id: str) -> tuple:
        """解析 Markdown 内容"""
        import re
        
        lines = md_content.split("\n")
        sections = []
        chunks = []
        edges = []
        section_idx = 0
        
        current_section_content = []
        
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                if current_section_content:
                    section_id = f"{doc_id}_sec_{section_idx}"
                    section_chunks = self._create_chunks(section_id, current_section_content, doc_id)
                    chunks.extend(section_chunks)
                    section_idx += 1
                    current_section_content = []
                
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                section_id = f"{doc_id}_sec_{section_idx}"
                
                sections.append(SectionNode(
                    section_id=section_id,
                    doc_id=doc_id,
                    title=title,
                    level=level,
                    hierarchy_path=str(section_idx),
                    content="",
                    order=section_idx,
                    parent_section_id=None
                ))
                
                edges.append(GraphEdge(
                    edge_type=EdgeType.HAS_SECTION,
                    from_id=doc_id,
                    to_id=section_id,
                    properties={"order": section_idx}
                ))
                
                section_idx += 1
            else:
                current_section_content.append(line)
        
        if current_section_content:
            section_id = f"{doc_id}_sec_{section_idx}"
            section_chunks = self._create_chunks(section_id, current_section_content, doc_id)
            chunks.extend(section_chunks)
        
        if not sections:
            sections.append(SectionNode(
                section_id=f"{doc_id}_sec_0",
                doc_id=doc_id,
                title="全文",
                level=1,
                hierarchy_path="0",
                content=md_content,
                order=0,
                parent_section_id=None
            ))
        
        for i, chunk in enumerate(chunks):
            edges.append(GraphEdge(
                edge_type=EdgeType.CONTAINS_CHUNK,
                from_id=chunk.section_id,
                to_id=chunk.chunk_id,
                properties={"order": chunk.position}
            ))
            
            if i > 0:
                edges.append(GraphEdge(
                    edge_type=EdgeType.NEXT_CHUNK,
                    from_id=chunks[i-1].chunk_id,
                    to_id=chunk.chunk_id,
                    properties={"distance": 1}
                ))
        
        return sections, chunks, edges
    
    def _create_chunks(self, section_id: str, lines: List[str], doc_id: str) -> List[ChunkNode]:
        """创建 chunks"""
        from src.chunking.text_chunker import ChunkerFactory
        
        chunker = ChunkerFactory.create(
            settings.CHUNKING_STRATEGY,
            {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "unit": settings.CHUNKING_UNIT
            }
        )
        
        text = "\n".join(lines)
        text_chunks = chunker.chunk(text)
        
        chunks = []
        for idx, tc in enumerate(text_chunks):
            chunks.append(ChunkNode(
                chunk_id=f"{section_id}_chunk_{idx}",
                section_id=section_id,
                doc_id=doc_id,
                text=tc.text,
                token_count=tc.token_count,
                position=idx,
                start_char=tc.start_char,
                end_char=tc.end_char
            ))
        
        return chunks
