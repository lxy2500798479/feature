"""
TXT 文档解析器
"""
import hashlib
import uuid
from pathlib import Path
from typing import List
from src.core.models import (
    DocumentMetadata,
    SectionNode,
    ChunkNode,
    GraphEdge,
    ParsedDocument
)
from src.parsers.base import BaseParser
from src.config import settings
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
        self.structure_mode = self.config.get("structure_mode", "none")
        
        self.structure_config = STRUCTURE_PRESETS.get(self.structure_mode, STRUCTURE_PRESETS["none"])
    
    def supports(self, file_path: str) -> bool:
        """检查是否支持该文件"""
        ext = Path(file_path).suffix.lower()
        return ext in [".txt", ".text"]
    
    def parse(self, file_path: str) -> ParsedDocument:
        """解析 TXT 文件"""
        logger.info(f"TxtParser 解析: {file_path}")
        
        file_path_obj = Path(file_path)
        
        # 生成 doc_id
        doc_id = self._generate_doc_id(file_path)
        
        # 读取内容
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
        
        # 分块
        chunks = self._chunk_content(content, doc_id, sections[0].section_id if sections else None)
        
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
        """解析文档结构（章节）"""
        if not self.structure_config.get("enabled", False):
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
        
        # 简单章节识别
        import re
        lines = content.split("\n")
        sections = []
        section_id = f"{doc_id}_section_0"
        
        # 默认章节
        sections.append(
            SectionNode(
                section_id=section_id,
                doc_id=doc_id,
                title="全文",
                level=1,
                hierarchy_path="1",
                content=content,
                order=0,
            )
        )
        
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
            
            chunk_id = f"{doc_id}_chunk_{position}"
            
            chunk = ChunkNode(
                chunk_id=chunk_id,
                section_id=section_id,
                doc_id=doc_id,
                text=text,
                token_count=len(text),  # 简化：使用字符数作为 token 估计
                position=position,
                start_char=start,
                end_char=end,
            )
            chunks.append(chunk)
            
            start = end - self.chunk_overlap
            position += 1
        
        return chunks
    
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
