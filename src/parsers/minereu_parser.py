"""
Mineru 文档解析器 - 调用 Mineru API 解析 PDF/DOCX
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests

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


class MinereuParser(BaseParser):
    """
    Mineru 文档解析器
    
    调用 Mineru API 进行文档解析，支持 PDF、DOCX 等格式。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.api_url = self.config.get("api_url", settings.MINERU_API_URL)
        self.backend = self.config.get("backend", settings.MINERU_BACKEND)
        self.api_key = self.config.get("api_key", settings.MINEREU_API_KEY)
        self.logger = logger
    
    def supports(self, file_path: str) -> bool:
        """检查是否支持该文件类型"""
        ext = Path(file_path).suffix.lower()
        return ext in [".pdf", ".docx", ".doc"]
    
    def parse(self, file_path: str) -> ParsedDocument:
        """解析文档"""
        import time
        total_start = time.time()
        self.logger.info(f"Mineru 解析: {file_path}")

        # 1. 调用 Mineru API
        mineru_result = self._call_mineru_api(file_path)

        # 从结果中提取 markdown 和 content_list
        md_content = mineru_result.get("markdown", "")
        content_list = mineru_result.get("content_list", [])

        # 2. 提取元数据
        metadata = self._extract_metadata(file_path)

        # 3. 解析 Markdown 内容
        sections, chunks, edges = self._parse_markdown(md_content, metadata.doc_id)

        total_time = time.time() - total_start
        self.logger.info(f"Mineru 解析完成: {time.time() - total_time:.2f}秒, sections={len(sections)}, chunks={len(chunks)}")

        # 保存 raw_content_list 供图片处理使用
        raw_content_list = self._parse_content_list(content_list)

        return ParsedDocument(
            metadata=metadata,
            sections=sections,
            chunks=chunks,
            edges=edges,
            raw_content_list=raw_content_list,
            extract_dir=str(Path(file_path).parent)
        )
    
    def _call_mineru_api(self, file_path: str) -> Dict[str, Any]:
        """调用 Mineru API - 使用代理服务的 /file_parse 接口"""
        if not self.api_url:
            raise ValueError("MINERU_API_URL 未配置")

        # 使用代理服务的 /file_parse 接口
        url = f"{self.api_url}/file_parse"

        # 准备文件
        files = {
            "files": (Path(file_path).name, open(file_path, "rb"), self._get_content_type(file_path))
        }

        # 准备表单数据 - 与远程服务参数匹配
        data = {
            "output_dir": "./output",
            "lang_list": ["ch"],
            "backend": "hybrid-auto-engine",
            "parse_method": "auto",
            "formula_enable": True,
            "table_enable": True,
            "return_md": True,
            "return_middle_json": False,
            "return_model_output": False,
            "return_content_list": True,  # 需要返回 content_list 以便处理图片
            "return_images": False,
            "response_format_zip": False,
            "start_page_id": 0,
            "end_page_id": 99999,
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(url, files=files, data=data, headers=headers, timeout=600)
            response.raise_for_status()
            result = response.json()

            # 检查返回的错误
            if result.get("status_code") and result.get("status_code") != 200:
                raise RuntimeError(f"Mineru API error: {result.get('message', result)}")

            return result

        finally:
            files["files"][1].close()

    def _get_content_type(self, file_path: str) -> str:
        """根据文件扩展名获取 Content-Type"""
        ext = Path(file_path).suffix.lower()
        content_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
        }
        return content_types.get(ext, "application/octet-stream")
    
    def _parse_markdown(self, md_content: str, doc_id: str) -> tuple:
        """解析 Markdown 内容为 sections, chunks, edges"""
        import re
        
        # 按标题分割 sections
        lines = md_content.split("\n")
        sections = []
        chunks = []
        edges = []
        
        current_section = None
        current_section_content = []
        section_order = 0
        
        for line in lines:
            # 检测标题
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                # 保存之前的 section
                if current_section is not None:
                    section_chunks = self._create_chunks(current_section["section_id"], current_section_content, doc_id)
                    chunks.extend(section_chunks)
                
                # 创建新 section
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                section_id = f"{doc_id}_sec_{section_order}"
                
                current_section = {
                    "section_id": section_id,
                    "title": title,
                    "level": level,
                    "content": ""
                }
                current_section_content = []
                section_order += 1
            
            elif current_section is not None:
                current_section_content.append(line)
                current_section["content"] += line + "\n"
        
        # 保存最后一个 section
        if current_section is not None:
            section_chunks = self._create_chunks(current_section["section_id"], current_section_content, doc_id)
            chunks.extend(section_chunks)
        
        # 构建 sections 列表
        section_idx = 0
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
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
                
                # 添加 Document -> Section 边
                edges.append(GraphEdge(
                    edge_type=EdgeType.HAS_SECTION.value,
                    src_id=doc_id,
                    dst_id=section_id,
                    properties={"order": section_idx}
                ))
                
                section_idx += 1
        
        # 如果没有 section，创建一个默认的
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
        
        # 添加 Section -> Chunk 边
        for chunk in chunks:
            edges.append(GraphEdge(
                edge_type=EdgeType.CONTAINS_CHUNK.value,
                src_id=chunk.section_id,
                dst_id=chunk.chunk_id,
                properties={"order": chunk.position}
            ))
        
        # 添加 NEXT_CHUNK 边
        for i in range(len(chunks) - 1):
            edges.append(GraphEdge(
                edge_type=EdgeType.NEXT_CHUNK.value,
                src_id=chunks[i].chunk_id,
                dst_id=chunks[i + 1].chunk_id,
                properties={"distance": 1}
            ))
        
        return sections, chunks, edges
    
    def _create_chunks(self, section_id: str, lines: List[str], doc_id: str) -> List[ChunkNode]:
        """为 section 创建 chunks"""
        from src.chunking.text_chunker import ChunkerFactory, FixedSizeChunker
        
        # 按配置的分块策略
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
    
    def _parse_content_list(self, content_list: Any) -> List[Dict[str, Any]]:
        """解析 content_list（供图片处理使用）"""
        if isinstance(content_list, list):
            return content_list
        return []
