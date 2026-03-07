"""
Mineru 文档解析器 - 调用 Mineru API 解析 PDF/DOCX
"""
import json
import zipfile
import io
import shutil
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

# 项目根目录下的 mineruData 文件夹
MINERU_DATA_DIR = Path(__file__).parent.parent.parent / "mineruData"


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
        # 创建 mineruData 目录
        MINERU_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def supports(self, file_path: str) -> bool:
        """检查是否支持该文件类型"""
        ext = Path(file_path).suffix.lower()
        return ext in [".pdf", ".docx", ".doc"]

    def parse(self, file_path: str) -> ParsedDocument:
        """解析文档"""
        import time
        total_start = time.time()
        self.logger.info(f"Mineru 解析: {file_path}")

        # 1. 调用 Mineru API 返回压缩包
        extract_dir = self._call_mineru_api_zip(file_path)

        # 2. 从解压的文件夹中读取解析结果
        md_content = self._read_markdown(extract_dir)
        content_list = self._read_content_list(extract_dir)

        # 3. 提取元数据
        metadata = self._extract_metadata(file_path)

        # 4. 解析 Markdown 内容
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
            extract_dir=extract_dir
        )

    def _call_mineru_api_zip(self, file_path: str) -> str:
        """调用 Mineru API 并返回解压后的目录"""
        if not self.api_url:
            raise ValueError("MINERU_API_URL 未配置")

        # 使用代理服务的 /file_parse 接口，请求返回 ZIP
        url = f"{self.api_url}/file_parse"

        # 准备文件
        file_name = Path(file_path).name
        files = {
            "files": (file_name, open(file_path, "rb"), self._get_content_type(file_path))
        }

        # 准备表单数据 - 请求返回 ZIP 格式
        data = {
            "output_dir": "./output",
            "lang_list": ["ch"],
            "backend": "pipeline",
            "parse_method": "auto",
            "formula_enable": True,
            "table_enable": True,
            "return_md": True,
            "return_middle_json": False,
            "return_model_output": False,
            "return_content_list": True,
            "return_images": True,  # 返回图片
            "response_format_zip": True,  # 关键：请求返回 ZIP 格式
            "start_page_id": 0,
            "end_page_id": 99999,
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            self.logger.info(f"请求 Mineru API: {url}")
            response = requests.post(url, files=files, data=data, headers=headers, timeout=600)
            response.raise_for_status()

            # 检查返回的是 ZIP 还是 JSON
            content_type = response.headers.get("Content-Type", "")

            # 创建输出目录（以文档名为目录名）
            doc_name = Path(file_path).stem
            extract_dir = MINERU_DATA_DIR / doc_name

            # 如果目录已存在，先删除
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

            # 检查是否是 ZIP 格式
            if "zip" in content_type or response.content[:2] == b'PK':
                # 解压 ZIP 文件
                self.logger.info(f"解压 ZIP 到: {extract_dir}")
                with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                self.logger.info(f"Mineru 解析文件已保存到: {extract_dir}")
            else:
                # 返回的是 JSON，解析并保存
                self.logger.warning("API 返回 JSON 而非 ZIP，回退到 JSON 模式")
                result = response.json()
                md_content = result.get("markdown", "")

                # 保存 markdown 文件
                md_file = extract_dir / f"{doc_name}.md"
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)

                # 保存 content_list
                content_list = result.get("content_list", [])
                if content_list:
                    import json
                    cl_file = extract_dir / "content_list.json"
                    with open(cl_file, 'w', encoding='utf-8') as f:
                        json.dump(content_list, f, ensure_ascii=False)

                self.logger.info(f"Mineru 解析结果已保存到: {extract_dir}")

            return str(extract_dir)

        finally:
            files["files"][1].close()

    def _read_markdown(self, extract_dir: str) -> str:
        """从解压目录中读取 markdown 文件"""
        md_path = Path(extract_dir)

        # 递归查找 .md 文件
        md_files = list(md_path.rglob("*.md"))
        if md_files:
            with open(md_files[0], 'r', encoding='utf-8') as f:
                return f.read()

        self.logger.warning(f"未找到 markdown 文件: {extract_dir}")
        return ""

    def _read_content_list(self, extract_dir: str) -> List[Dict[str, Any]]:
        """从解压目录中读取 content_list.json"""
        extract_path = Path(extract_dir)

        # 递归查找 content_list.json
        for json_file in extract_path.rglob("content_list.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    self.logger.warning(f"JSON 解析失败: {json_file}")

        self.logger.warning(f"未找到 content_list.json: {extract_dir}")
        return []

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
        from src.chunking.text_chunker import ChunkerFactory

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
