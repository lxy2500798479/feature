"""
Image Service - 图片处理服务
"""
from typing import Dict, List, Any, Optional
from pathlib import Path
import os

from src.config import settings
from src.utils.vision_client import VisionClient
from src.utils.logger import logger
from src.core.models import ChunkNode


class ImageService:
    """图片处理服务"""
    
    def __init__(self):
        self.vision_client = VisionClient()
        self.enabled = getattr(settings, 'IMAGE_PROCESSING_ENABLED', True)
    
    def process_images(
        self,
        content_list: List[Dict[str, Any]],
        doc_id: str,
        extract_dir: str,
        page_text_map: Dict[int, str]
    ) -> Dict[str, List[ChunkNode]]:
        """处理文档中的图片"""
        if not self.enabled:
            logger.info("图片处理已禁用")
            return {"image_chunks": [], "table_chunks": []}
        
        image_chunks = []
        table_chunks = []
        
        for item in content_list:
            item_type = item.get("type", "")
            
            if item_type == "image":
                img_chunk = self._process_image(item, doc_id, page_text_map)
                if img_chunk:
                    image_chunks.append(img_chunk)
            
            elif item_type == "table":
                tbl_chunk = self._process_table(item, doc_id)
                if tbl_chunk:
                    table_chunks.append(tbl_chunk)
        
        return {
            "image_chunks": image_chunks,
            "table_chunks": table_chunks
        }
    
    def _process_image(self, item: Dict, doc_id: str, page_text_map: Dict[int, str]) -> Optional[ChunkNode]:
        """处理单张图片"""
        img_path = item.get("img_path", "")
        page_idx = item.get("page_idx", 0)
        
        if not img_path or not os.path.exists(img_path):
            return None
        
        description = self.vision_client.describe_image(img_path)
        
        if not description:
            return None
        
        context = page_text_map.get(page_idx, "")
        
        chunk_id = f"img_{doc_id}_{hash(img_path) % 100000}"
        
        return ChunkNode(
            chunk_id=chunk_id,
            section_id=f"{doc_id}_sec_0",
            doc_id=doc_id,
            text=f"[图片描述] {description}\n\n[页面上下文] {context[:500]}",
            token_count=len(description.split()) + len(context[:500].split()),
            position=0,
            start_char=0,
            end_char=len(description)
        )
    
    def _process_table(self, item: Dict, doc_id: str) -> Optional[ChunkNode]:
        """处理表格"""
        table_html = item.get("html", "")
        
        if not table_html:
            return None
        
        import re
        text = re.sub(r'<[^>]+>', ' ', table_html)
        text = ' '.join(text.split())
        
        chunk_id = f"tbl_{doc_id}_{hash(table_html) % 100000}"
        
        return ChunkNode(
            chunk_id=chunk_id,
            section_id=f"{doc_id}_sec_0",
            doc_id=doc_id,
            text=f"[表格内容] {text[:1000]}",
            token_count=len(text[:1000].split()),
            position=0,
            start_char=0,
            end_char=len(text)
        )
