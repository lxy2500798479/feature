"""
Image Service - 图片处理服务
"""
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import os

from src.config import settings
from src.utils.vision_client import VisionClient
from src.utils.logger import logger
from src.core.models import ChunkNode


# 过滤配置（从 settings 读取）
IMAGE_MIN_HEIGHT = getattr(settings, 'IMAGE_MIN_HEIGHT', 100)  # 最小高度阈值，低于此值的图片视为图标过滤掉
PAGE_MIN_TEXT_LENGTH = getattr(settings, 'IMAGE_PAGE_MIN_TEXT_LENGTH', 50)  # 页面最小文字数量


class ImageService:
    """图片处理服务"""

    def __init__(self):
        self.vision_client = VisionClient()
        self.enabled = getattr(settings, 'IMAGE_PROCESSING_ENABLED', True)
        self.max_concurrency = getattr(settings, 'IMAGE_MAX_CONCURRENCY', 10)

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

        # 1. 计算每页文字数量
        page_text_count = self._calculate_page_text_count(content_list)

        # 2. 分析图片信息，确定需要处理的图片
        images_to_process = self._filter_images(content_list, page_text_count, extract_dir)
        logger.info(f"图片过滤后待处理: {len(images_to_process)}/{len([i for i in content_list if i.get('type') == 'image'])}")

        if not images_to_process:
            return {"image_chunks": [], "table_chunks": []}

        # 3. 并发处理图片
        image_chunks = asyncio.run(
            self._process_images_concurrent(images_to_process, doc_id, page_text_map)
        )

        # 4. 处理表格（暂不同步）
        table_chunks = []
        for item in content_list:
            if item.get("type") == "table":
                tbl_chunk = self._process_table(item, doc_id)
                if tbl_chunk:
                    table_chunks.append(tbl_chunk)

        return {
            "image_chunks": image_chunks,
            "table_chunks": table_chunks
        }

    def _calculate_page_text_count(self, content_list: List[Dict[str, Any]]) -> Dict[int, int]:
        """计算每页的文字数量"""
        page_text_count = {}
        for item in content_list:
            if item.get("type") == "text":
                text = item.get("text", "")
                page = item.get("page_idx", 0)
                page_text_count[page] = page_text_count.get(page, 0) + len(text)
        return page_text_count

    def _filter_images(
        self,
        content_list: List[Dict[str, Any]],
        page_text_count: Dict[int, int],
        extract_dir: str
    ) -> List[Dict[str, Any]]:
        """智能过滤图片"""
        filtered_images = []
        img_dir = Path(extract_dir) / "images"

        # 获取最后一页页码
        max_page = max(page_text_count.keys()) if page_text_count else 0

        for item in content_list:
            if item.get("type") != "image":
                continue

            page_idx = item.get("page_idx", 0)
            bbox = item.get("bbox", [])

            # 条件1: 检查图片尺寸（高度）
            if bbox and len(bbox) == 4:
                height = bbox[3] - bbox[1]
                if height < IMAGE_MIN_HEIGHT:
                    logger.debug(f"过滤小图片(图标): 页{page_idx}, 高度{height}px")
                    continue

            # 条件2: 检查页面文字数量（章节标题页）
            text_count = page_text_count.get(page_idx, 0)
            if text_count < PAGE_MIN_TEXT_LENGTH:
                logger.debug(f"过滤章节标题页图片: 页{page_idx}, 文字{text_count}字")
                continue

            # 条件3: 检查最后一页（结束页）
            if page_idx == max_page:
                # 检查最后一页文字是否极少（可能是"谢谢观看"之类的）
                last_page_text = ""
                for cl_item in content_list:
                    if cl_item.get("page_idx") == max_page and cl_item.get("type") == "text":
                        last_page_text += cl_item.get("text", "")

                # 如果最后一页文字少于30字，认为是结束页
                if len(last_page_text.strip()) < 30:
                    logger.debug(f"过滤结束页图片: 页{page_idx}, 文字\"{last_page_text[:20]}...\"")
                    continue

            # 条件4: 检查图片文件是否存在
            img_path = item.get("img_path", "")
            if img_path:
                full_img_path = img_dir / img_path if not Path(img_path).is_absolute() else Path(img_path)
                if not full_img_path.exists():
                    logger.warning(f"图片文件不存在: {full_img_path}")
                    continue

                filtered_images.append({
                    **item,
                    "full_path": str(full_img_path)
                })

        return filtered_images

    async def _process_images_concurrent(
        self,
        images: List[Dict[str, Any]],
        doc_id: str,
        page_text_map: Dict[int, str]
    ) -> List[ChunkNode]:
        """并发处理多张图片"""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process_one(img_item: Dict) -> Optional[ChunkNode]:
            async with semaphore:
                return await asyncio.to_thread(
                    self._process_image, img_item, doc_id, page_text_map
                )

        results = await asyncio.gather(*[process_one(img) for img in images])
        return [r for r in results if r is not None]

    def _process_image(self, item: Dict, doc_id: str, page_text_map: Dict[int, str]) -> Optional[ChunkNode]:
        """处理单张图片"""
        img_path = item.get("full_path", item.get("img_path", ""))
        page_idx = item.get("page_idx", 0)

        if not img_path or not os.path.exists(img_path):
            return None

        # 1. 先识别图片类型
        type_info = self.vision_client.classify_image_type(img_path)
        
        # 2. 如果需要生成详细描述，再调用视觉模型
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
            end_char=len(description),
            # 扩展字段：图片类型和图谱信息
            metadata={
                "image_type": type_info.get("type").value if type_info.get("type") else "unknown",
                "need_graph": type_info.get("need_graph", False),
                "graph_entities": type_info.get("entities", []),
                "graph_relations": type_info.get("relations", []),
                "description": type_info.get("description", description[:100])
            }
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
