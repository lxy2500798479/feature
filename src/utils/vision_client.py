"""
视觉模型客户端 - 使用 qwen2.5-vl-7b
"""
from typing import Optional, Dict, Any, List
from enum import Enum
import base64
import requests

from src.config import settings
from src.utils.logger import logger


class ImageGraphType(Enum):
    """图片类型枚举 - 用于知识图谱"""
    ARCHITECTURE = "architecture"      # 架构图
    FLOWCHART = "flowchart"           # 流程图
    UML = "uml"                        # UML图/类图
    MINDMAP = "mindmap"               # 思维导图
    ORG_CHART = "org_chart"           # 组织架构图
    ENTITY_RELATION = "entity_relation"  # 实体关系图
    TIMELINE = "timeline"              # 时间线
    COMPARISON = "comparison"          # 对比图
    SCREENSHOT = "screenshot"          # 界面截图
    CHART = "chart"                   # 统计图表
    PHOTO = "photo"                   # 照片
    DECORATION = "decoration"         # 装饰性图片（图标、插图等）
    UNKNOWN = "unknown"               # 未知类型


# 配置：哪些类型需要加入知识图谱
IMAGE_TYPES_FOR_GRAPH = {
    ImageGraphType.ARCHITECTURE,
    ImageGraphType.FLOWCHART,
    ImageGraphType.UML,
    ImageGraphType.MINDMAP,
    ImageGraphType.ORG_CHART,
    ImageGraphType.ENTITY_RELATION,
    ImageGraphType.TIMELINE,
    ImageGraphType.COMPARISON,
}


class VisionClient:
    """视觉模型客户端"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_url = api_url or getattr(settings, "VISION_API_URL", None)
        self.api_key = api_key or getattr(settings, "VISION_API_KEY", None)
        self.model = model or getattr(settings, "VISION_MODEL", "qwen2.5-vl-7b")

    def describe_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """描述图像 - 支持 HTTP URL 或本地文件路径"""
        if not self.api_url:
            logger.warning("VISION_API_URL 未配置")
            return ""

        if not prompt:
            prompt = "请描述这张图片的内容"

        # 检测图片来源：HTTP URL 或本地文件
        if image_path.startswith("http://") or image_path.startswith("https://"):
            image_url = image_path
        else:
            # 本地文件：转换为 base64
            image_url = self._convert_to_base64(image_path)
            if not image_url:
                return ""

        # 调用视觉模型 API - 使用 image_url 格式
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            "max_tokens": 512
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]

            return ""

        except Exception as e:
            logger.error(f"图像描述失败: {e}")
            return ""

    def _convert_to_base64(self, image_path: str) -> str:
        """将本地图片转换为 base64 格式"""
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # 根据文件扩展名判断 MIME 类型
            ext = image_path.lower().split(".")[-1]
            mime_type = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "gif": "image/gif",
                "webp": "image/webp"
            }.get(ext, "image/jpeg")

            return f"data:{mime_type};base64,{image_data}"

        except Exception as e:
            logger.error(f"图片 base64 转换失败: {e}")
            return ""

    def describe_images_batch(self, image_paths: List[str], prompt: Optional[str] = None) -> List[str]:
        """批量描述图像"""
        results = []
        for path in image_paths:
            result = self.describe_image(path, prompt)
            results.append(result)
        return results

    def classify_image_type(self, image_path: str) -> Dict[str, Any]:
        """
        识别图片类型，并判断是否需要加入知识图谱
        
        Returns:
            {
                "type": ImageGraphType,  # 图片类型
                "need_graph": bool,       # 是否需要加入图谱
                "description": str,       # 图片描述
                "entities": List[str],    # 提取的实体
                "relations": List[Dict]  # 提取的关系 [(实体1, 关系, 实体2)]
            }
        """
        type_prompt = """请分析这张图片的类型和内容。

图片类型分类：
- architecture: 架构图（系统架构、软件架构、网络架构等）
- flowchart: 流程图（业务流程、工作流程、算法流程等）
- uml: UML图/类图（类之间的关系、对象结构等）
- mindmap: 思维导图
- org_chart: 组织架构图（人员结构、部门层级等）
- entity_relation: 实体关系图（E-R图、数据库Schema等）
- timeline: 时间线
- comparison: 对比图（表格、对比分析等）
- screenshot: 界面截图
- chart: 统计图表（柱状图、饼图、折线图等）
- photo: 照片
- decoration: 装饰性图片（图标、插图、背景图等）
- unknown: 未知类型

请按以下JSON格式返回：
```json
{
    "type": "类型名称",
    "need_graph": true/false,
    "description": "简要描述图片内容（不超过100字）",
    "entities": ["实体1", "实体2", ...],
    "relations": [["实体1", "关系名称", "实体2"], ...]
}
```

只返回JSON，不要其他内容。"""

        result_text = self.describe_image(image_path, type_prompt)
        
        # 解析JSON结果
        try:
            import json
            # 尝试提取JSON部分
            json_match = result_text.strip()
            if "```json" in json_match:
                json_match = json_match.split("```json")[1].split("```")[0]
            elif "```" in json_match:
                json_match = json_match.split("```")[1].split("```")[0]
            
            data = json.loads(json_match.strip())
            
            # 转换为枚举类型
            img_type = ImageGraphType.UNKNOWN
            type_str = data.get("type", "unknown").lower()
            for et in ImageGraphType:
                if et.value == type_str:
                    img_type = et
                    break
            
            return {
                "type": img_type,
                "need_graph": data.get("need_graph", img_type in IMAGE_TYPES_FOR_GRAPH),
                "description": data.get("description", ""),
                "entities": data.get("entities", []),
                "relations": data.get("relations", [])
            }
        except Exception as e:
            logger.warning(f"图片类型解析失败: {e}, 使用默认分类")
            return {
                "type": ImageGraphType.UNKNOWN,
                "need_graph": False,
                "description": result_text[:100] if result_text else "",
                "entities": [],
                "relations": []
            }
