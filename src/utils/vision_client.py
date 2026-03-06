"""
视觉模型客户端 - 使用 qwen2.5-vl-7b
"""
from typing import Optional, Dict, Any, List
import base64
import requests

from src.config import settings
from src.utils.logger import logger


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
