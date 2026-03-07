"""
文档解析器基类
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
from src.core.models import ParsedDocument
class BaseParser(ABC):
    """解析器基类"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """解析文件"""
        pass

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """检查是否支持该文件类型"""
        pass

    def _extract_metadata(self, file_path: str):
        """提取文档元数据"""
        from datetime import datetime
        from src.core.models import DocumentMetadata
        import hashlib

        path = Path(file_path)
        doc_id = hashlib.md5(str(path.absolute()).encode()).hexdigest()

        return DocumentMetadata(
            doc_id=doc_id,
            title=path.stem,
            file_path=str(path.absolute()),
            file_type=path.suffix.lower()
        )


class ParserRegistry:
    """解析器注册表"""
    
    _parsers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, parser_class: type):
        cls._parsers[name] = parser_class
        logger.info(f"Registered parser: {name}")
    
    @classmethod
    def get_parser(cls, name: str, config: Optional[Dict] = None) -> BaseParser:
        if name not in cls._parsers:
            raise ValueError(f"Parser '{name}' not found")
        return cls._parsers[name](config)
    
    @classmethod
    def get_suitable_parser(cls, file_path: str, config: Optional[Dict] = None) -> Optional[BaseParser]:
        for name, parser_class in cls._parsers.items():
            parser = parser_class(config)
            if parser.supports(file_path):
                return parser
        return None


from src.utils.logger import logger
