"""
文档解析器模块
"""
from .base import BaseParser, ParserRegistry
from .minereu_parser import MinereuParser
from .docling_parser import DoclingParser
from .txt_parser import TxtParser, STRUCTURE_PRESETS
from src.utils.logger import logger

# 注册解析器（txt 优先以便 .txt 走 TxtParser）
ParserRegistry.register("txt", TxtParser)
ParserRegistry.register("minereu", MinereuParser)
ParserRegistry.register("docling", DoclingParser)

__all__ = [
    "BaseParser",
    "ParserRegistry",
    "MinereuParser",
    "DoclingParser",
    "TxtParser",
    "STRUCTURE_PRESETS",
]
