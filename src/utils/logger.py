"""
日志工具模块
"""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "feature") -> logging.Logger:
    """设置日志器"""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # 控制台处理器
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    
    # 格式化
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console.setFormatter(formatter)
    
    logger.addHandler(console)
    return logger


logger = setup_logger()
