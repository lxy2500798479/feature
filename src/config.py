"""
配置管理模块
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """全局配置"""
    
    # 应用配置
    APP_NAME: str = "FinalRAG"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # NebulaGraph配置
    NEBULA_HOST: str = "127.0.0.1"
    NEBULA_PORT: int = 9669
    NEBULA_USER: str = "root"
    NEBULA_PASSWORD: str = "nebula"
    NEBULA_SPACE: str = "finalrag"
    
    # 文档解析配置
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_SECTION_LEVEL: int = 5
    
    # 分块策略配置
    CHUNKING_STRATEGY: str = "fixed"
    CHUNKING_UNIT: str = "char"
    
    # Mineru API配置
    MINERU_API_URL: Optional[str] = None
    MINERU_BACKEND: str = "gpu"
    MINEREU_API_KEY: Optional[str] = None
    
    # Docling配置
    DOCLING_ENABLED: bool = True
    
    # 向量嵌入配置
    EMBEDDING_SERVICE_URL: Optional[str] = None
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MODE: str = "remote"
    
    # 异步向量嵌入配置
    ASYNC_EMBEDDING_ENABLED: bool = True
    EMBEDDING_TASK_TIMEOUT: int = 1800
    EMBEDDING_TASK_RETRY: int = 3
    
    # 向量数据库配置
    VECTOR_DB_TYPE: str = "milvus"
    MILVUS_HOST: str = "192.168.10.10"
    MILVUS_PORT: int = 30493
    
    # 摘要生成配置
    SUMMARY_ENABLED: bool = True
    SUMMARY_MODEL: str = "qwen2.5-14b"
    SUMMARY_MAX_LENGTH: int = 200
    SUMMARY_CONCURRENCY: int = 3
    SUMMARY_TIMEOUT: int = 600
    SUMMARY_API_URL: Optional[str] = None
    SUMMARY_API_KEY: Optional[str] = None
    
    # LLM文本模型配置
    LLM_API_URL: str = "https://new-api-dev.10rig.com:8443/v1/chat/completions"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    
    # 概念图谱配置
    CONCEPT_GRAPH_N_PROCESS: Optional[int] = None
    CONCEPT_GRAPH_COOCCUR_WORKERS: Optional[int] = None
    CONCEPT_GRAPH_MAX_EDGES_PER_NODE: int = 20

    # 视觉模型配置 (qwen2.5-vl-7b)
    VISION_API_URL: str = "https://new-api-dev.10rig.com:8443/v1/chat/completions"
    VISION_API_KEY: Optional[str] = None
    VISION_MODEL: str = "qwen2.5-vl-7b"

    # 图片处理配置
    IMAGE_PROCESSING_ENABLED: bool = True
    IMAGE_MAX_CONCURRENCY: int = 10
    IMAGE_MIN_HEIGHT: int = 100  # 最小高度阈值，低于此值的图片视为图标过滤掉
    IMAGE_PAGE_MIN_TEXT_LENGTH: int = 50  # 页面最小文字数量，低于此值的页面图片全部过滤

    # 数据库重置配置
    RESET_GRAPH_DB: bool = False
    RESET_VECTOR_DB: bool = False

    # 上传配置
    UPLOAD_DIR: Path = PROJECT_ROOT / "uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    class Config:
        env_file = ENV_FILE
        env_file_encoding = "utf-8"


settings = Settings()
