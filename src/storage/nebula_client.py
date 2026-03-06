"""
NebulaGraph 客户端
"""
from contextlib import contextmanager
from typing import List, Dict, Optional, Any
import nebula3
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool
from src.config import settings
from src.utils.logger import logger


class NebulaQueryError(Exception):
    """NebulaGraph 查询异常"""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error


class NebulaClient:
    """NebulaGraph 客户端"""
    
    def __init__(self):
        self.host = settings.NEBULA_HOST
        self.port = settings.NEBULA_PORT
        self.user = settings.NEBULA_USER
        self.password = settings.NEBULA_PASSWORD
        self.space_name = settings.NEBULA_SPACE
        
        self.config = Config()
        self.config.max_connection_pool_size = 10
        
        self.connection_pool: Optional[ConnectionPool] = None
        self._initialized = False
    
    def connect(self):
        """连接到 NebulaGraph"""
        try:
            self.connection_pool = ConnectionPool()
            ok = self.connection_pool.init(
                [(self.host, self.port)],
                self.config
            )
            if not ok:
                raise ConnectionError("Failed to initialize connection pool")
            
            logger.info(f"已连接到 NebulaGraph: {self.host}:{self.port}")
            self._initialized = True
        
        except Exception as e:
            logger.error(f"连接 NebulaGraph 失败: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self.connection_pool:
            self.connection_pool.close()
            logger.info("NebulaGraph 连接已关闭")
    
    @contextmanager
    def get_session(self):
        """获取会话上下文管理器"""
        if not self._initialized:
            self.connect()

        try:
            session = self.connection_pool.get_session(self.user, self.password)
        except Exception as e:
            logger.warning(f"NebulaGraph 获取 session 失败，尝试重建连接池: {e}")
            try:
                self.connection_pool.close()
            except Exception:
                pass
            self._initialized = False
            self.connect()
            session = self.connection_pool.get_session(self.user, self.password)

        try:
            yield session
        finally:
            session.release()
    
    def init_schema(self):
        """初始化图空间和 Schema"""
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            
            # 创建 Tag 和 Edge 类型
            tags = [
                """CREATE TAG IF NOT EXISTS Document (
                    doc_id string,
                    title string,
                    author string,
                    created_date string,
                    file_path string,
                    file_type string,
                    page_count int,
                    summary string,
                    graph_ready bool DEFAULT false,
                    embeddings_ready bool DEFAULT false,
                    embedding_task_status string DEFAULT "pending",
                    embedding_error string DEFAULT ""
                );""",
                """CREATE TAG IF NOT EXISTS Section (
                    doc_id string,
                    title string,
                    level int,
                    hierarchy_path string,
                    content string,
                    `order` int
                );""",
                """CREATE TAG IF NOT EXISTS Chunk (
                    section_id string,
                    doc_id string,
                    text string,
                    token_count int,
                    position int,
                    start_char int,
                    end_char int,
                    embedding_id string
                );""",
                """CREATE TAG IF NOT EXISTS Concept (
                    phrase string,
                    freq int,
                    community int,
                    doc_id string
                );""",
                """CREATE EDGE IF NOT EXISTS HAS_SECTION();""",
                """CREATE EDGE IF NOT EXISTS HAS_CHUNK();""",
                """CREATE EDGE IF NOT EXISTS CONTAINS_CONCEPT();""",
                """CREATE EDGE IF NOT EXISTS COOCCURS_WITH(weight double, cooccur int);""",
            ]
            
            for tag_query in tags:
                result = session.execute(tag_query)
                if not result.is_succeeded():
                    logger.warning(f"Schema 创建警告: {result.error_msg()}")
            
            logger.info("Schema 初始化成功")
