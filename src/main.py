"""
FastAPI 应用入口
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.api.routes import router
# 仅在本地嵌入模式下注册 embedding 路由
# from src.api.embedding_routes import router as embedding_router
from src.storage.nebula_client import NebulaClient
from src.storage.vector_client import MilvusClient
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 防止重复初始化（uvicorn reload 模式会启动多个进程）
    if hasattr(app.state, "_initialized") and app.state._initialized:
        yield
        return

    app.state._initialized = True

    # 启动时
    logger.info("正在启动 FinalRAG Phase 1...")

    # 初始化 NebulaGraph Schema
    try:
        nebula_client = NebulaClient()
        nebula_client.connect()

        if settings.RESET_GRAPH_DB:
            logger.warning("⚠️  RESET_GRAPH_DB=True，正在重置 NebulaGraph...")
            nebula_client.reset_space()
        else:
            nebula_client.init_schema()
            logger.info("NebulaGraph schema 已初始化")

        nebula_client.close()
    except Exception as e:
        logger.error(f"NebulaGraph 初始化失败: {e}")

    # 重置 Milvus（如果配置）
    if settings.RESET_VECTOR_DB:
        try:
            logger.warning("⚠️  RESET_VECTOR_DB=True，正在重置 Milvus collection...")
            milvus_client = MilvusClient()
            milvus_client.reset_collection()
        except Exception as e:
            logger.error(f"Milvus 重置失败（不阻止启动）: {e}")

    yield

    # 关闭时
    logger.info("正在关闭 FinalRAG Phase 1...")


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FinalRAG Phase 1: Meta-KG Construction",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api/v1")
# app.include_router(embedding_router, prefix="/api/v1")  # 仅本地模式需要


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "phase": "Phase 1: Meta-KG Construction",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
