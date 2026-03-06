"""
代理服务：MinerU + Embedding
- /file_parse: 转发到 MinerU 服务
- /embeddings: 文本向量化服务
"""
import os
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger

# 配置
MINERU_SERVICE_URL = os.getenv("MINERU_SERVICE_URL", "http://localhost:8001")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
MODEL_CACHE_DIR = os.getenv("HF_HOME", "./models")

app = FastAPI(
    title="MinerU + Embedding Proxy Service",
    description="代理服务：PDF解析 + 文本向量化",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
embedding_model = None


def load_embedding_model():
    """加载 Embedding 模型"""
    global embedding_model
    
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            cache_folder=MODEL_CACHE_DIR
        )
        
        logger.info(f"✅ Embedding model loaded (dim: {embedding_model.get_sentence_embedding_dimension()})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    logger.info("=" * 70)
    logger.info("🚀 Starting Proxy Service")
    logger.info("=" * 70)
    
    # 加载 Embedding 模型
    embedding_ok = load_embedding_model()
    
    logger.info(f"MinerU Service: {MINERU_SERVICE_URL}")
    logger.info(f"Embedding Model: {'✅ Ready' if embedding_ok else '❌ Failed'}")
    logger.info("=" * 70)
    logger.info("✨ Service ready!")


# ==================== MinerU 代理接口 ====================

@app.post("/file_parse")
async def file_parse_proxy(
    files: List[UploadFile] = File(...),
    output_dir: str = Form("./output"),
    lang_list: List[str] = Form(["ch"]),
    backend: str = Form("hybrid-auto-engine"),
    parse_method: str = Form("auto"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    server_url: Optional[str] = Form(None),
    return_md: bool = Form(True),
    return_middle_json: bool = Form(False),
    return_model_output: bool = Form(False),
    return_content_list: bool = Form(False),
    return_images: bool = Form(False),
    response_format_zip: bool = Form(False),
    start_page_id: int = Form(0),
    end_page_id: int = Form(99999),
):
    """
    代理 MinerU 的 file_parse 接口
    转发所有参数到 MinerU 服务
    """
    try:
        # 准备文件和表单数据
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append(
                ("files", (file.filename, content, file.content_type))
            )
        
        # 准备表单数据
        form_data = {
            "output_dir": output_dir,
            "backend": backend,
            "parse_method": parse_method,
            "formula_enable": formula_enable,
            "table_enable": table_enable,
            "return_md": return_md,
            "return_middle_json": return_middle_json,
            "return_model_output": return_model_output,
            "return_content_list": return_content_list,
            "return_images": return_images,
            "response_format_zip": response_format_zip,
            "start_page_id": start_page_id,
            "end_page_id": end_page_id,
        }
        
        # 添加 lang_list（多个值）
        for lang in lang_list:
            files_data.append(("lang_list", (None, lang)))
        
        # 添加 server_url（如果有）
        if server_url:
            form_data["server_url"] = server_url
        
        # 转发请求到 MinerU 服务
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{MINERU_SERVICE_URL}/file_parse",
                files=files_data,
                data=form_data,
            )
        
        # 返回响应
        if response_format_zip:
            # 返回 ZIP 文件
            return Response(
                content=response.content,
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=results.zip"}
            )
        else:
            # 返回 JSON
            return JSONResponse(
                status_code=response.status_code,
                content=response.json()
            )
    
    except httpx.TimeoutException:
        raise HTTPException(504, "MinerU service timeout")
    except httpx.ConnectError:
        raise HTTPException(503, f"Cannot connect to MinerU service at {MINERU_SERVICE_URL}")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, f"Proxy error: {str(e)}")


# ==================== Embedding 接口 ====================

class EmbeddingRequest(BaseModel):
    """Embedding 请求"""
    texts: List[str]
    normalize: bool = True
    batch_size: Optional[int] = 32


class EmbeddingResponse(BaseModel):
    """Embedding 响应"""
    embeddings: List[List[float]]
    dimension: int
    count: int
    model: str


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    生成文本嵌入向量
    
    Args:
        request: 包含文本列表的请求
    
    Returns:
        嵌入向量列表
    """
    if embedding_model is None:
        raise HTTPException(503, "Embedding model not loaded")
    
    if not request.texts:
        raise HTTPException(400, "texts cannot be empty")
    
    try:
        # 生成 embeddings
        embeddings = embedding_model.encode(
            request.texts,
            batch_size=request.batch_size,
            normalize_embeddings=request.normalize,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            dimension=embeddings.shape[1],
            count=len(embeddings),
            model=EMBEDDING_MODEL_NAME
        )
    
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, f"Embedding failed: {str(e)}")


# ==================== 健康检查 ====================

@app.get("/health")
async def health_check():
    """健康检查"""
    # 检查 MinerU 服务
    mineru_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{MINERU_SERVICE_URL}/health")
            if response.status_code == 200:
                mineru_status = "healthy"
            else:
                mineru_status = "unhealthy"
    except:
        mineru_status = "unreachable"
    
    return {
        "status": "healthy",
        "services": {
            "mineru": {
                "status": mineru_status,
                "url": MINERU_SERVICE_URL
            },
            "embedding": {
                "status": "ready" if embedding_model else "not_loaded",
                "model": EMBEDDING_MODEL_NAME,
                "dimension": (
                    embedding_model.get_sentence_embedding_dimension() 
                    if embedding_model else None
                )
            }
        }
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "MinerU + Embedding Proxy Service",
        "version": "1.0.0",
        "endpoints": {
            "file_parse": "POST /file_parse - PDF解析（代理到MinerU）",
            "embeddings": "POST /embeddings - 文本向量化",
            "health": "GET /health - 健康检查",
            "docs": "GET /docs - API文档"
        },
        "mineru_service": MINERU_SERVICE_URL
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"\n🌐 Starting Proxy Service on http://{host}:{port}")
    logger.info(f"📚 API docs: http://{host}:{port}/docs\n")
    
    uvicorn.run(
        "proxy_service:app",
        host=host,
        port=port,
        log_level="info"
    )
