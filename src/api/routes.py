"""
FastAPI 路由定义
"""
import asyncio
import json
import time

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List
import shutil
from pathlib import Path

from src.config import settings
from src.core.models import (
    QueryRequest,
    QueryResponse,
    EnhancedQueryRequest,
    EnhancedQueryResponse,
    QueryMeta,
)
from src.services.document_service import DocumentService
from src.services.query_service import QueryService
from src.query.budget_controller import BUDGET_PROFILES
from src.query.pipeline import QueryPipeline, PipelineResult
from src.query.router import QueryRouter
from src.query.coe_engine import CoEEngine
from src.query.graph_traversal import GraphTraversal
from src.query.lazy_enhancer import LazyEnhancer
from src.query.answer_synthesizer import AnswerSynthesizer
from src.query.degradation import DegradationManager
from src.query.subgraph_cache import SubgraphCache
from src.embedding.text_embedder import TextEmbedder
from src.utils.llm_client import LLMClient
from pydantic import BaseModel
from src.storage.nebula_client import NebulaClient
from src.storage.vector_client import MilvusClient
from src.utils.logger import logger

router = APIRouter()

# 有效的检索模式和预算档位
VALID_RETRIEVAL_MODES = {"auto", "vector", "hybrid", "graph"}
VALID_BUDGET_PROFILES = set(BUDGET_PROFILES.keys())

# 初始化服务
document_service = DocumentService()
query_service = QueryService()


@router.post("/documents/upload", tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,  # FastAPI 会自动注入
    file: UploadFile = File(...),
    async_mode: bool = True  # 默认使用异步模式
):
    """
    上传并解析文档
    
    Args:
        file: 上传的文件
        background_tasks: 后台任务
        async_mode: 是否使用异步模式（True=快速返回，向量后台生成；False=等待完成）
        
    Returns:
        dict: 上传结果
    """
    # ── 文件校验 ──
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".xlsx"}
    MAX_FILE_SIZE_MB = getattr(settings, "MAX_UPLOAD_SIZE_MB", 100)
    MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024

    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"不支持的文件类型: '{ext}'，允许: {sorted(ALLOWED_EXTENSIONS)}",
            )

    # 读取内容并检查大小
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大: {len(content) / 1024 / 1024:.1f}MB，上限 {MAX_FILE_SIZE_MB}MB",
        )
    # 重置文件指针供后续使用
    await file.seek(0)

    try:
        # 保存文件到 uploads 目录
        upload_dir = Path(settings.UPLOAD_DIR) if hasattr(settings, 'UPLOAD_DIR') else Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"文件已上传: {file_path}")
        
        # 处理文档（快速阶段：解析 + 图谱构建）
        result = document_service.process_document(str(file_path), async_mode=async_mode)
        
        # 如果是异步模式，启动后台任务生成向量
        if async_mode:
            logger.info(f"启动后台任务生成向量嵌入: {result['doc_id']}")
            background_tasks.add_task(
                document_service.generate_embeddings_async,
                result['doc_id']
            )
            # 阶段 3：图片处理（与阶段 2 并行）
            logger.info(f"启动后台任务处理图片: {result['doc_id']}")
            background_tasks.add_task(
                document_service.process_images_async,
                result['doc_id']
            )
        
        return result
    
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    查询文档
    
    Args:
        request: 查询请求
        
    Returns:
        QueryResponse: 查询结果
    """
    try:
        result = query_service.query(request)
        return result
    
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", tags=["Documents"])
async def list_documents():
    """列出所有已入库的文档"""
    try:
        return document_service.list_documents()
    except Exception as e:
        logger.error(f"列出文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}", tags=["Documents"])
async def get_document(doc_id: str):
    """
    获取文档信息
    
    Args:
        doc_id: 文档ID
        
    Returns:
        dict: 文档信息
    """
    try:
        doc = document_service.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}/status", tags=["Documents"])
async def get_document_status(doc_id: str):
    """
    获取文档处理状态
    
    Args:
        doc_id: 文档ID
        
    Returns:
        dict: 文档状态信息，包括：
            - doc_id: 文档ID
            - graph_ready: 图谱是否就绪
            - embeddings_ready: 向量是否就绪
            - embedding_task_status: 任务状态
            - chunks_count: 文本块数量
            - embeddings_count: 向量数量
    """
    try:
        status = document_service.get_document_status(doc_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", tags=["System"])
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "FinalRAG",
        "phase": "Phase 2D: Fusion GraphRAG",
        "version": "0.2.0",
        "features": {
            "document_parsing": True,
            "concept_graph": True,
            "vector_embedding": True,
            "vector_retrieval": True,
            "graph_navigation": True,
            "enhanced_query": True,
            "lazy_enhance": True,
            "streaming": True,
        }
    }


# ── Phase 2D: 增强查询端点 ──


def _validate_enhanced_request(request: EnhancedQueryRequest) -> None:
    """校验增强查询请求参数，无效时抛 HTTPException 422"""
    if request.retrieval_mode not in VALID_RETRIEVAL_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"无效的 retrieval_mode: '{request.retrieval_mode}'，"
                   f"可选值: {sorted(VALID_RETRIEVAL_MODES)}",
        )
    if request.budget_profile not in VALID_BUDGET_PROFILES:
        raise HTTPException(
            status_code=422,
            detail=f"无效的 budget_profile: '{request.budget_profile}'，"
                   f"可选值: {sorted(VALID_BUDGET_PROFILES)}",
        )


# ── Pipeline 单例（避免每请求重建连接） ──

_pipeline_instance = None
_embedder_instance = None


def _get_pipeline():
    """获取 Pipeline 单例。首次调用时初始化，后续复用。"""
    global _pipeline_instance, _embedder_instance
    if _pipeline_instance is not None:
        return _pipeline_instance, _embedder_instance

    nebula = NebulaClient()
    nebula.connect()
    vector = MilvusClient()
    vector.init_collection()
    embedder = TextEmbedder()

    # 答案合成：DeepSeek（对话质量优先）
    llm = LLMClient()

    # 摘要/实体抽取：本地 qwen3.5-27b（成本低、私有化）
    # 优先读 SUMMARY_API_URL，未配时回退到 LLM_API_URL
    from src.config import settings as _settings
    summary_llm = LLMClient(
        api_url=_settings.SUMMARY_API_URL or _settings.LLM_API_URL,
        api_key=_settings.SUMMARY_API_KEY or _settings.LLM_API_KEY,
        model=_settings.SUMMARY_MODEL,
        timeout=_settings.SUMMARY_TIMEOUT,
    )

    router_inst = QueryRouter()
    coe = CoEEngine(vector_client=vector, nebula_client=nebula, embedder=embedder)
    traversal = GraphTraversal(nebula_client=nebula)
    # LazyEnhancer 实体抽取在查询路径上同步执行，用 DeepSeek（快速、无思考过程开销）
    # qwen3.5-27b-fp8 是推理模型，每次请求需生成 500+ 思考 tokens，延迟 50-70s，不适合实时路径
    lazy_enhancer = LazyEnhancer(llm_client=llm, vector_client=vector, nebula_client=nebula)
    synthesizer = AnswerSynthesizer(llm_client=llm)
    cache = SubgraphCache()
    degradation = DegradationManager()

    _pipeline_instance = QueryPipeline(
        router=router_inst,
        coe_engine=coe,
        graph_traversal=traversal,
        lazy_enhancer=lazy_enhancer,
        synthesizer=synthesizer,
        degradation_manager=degradation,
        cache=cache,
        vector_client=vector,
    )
    _embedder_instance = embedder
    logger.info("Pipeline 单例已初始化")
    return _pipeline_instance, _embedder_instance


@router.post(
    "/query/enhanced",
    response_model=EnhancedQueryResponse,
    tags=["Query"],
)
async def enhanced_query(request: EnhancedQueryRequest):
    """
    增强查询端点

    支持 retrieval_mode / enable_lazy_enhance / budget_profile 参数。
    非流式模式，返回完整 JSON 响应。
    """
    _validate_enhanced_request(request)

    start = time.time()
    query_timeout = getattr(settings, "QUERY_TIMEOUT_SECONDS", 120)
    try:
        pipeline, embedder = _get_pipeline()

        embed_start = time.time()
        query_embedding = embedder.embed_single(request.query)
        embed_ms = (time.time() - embed_start) * 1000
        logger.info(f"查询嵌入耗时: {embed_ms:.0f}ms")

        pipe_result = await asyncio.wait_for(
            asyncio.to_thread(
                pipeline.execute,
                query=request.query,
                query_embedding=query_embedding,
                budget_profile=request.budget_profile,
                stream=False,
                top_k=request.top_k,
                enable_lazy_enhance=request.enable_lazy_enhance,
                override_query_type=request.override_query_type,
            ),
            timeout=query_timeout,
        )

        took_ms = (time.time() - start) * 1000
        return EnhancedQueryResponse(
            query=request.query,
            answer=pipe_result.answer,
            results=pipe_result.chunks,
            total=len(pipe_result.chunks),
            took_ms=round(took_ms, 1),
            meta=QueryMeta(
                query_type=pipe_result.query_type,
                retrieval_paths_used=pipe_result.retrieval_paths_used,
                budget_consumed=pipe_result.budget_summary,
                latency_breakdown=pipe_result.latency_breakdown,
                degraded=pipe_result.degraded,
                degradation_reasons=pipe_result.degradation_reasons,
                trace_id=pipe_result.trace_id,
            ),
        )
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        took_ms = (time.time() - start) * 1000
        logger.error(f"增强查询超时: {took_ms:.0f}ms (上限 {query_timeout}s)")
        raise HTTPException(
            status_code=504,
            detail=f"查询超时: 超过 {query_timeout} 秒未完成",
        )
    except Exception as e:
        logger.error(f"增强查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream", tags=["Query"])
async def query_stream(request: EnhancedQueryRequest):
    """
    SSE 流式查询端点

    以 Server-Sent Events 格式返回：
      - event: meta  → 查询元信息
      - event: token → 答案 token（增量）
      - event: done  → 结束信号
    """
    _validate_enhanced_request(request)

    async def event_generator():
        start = time.time()
        try:
            pipeline, embedder = _get_pipeline()
            query_embedding = await asyncio.to_thread(
                embedder.embed_single, request.query
            )

            # 获取第一个 doc_id（如果有）
            doc_id = request.doc_ids[0] if request.doc_ids else None

            pipe_result = await asyncio.to_thread(
                pipeline.execute,
                query=request.query,
                query_embedding=query_embedding,
                budget_profile=request.budget_profile,
                stream=True,
                top_k=request.top_k,
                enable_lazy_enhance=request.enable_lazy_enhance,
                override_query_type=request.override_query_type,
                doc_id=doc_id,
            )

            meta = QueryMeta(
                query_type=pipe_result.query_type,
                retrieval_paths_used=pipe_result.retrieval_paths_used,
                budget_consumed=pipe_result.budget_summary,
                latency_breakdown=pipe_result.latency_breakdown,
                degraded=pipe_result.degraded,
                degradation_reasons=pipe_result.degradation_reasons,
                trace_id=pipe_result.trace_id,
            )
            yield f"event: meta\ndata: {meta.model_dump_json()}\n\n"

            answer_stream = getattr(pipe_result, '_answer_stream', None)
            if answer_stream is not None:
                import json as _json
                import queue as _queue
                token_q: _queue.Queue = _queue.Queue()
                _SENTINEL = object()

                def _drain():
                    try:
                        for tok in answer_stream:
                            token_q.put(tok)
                    except Exception as exc:
                        token_q.put(exc)
                    finally:
                        token_q.put(_SENTINEL)

                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, _drain)

                while True:
                    item = await asyncio.to_thread(token_q.get)
                    if item is _SENTINEL:
                        break
                    if isinstance(item, Exception):
                        yield f'event: error\ndata: {_json.dumps({"error": str(item)})}\n\n'
                        break
                    yield f'event: token\ndata: {_json.dumps({"text": item})}\n\n'
            elif pipe_result.answer:
                import json as _json
                yield f'event: token\ndata: {_json.dumps({"text": pipe_result.answer})}\n\n'

            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.error(f"流式查询异常: {e}")
            yield f'event: error\ndata: {str(e)}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ── 知识图谱可视化 API ──


@router.get("/graph/overview", tags=["Graph"])
async def graph_overview(doc_id: str = "", limit: int = 200, mode: str = "entity"):
    """
    返回图谱数据（节点 + 边），供前端 ReactFlow 渲染。

    - doc_id: 可选，限定某篇文档
    - limit:  返回的节点数上限
    - mode:   "entity"（LLM 实体关系图，默认）| "concept"（SpaCy 共现图）
    """
    try:
        pipeline, _ = _get_pipeline()
        nebula = pipeline.coe_engine.nebula_client

        # ── Entity 模式：走 Enhanced-KG ──────────────────────────────────────
        if mode == "entity":
            with nebula.get_session() as session:
                session.execute(f"USE {nebula.space_name};")

                # 1. 取 Entity 节点
                if doc_id:
                    ent_q = (
                        f'LOOKUP ON Entity WHERE Entity.doc_id == "{doc_id}" '
                        f'YIELD id(vertex) AS eid, '
                        f'Entity.name AS name, '
                        f'Entity.entity_type AS etype, '
                        f'Entity.description AS `desc` '
                        f'| LIMIT {limit};'
                    )
                else:
                    ent_q = (
                        f'LOOKUP ON Entity '
                        f'YIELD id(vertex) AS eid, '
                        f'Entity.name AS name, '
                        f'Entity.entity_type AS etype, '
                        f'Entity.description AS `desc` '
                        f'| LIMIT {limit};'
                    )

                er = session.execute(ent_q)
                if not er.is_succeeded():
                    # 索引未建好（schema 未重置）或 Entity 数据为空 → 返回空图谱
                    logger.warning(f"Entity LOOKUP 失败（可能索引未就绪）: {er.error_msg()}")
                    return {
                        "nodes": [], "edges": [], "communities": [],
                        "stats": {"total_nodes": 0, "total_edges": 0, "total_communities": 0},
                        "mode": "entity",
                        "hint": "Entity 索引未就绪或尚无数据，请清空数据库后重新上传文档",
                    }

                nodes = []
                entity_ids = []
                seen_eids: set = set()
                ENTITY_TYPE_MAP = {
                    "PERSON": 0, "ORG": 1, "PRODUCT": 2,
                    "LOCATION": 3, "EVENT": 4, "CONCEPT": 5,
                }
                for i in range(er.row_size()):
                    row = er.row_values(i)
                    eid = row[0].as_string()
                    if eid in seen_eids:
                        continue
                    seen_eids.add(eid)
                    name = row[1].as_string()
                    etype = row[2].as_string()
                    desc = row[3].as_string() if not row[3].is_empty() else ""
                    community = ENTITY_TYPE_MAP.get(etype, 6)
                    entity_ids.append(eid)
                    nodes.append({
                        "id": eid,
                        "phrase": name,
                        "community": community,
                        "entity_type": etype,
                        "description": desc,
                    })
                    if len(nodes) >= limit:
                        break

                # 2. 取 RELATION 边
                edges = []
                eid_set = set(entity_ids)
                seen_edges: set = set()
                BATCH = 50
                for batch_start in range(0, len(entity_ids), BATCH):
                    batch_ids = entity_ids[batch_start:batch_start + BATCH]
                    ids_str = ", ".join(f'"{eid}"' for eid in batch_ids)
                    rel_q = (
                        f"GO FROM {ids_str} OVER RELATION "
                        f"YIELD src(edge) AS source, dst(edge) AS target, "
                        f"RELATION.relation_type AS rel_type, "
                        f"RELATION.strength AS strength;"
                    )
                    rr = session.execute(rel_q)
                    if rr.is_succeeded():
                        for i in range(rr.row_size()):
                            row = rr.row_values(i)
                            s = row[0].as_string()
                            t = row[1].as_string()
                            if t not in eid_set:
                                continue
                            rt = row[2].as_string() if not row[2].is_empty() else "RELATED"
                            w = row[3].as_double() if not row[3].is_empty() else 1.0
                            key = (s, t)
                            if key not in seen_edges:
                                seen_edges.add(key)
                                edges.append({"source": s, "target": t, "weight": round(w, 4), "label": rt})

                type_set = set(n["entity_type"] for n in nodes)
                return {
                    "nodes": nodes,
                    "edges": edges,
                    "communities": sorted(ENTITY_TYPE_MAP.get(t, 6) for t in type_set),
                    "stats": {
                        "total_nodes": len(nodes),
                        "total_edges": len(edges),
                        "total_communities": len(type_set),
                    },
                    "mode": "entity",
                }

        # ── Concept 模式：走 SpaCy 共现图（原逻辑）────────────────────────────
        with nebula.get_session() as session:
            session.execute(f"USE {nebula.space_name};")

            # 1. 取概念节点
            if doc_id:
                # 直接用 doc_id 属性过滤（已建 idx_concept_doc_id 索引）
                concept_query = (
                    f'LOOKUP ON Concept WHERE Concept.doc_id == "{doc_id}" '
                    f"YIELD id(vertex) AS concept_id, "
                    f"Concept.phrase AS phrase, "
                    f"Concept.community AS community "
                    f"| LIMIT {limit};"
                )
            else:
                # 无 doc_id 时：查全部概念（依赖 idx_concept_phrase 索引）
                concept_query = (
                    f"LOOKUP ON Concept "
                    f"YIELD id(vertex) AS concept_id, "
                    f"Concept.phrase AS phrase, "
                    f"Concept.community AS community "
                    f"| LIMIT {limit};"
                )

            result = session.execute(concept_query)
            if not result.is_succeeded():
                raise HTTPException(500, f"概念查询失败: {result.error_msg()}")

            nodes = []
            concept_ids = []
            seen_concept_ids: set = set()
            for i in range(result.row_size()):
                row = result.row_values(i)
                cid = row[0].as_string()
                if cid in seen_concept_ids:
                    continue
                seen_concept_ids.add(cid)
                phrase = row[1].as_string()
                community = row[2].as_int() if not row[2].is_empty() else -1
                concept_ids.append(cid)
                nodes.append({
                    "id": cid,
                    "phrase": phrase,
                    "community": community,
                })
                if len(nodes) >= limit:
                    break

            # 2. 取这些概念间的 COOCCURS_WITH 边（分批查询避免 nGQL 过大）
            edges = []
            concept_id_set = set(concept_ids)
            seen_edges: set = set()
            BATCH = 50
            for batch_start in range(0, len(concept_ids), BATCH):
                batch_ids = concept_ids[batch_start:batch_start + BATCH]
                ids_str = ", ".join(f'"{cid}"' for cid in batch_ids)
                edge_query = (
                    f"GO FROM {ids_str} OVER COOCCURS_WITH "
                    f"YIELD "
                    f"src(edge) AS source, "
                    f"dst(edge) AS target, "
                    f"COOCCURS_WITH.weight AS weight;"
                )
                edge_result = session.execute(edge_query)
                if edge_result.is_succeeded():
                    for i in range(edge_result.row_size()):
                        row = edge_result.row_values(i)
                        s = row[0].as_string()
                        t = row[1].as_string()
                        if t not in concept_id_set:
                            continue
                        w = row[2].as_double()
                        key = tuple(sorted([s, t]))
                        if key not in seen_edges:
                            seen_edges.add(key)
                            edges.append({"source": s, "target": t, "weight": round(w, 4)})

            # 3. 统计社区信息
            community_set = set(n["community"] for n in nodes if n["community"] >= 0)

        return {
            "nodes": nodes,
            "edges": edges,
            "communities": sorted(community_set),
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "total_communities": len(community_set),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"图谱概览查询失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


# ── Admin ──────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    reset_graph: bool = True
    reset_vector: bool = True


@router.post("/admin/reset", tags=["Admin"])
async def reset_databases(req: ResetRequest):
    """
    清空 NebulaGraph 和/或 Milvus 数据。
    NebulaGraph reset 需要约 15 秒（DROP + CREATE SPACE + 等心跳）。
    完成后 Pipeline 单例也会被重置，下次查询时自动重建连接。
    """
    global _pipeline_instance, _embedder_instance

    results: dict = {}

    try:
        if req.reset_graph:
            logger.warning("🗑  开始重置 NebulaGraph 图空间…")
            await asyncio.to_thread(document_service.nebula_client.reset_space)
            results["graph"] = "reset"
            logger.warning("✅ NebulaGraph 图空间重置完成")

        if req.reset_vector:
            logger.warning("🗑  开始重置 Milvus collection…")
            document_service.vector_client.reset_collection()
            results["vector"] = "reset"
            logger.warning("✅ Milvus collection 重置完成")

        # Pipeline 单例持有旧连接，重置后需要重建
        _pipeline_instance = None
        _embedder_instance = None

        return {"status": "ok", "reset": results}

    except Exception as e:
        logger.error(f"数据库重置失败: {e}")
        raise HTTPException(500, f"重置失败: {e}")
