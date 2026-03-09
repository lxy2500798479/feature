"""
查询管道 - FinalRAG CoE 完整流程

基于设计文档的执行顺序:
  1. Router  → 识别 factual / relational / global
  2. CoE     → 三步层级导航检索
  3. Lazy Enhance（关联型）→ 按需图谱增强
  4. Synthesizer → 调用 DeepSeek 合成答案
  5. Degradation → 检索为空时兜底
"""
import time
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from src.utils.logger import logger
from src.query.budget_controller import get_budget_profile


@dataclass
class PipelineResult:
    """管道执行结果"""
    answer: str
    sources: List[dict] = field(default_factory=list)
    graph_context: Optional[dict] = None
    vector_context: Optional[List[dict]] = None
    metadata: Optional[dict] = field(default_factory=dict)
    query_type: str = "auto"
    retrieval_paths_used: List[str] = field(default_factory=list)
    budget_summary: Optional[dict] = None
    latency_breakdown: Optional[dict] = None
    degraded: bool = False
    degradation_reasons: List[str] = field(default_factory=list)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    _answer_stream: Optional[list] = None


class QueryPipeline:
    """查询管道 - 协调各个组件完成查询"""

    def __init__(
        self,
        router,
        coe_engine,
        graph_traversal,
        synthesizer,
        cache,
        degradation_manager,
        vector_client=None,
        lazy_enhancer=None,
        budget_profile: str = "medium"
    ):
        self.router = router
        self.coe_engine = coe_engine
        self.graph_traversal = graph_traversal
        self.synthesizer = synthesizer
        self.cache = cache
        self.degradation_manager = degradation_manager
        self.vector_client = vector_client
        self.lazy_enhancer = lazy_enhancer
        self.budget_profile = budget_profile

    def execute(
        self,
        query: str,
        query_embedding=None,
        budget_profile: str = "medium",
        stream: bool = False,
        top_k: int = 10,
        enable_lazy_enhance: bool = True,
        enable_entity_persist: bool = True,  # 新增：是否启用实体持久化
        override_query_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        **kwargs
    ) -> PipelineResult:
        """执行查询管道"""
        t_start = time.time()
        trace_id = str(uuid.uuid4())[:8]
        latency = {}

        logger.info(f"[{trace_id}] 开始查询: {query[:50]}... (type={override_query_type or 'auto'}, doc={doc_id})")

        # 获取预算配置
        budget = get_budget_profile(budget_profile)
        budget["max_vector_results"] = top_k

        # 1. 路由决策
        t0 = time.time()
        route = self.router.route(query, override_query_type=override_query_type)
        query_type = route["query_type"]
        latency["router"] = round(time.time() - t0, 3)
        logger.info(f"[{trace_id}] 路由结果: {query_type} ({route['description']})")

        # 2. CoE 三步检索
        t0 = time.time()
        coe_results = self.coe_engine.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            doc_id=doc_id,
            use_graph=route.get("use_graph", False),
            use_community=route.get("use_community", False),
        )
        latency["coe_retrieval"] = round(time.time() - t0, 3)

        vector_chunks = coe_results.get("vector_chunks", [])
        graph_chunks = coe_results.get("graph_chunks", [])
        community_ctx = coe_results.get("community_context", [])
        retrieval_paths = coe_results.get("retrieval_paths", [])

        logger.info(f"[{trace_id}] CoE 检索: vector={len(vector_chunks)}, graph={len(graph_chunks)}, community={len(community_ctx)}")

        # 3. 降级检测
        if not vector_chunks and not graph_chunks and not community_ctx:
            logger.warning(f"[{trace_id}] 检索结果为空，触发降级")
            result = self.degradation_manager.handle_degradation(query)
            result.trace_id = trace_id
            result.latency_breakdown = latency
            return result

        # 4. Lazy Entity Build（仅 relational 查询的持久化实体图谱）
        # 首次查询时从 chunks 抽取实体并存储，后续查询直接使用
        entity_graph_result = None
        if enable_entity_persist and doc_id and self.lazy_enhancer and query_type == "relational":
            t0 = time.time()
            try:
                # 检查是否已有持久化的实体，没有则抽取并存储
                entity_graph_result = self.lazy_enhancer.build(
                    doc_id=doc_id,
                    chunks=vector_chunks,
                    force_rebuild=False,
                )
                if entity_graph_result:
                    retrieval_paths.append(f"entity_graph(from_cache={entity_graph_result.get('from_cache', True)})")
                    logger.info(
                        f"[{trace_id}] 实体图谱: "
                        f"缓存={entity_graph_result.get('from_cache', True)}, "
                        f"实体={len(entity_graph_result.get('entities', []))}, "
                        f"新增={entity_graph_result.get('new_entities', 0)}"
                    )
            except Exception as e:
                logger.warning(f"[{trace_id}] 实体图谱构建失败（跳过）: {e}")
            latency["entity_graph"] = round(time.time() - t0, 3)

        # 5. Lazy Enhance（仅 relational 类型 + enable_lazy_enhance=True）
        lazy_entities = []
        if (
            query_type == "relational"
            and enable_lazy_enhance
            and self.lazy_enhancer is not None
        ):
            t0 = time.time()
            try:
                enhance_result = self.lazy_enhancer.enhance(
                    query=query,
                    seed_chunks=vector_chunks,
                    doc_id=doc_id,
                    query_embedding=query_embedding,
                    top_k=5,
                )
                extra_chunks = enhance_result.get("extra_chunks", [])
                lazy_entities = enhance_result.get("graph_entities", [])
                if extra_chunks:
                    # 把扩展 chunks 追加到 vector_chunks（去重）
                    existing_cids = {c.get("chunk_id", "") for c in vector_chunks}
                    for ec in extra_chunks:
                        if ec.get("chunk_id", "") not in existing_cids:
                            vector_chunks.append(ec)
                    retrieval_paths.append(f"lazy_enhance(+{len(extra_chunks)})")
                    logger.info(f"[{trace_id}] Lazy Enhance: +{len(extra_chunks)} chunks, "
                                f"entities={enhance_result.get('entities', [])[:3]}")
            except Exception as e:
                logger.warning(f"[{trace_id}] Lazy Enhance 失败（跳过）: {e}")
            latency["lazy_enhance"] = round(time.time() - t0, 3)

        # 6. 答案合成
        t0 = time.time()
        graph_ctx = {"chunks": graph_chunks, "communities": community_ctx} if (graph_chunks or community_ctx) else None

        if stream:
            # 流式模式：把生成器挂到 _answer_stream，routes 负责逐 token 发送
            answer_gen = self.synthesizer.synthesize_stream(
                query=query,
                vector_context=vector_chunks,
                graph_context=graph_ctx,
            )
            latency["synthesis_start"] = round(time.time() - t0, 3)
            latency["total"] = round(time.time() - t_start, 3)
            logger.info(f"[{trace_id}] 流式模式就绪: 耗时={latency['total']}s")

            return PipelineResult(
                answer="",
                _answer_stream=answer_gen,
                sources=vector_chunks[:5],
                vector_context=vector_chunks,
                graph_context=graph_ctx,
                query_type=query_type,
                retrieval_paths_used=retrieval_paths,
                budget_summary=budget,
                latency_breakdown=latency,
                degraded=False,
                trace_id=trace_id,
            )
        else:
            # 非流式模式：等待完整答案
            answer = self.synthesizer.synthesize(
                query=query,
                vector_context=vector_chunks,
                graph_context=graph_ctx,
            )
            latency["synthesis"] = round(time.time() - t0, 3)
            latency["total"] = round(time.time() - t_start, 3)
            logger.info(f"[{trace_id}] 完成: 耗时={latency['total']}s, 答案长度={len(answer)}")

            return PipelineResult(
                answer=answer,
                sources=vector_chunks[:5],
                vector_context=vector_chunks,
                graph_context=graph_ctx,
                query_type=query_type,
                retrieval_paths_used=retrieval_paths,
                budget_summary=budget,
                latency_breakdown=latency,
                degraded=False,
                trace_id=trace_id,
            )
