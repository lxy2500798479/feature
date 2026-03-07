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
        self.logger = logger

    def _parse_vertex_props(self, value) -> Dict:
        """解析 NebulaGraph Vertex 对象的属性"""
        # Nebula3 Value 需要用 get_vVal() 方法而非直接访问属性
        try:
            vertex = value.get_vVal()
        except Exception:
            return {}

        if not vertex or not vertex.tags:
            return {}

        props = {}
        for tag in vertex.tags:
            if tag.props:
                for k, v in tag.props.items():
                    key = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                    props[key] = self._parse_value(v)

        return props

    def _parse_value(self, value) -> Any:
        """解析 NebulaGraph Value 对象"""
        if value is None:
            return None

        try:
            t = value.getType()
        except Exception:
            return None

        # 0=NULL, 2=BOOL, 3=INT, 4=FLOAT, 5=STRING, 9=VERTEX
        if t == 0:
            return None
        elif t == 2:
            return value.get_bVal()
        elif t == 3:
            return value.get_iVal()
        elif t == 4:
            return value.get_fVal()
        elif t == 5:
            s = value.get_sVal()
            return s.decode('utf-8') if isinstance(s, bytes) else s
        elif t == 9:
            return self._parse_vertex_props(value)
        else:
            return None
    
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
                    `order` int,
                    summary string DEFAULT ""
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
                """CREATE TAG IF NOT EXISTS CommunitySummary (
                    doc_id string,
                    community_id int,
                    summary string
                );""",
                """CREATE EDGE IF NOT EXISTS HAS_SECTION();""",
                """CREATE EDGE IF NOT EXISTS HAS_CHUNK();""",
                """CREATE EDGE IF NOT EXISTS CONTAINS_CONCEPT();""",
                """CREATE EDGE IF NOT EXISTS COOCCURS_WITH(weight double, cooccur int);""",
                # ── Enhanced-KG: 真正的实体关系图谱 ──
                """CREATE TAG IF NOT EXISTS Entity (
                    name string,
                    entity_type string,
                    description string DEFAULT "",
                    doc_id string,
                    chunk_id string DEFAULT ""
                );""",
                """CREATE EDGE IF NOT EXISTS RELATION(
                    relation_type string,
                    description string DEFAULT "",
                    strength double DEFAULT 1.0
                );""",
            ]
            
            for tag_query in tags:
                result = session.execute(tag_query)
                if not result.is_succeeded():
                    logger.warning(f"Schema 创建警告: {result.error_msg()}")

            # 创建 LOOKUP 所需的 Tag 索引
            index_queries = [
                "CREATE TAG INDEX IF NOT EXISTS idx_concept_phrase ON Concept(phrase(100));",
                "CREATE TAG INDEX IF NOT EXISTS idx_concept_doc_id ON Concept(doc_id(64));",
                "CREATE TAG INDEX IF NOT EXISTS idx_entity_doc_id ON Entity(doc_id(64));",
                "CREATE TAG INDEX IF NOT EXISTS idx_entity_name ON Entity(name(100));",
            ]
            for iq in index_queries:
                r = session.execute(iq)
                if not r.is_succeeded():
                    logger.warning(f"索引创建警告: {r.error_msg()}")
            
            logger.info("Schema 初始化成功")

    def _init_schema_in_session(self, session):
        """在已有 session 中初始化 Schema"""
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
                `order` int,
                summary string DEFAULT ""
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
            """CREATE TAG IF NOT EXISTS CommunitySummary (
                doc_id string,
                community_id int,
                summary string
            );""",
            """CREATE EDGE IF NOT EXISTS HAS_SECTION();""",
            """CREATE EDGE IF NOT EXISTS HAS_CHUNK();""",
            """CREATE EDGE IF NOT EXISTS CONTAINS_CONCEPT();""",
            """CREATE EDGE IF NOT EXISTS COOCCURS_WITH(weight double, cooccur int);""",
            # ── Enhanced-KG: 真正的实体关系图谱 ──
            """CREATE TAG IF NOT EXISTS Entity (
                name string,
                entity_type string,
                description string DEFAULT "",
                doc_id string,
                chunk_id string DEFAULT ""
            );""",
            """CREATE EDGE IF NOT EXISTS RELATION(
                relation_type string,
                description string DEFAULT "",
                strength double DEFAULT 1.0
            );""",
        ]

        for tag_query in tags:
            result = session.execute(tag_query)
            if not result.is_succeeded():
                logger.warning(f"Schema 创建警告: {result.error_msg()}")

        # 创建 LOOKUP 所需的 Tag 索引
        index_queries = [
            "CREATE TAG INDEX IF NOT EXISTS idx_concept_phrase ON Concept(phrase(100));",
            "CREATE TAG INDEX IF NOT EXISTS idx_concept_doc_id ON Concept(doc_id(64));",
            "CREATE TAG INDEX IF NOT EXISTS idx_entity_doc_id ON Entity(doc_id(64));",
            "CREATE TAG INDEX IF NOT EXISTS idx_entity_name ON Entity(name(100));",
        ]
        for iq in index_queries:
            r = session.execute(iq)
            if not r.is_succeeded():
                logger.warning(f"索引创建警告: {r.error_msg()}")

        logger.info("Schema 初始化成功")

    def reset_space(self):
        """清空图空间所有数据"""
        with self.get_session() as session:
            # 删除并重新创建图空间
            session.execute(f"DROP SPACE IF EXISTS {self.space_name};")
            result = session.execute(f"CREATE SPACE {self.space_name} (partition_num=15, replica_factor=1, vid_type=fixed_string(64));")
            logger.info(f"创建图空间: {result.is_succeeded()}")
            if not result.is_succeeded():
                logger.error(f"创建图空间失败: {result.error_msg()}")
                return  # 直接返回，不再继续

            # NebulaGraph 的 CREATE SPACE 是异步的，需要等待心跳周期（默认10秒）
            import time
            logger.info("等待图空间创建完成（约15秒）...")
            time.sleep(15)

            # 切换到新创建的 space
            session.execute(f"USE {self.space_name};")

            # 重新初始化 schema
            self._init_schema_in_session(session)
            logger.warning(f"已重置 NebulaGraph 图空间: {self.space_name}")

    def get_documents(self) -> List[Dict]:
        """获取所有文档（用 SCAN VERTEX 避免 MATCH 全表扫描问题）"""
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            # MATCH (d:Document) 在 NebulaGraph 没有索引时可能返回 0 行
            # 改用 SCAN VERTEX，直接扫描 Document tag 的所有顶点
            result = session.execute('SCAN VERTEX WITH LIMIT 200 YIELD VERTEX AS v;')

            # fallback: 若 SCAN 不可用则用 MATCH
            if not result.is_succeeded():
                result = session.execute('MATCH (d:Document) RETURN d;')

            documents = []
            if result.is_succeeded():
                for row in result.rows():
                    doc_value = row.values[0]
                    doc_data = self._parse_vertex_props(doc_value)
                    if doc_data and "doc_id" in doc_data:
                        documents.append({
                            "doc_id": doc_data.get("doc_id", ""),
                            "title": doc_data.get("title", ""),
                            "file_path": doc_data.get("file_path", ""),
                            "file_type": doc_data.get("file_type", ""),
                            "graph_ready": doc_data.get("graph_ready", False),
                            "embeddings_ready": doc_data.get("embeddings_ready", False),
                            "embedding_task_status": doc_data.get("embedding_task_status", ""),
                        })
            return documents

    def update_document(self, doc_id: str, properties: Dict):
        """更新文档属性（使用 NebulaGraph 正确语法 UPDATE VERTEX ON Tag）"""
        if not properties:
            return
        with self.get_session() as session:
            r = session.execute(f"USE {self.space_name};")
            if not r.is_succeeded():
                self.logger.error(f"update_document: USE {self.space_name} 失败: {r.error_msg()}")
                return
            set_clauses = []
            for key, value in properties.items():
                if isinstance(value, str):
                    escaped = value.replace('"', '\\"')
                    set_clauses.append(f'{key} = "{escaped}"')
                elif isinstance(value, bool):
                    set_clauses.append(f'{key} = {str(value).lower()}')
                else:
                    set_clauses.append(f'{key} = {value}')

            if set_clauses:
                # NebulaGraph 正确语法: UPDATE VERTEX ON <tag> "<vid>" SET ...
                query = f'UPDATE VERTEX ON Document "{doc_id}" SET {", ".join(set_clauses)};'
                result = session.execute(query)
                if not result.is_succeeded():
                    self.logger.error(
                        f"update_document 失败 doc_id={doc_id}: {result.error_msg()}"
                    )

    def insert_document(self, metadata):
        """插入文档节点"""
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            
            # 支持 Pydantic 模型或 Dict
            if hasattr(metadata, 'model_dump'):
                meta = metadata.model_dump()
            elif hasattr(metadata, 'dict'):
                meta = metadata.dict()
            else:
                meta = metadata
            
            doc_id = meta.get("doc_id", "")
            title = meta.get("title", "")
            author = meta.get("author", "")
            file_path = meta.get("file_path", "")
            file_type = meta.get("file_type", "")
            page_count = meta.get("page_count", 0) or 0
            summary = meta.get("summary", "")
            
            # 正确的 NebulaGraph 语法: VALUES "vid":(props)
            # 注意: None 值需要转换为空字符串或 0
            query = f'''INSERT VERTEX Document(
                doc_id, title, author, created_date, file_path, file_type, page_count, summary, graph_ready, embeddings_ready, embedding_task_status
            ) VALUES "{doc_id}":(
                "{doc_id}", "{title}", "{author}", "",
                "{file_path}", "{file_type}", {page_count}, "{summary}", false, false, "processing"
            );'''
            
            self.logger.info(f"插入文档: {query}")
            result = session.execute(query)
            self.logger.info(f"插入结果: {result.is_succeeded()}, error: {result.error_msg() if not result.is_succeeded() else 'None'}")

    def insert_sections(self, sections):
        """插入章节节点"""
        if not sections:
            return
            
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            
            for section in sections:
                # 支持 Pydantic 模型或 Dict
                if hasattr(section, 'model_dump'):
                    sec = section.model_dump()
                elif hasattr(section, 'dict'):
                    sec = section.dict()
                else:
                    sec = section
                    
                section_id = sec.get("section_id", "")
                doc_id = sec.get("doc_id", "")
                title = sec.get("title", "")
                level = sec.get("level", 1)
                hierarchy_path = sec.get("hierarchy_path", "")
                content = sec.get("content", "")
                order = sec.get("order", 0)
                
                query = f'''INSERT VERTEX Section(
                    doc_id, title, level, hierarchy_path, content, `order`
                ) VALUES "{section_id}":(
                    "{doc_id}", "{title}", {level}, 
                    "{hierarchy_path}", "{content}", {order}
                );'''
                session.execute(query)

    def insert_chunks(self, chunks):
        """插入文本块节点"""
        if not chunks:
            return
            
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            
            for chunk in chunks:
                # 支持 Pydantic 模型或 Dict
                if hasattr(chunk, 'model_dump'):
                    ch = chunk.model_dump()
                elif hasattr(chunk, 'dict'):
                    ch = chunk.dict()
                else:
                    ch = chunk
                    
                chunk_id = ch.get("chunk_id", "")
                section_id = ch.get("section_id", "")
                doc_id = ch.get("doc_id", "")
                text = ch.get("text", "")
                token_count = ch.get("token_count", 0)
                position = ch.get("position", 0)
                
                # 清理文本中的特殊字符
                text = text.replace('"', '\\"').replace('\n', ' ')
                
                query = f'''INSERT VERTEX Chunk(
                    section_id, doc_id, text, token_count, position
                ) VALUES "{chunk_id}":(
                    "{section_id}", "{doc_id}", "{text}", 
                    {token_count}, {position}
                );'''
                session.execute(query)

    def insert_edges(self, edges):
        """插入边关系"""
        if not edges:
            return
            
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            
            for edge in edges:
                # 支持 Pydantic 模型或 Dict
                if hasattr(edge, 'model_dump'):
                    e = edge.model_dump()
                elif hasattr(edge, 'dict'):
                    e = edge.dict()
                else:
                    e = edge
                    
                src_id = e.get("src_id", "")
                dst_id = e.get("dst_id", "")
                edge_type = e.get("edge_type", "")
                
                if edge_type in ("has_section",):
                    query = f'INSERT EDGE HAS_SECTION() VALUES "{src_id}"->"{dst_id}":();'
                elif edge_type in ("has_chunk", "contains_chunk"):
                    query = f'INSERT EDGE HAS_CHUNK() VALUES "{src_id}"->"{dst_id}":();'
                elif edge_type == "contains_concept":
                    query = f'INSERT EDGE CONTAINS_CONCEPT() VALUES "{src_id}"->"{dst_id}":();'
                elif edge_type in ("next_chunk", "next_section", "belongs_to", "related_to", "has_concept"):
                    continue
                else:
                    continue
                    
                session.execute(query)

    def insert_concept_graph(self, doc_id: str, nodes: List[Dict], edges: List[Dict]):
        """插入概念图谱节点和边（Concept 节点 + COOCCURS_WITH 边）"""
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")

            # 插入 Concept 节点（字段：phrase, freq, community, doc_id）
            for node in nodes:
                concept_id = node.get("id", "")
                phrase = node.get("phrase", "").replace('"', '\\"').replace('\\', '\\\\')
                freq = int(node.get("freq", 0))
                community = int(node.get("community", -1))

                query = (
                    f'INSERT VERTEX Concept(phrase, freq, community, doc_id) '
                    f'VALUES "{concept_id}":("{phrase}", {freq}, {community}, "{doc_id}");'
                )
                r = session.execute(query)
                if not r.is_succeeded():
                    self.logger.warning(f"Concept 节点插入失败 {concept_id}: {r.error_msg()}")

            # 插入 COOCCURS_WITH 边（字段：weight, cooccur）
            for edge in edges:
                src_id = edge.get("from", "")
                dst_id = edge.get("to", "")
                weight = float(edge.get("weight", 0.0))
                cooccur = int(edge.get("cooccur", 0))

                query = (
                    f'INSERT EDGE COOCCURS_WITH(weight, cooccur) '
                    f'VALUES "{src_id}"->"{dst_id}":({weight}, {cooccur});'
                )
                r = session.execute(query)
                if not r.is_succeeded():
                    self.logger.warning(f"COOCCURS_WITH 边插入失败 {src_id}->{dst_id}: {r.error_msg()}")

    def store_community_summaries(self, doc_id: str, summaries: Dict[int, str]):
        """存储社区摘要"""
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            
            for community_id, summary in summaries.items():
                # 转义特殊字符
                summary = summary.replace('"', '\\"').replace('\n', ' ')
                
                query = f'''INSERT VERTEX CommunitySummary(
                    doc_id, community_id, summary
                ) VALUES "{doc_id}_{community_id}":(
                    "{doc_id}", {community_id}, "{summary}"
                );'''
                session.execute(query)

    def get_document_status(self, doc_id: str) -> Dict:
        """获取文档状态（FETCH 直接返回各属性列，避免 Vertex 类型解析差异）"""
        with self.get_session() as session:
            r = session.execute(f"USE {self.space_name};")
            if not r.is_succeeded():
                self.logger.warning(f"USE {self.space_name} 失败: {r.error_msg()}")
                return {"doc_id": doc_id, "embeddings_ready": False, "embedding_task_status": "unknown"}

            query = (
                f'FETCH PROP ON Document "{doc_id}" '
                f'YIELD Document.title AS title, '
                f'Document.graph_ready AS graph_ready, '
                f'Document.embeddings_ready AS embeddings_ready, '
                f'Document.embedding_task_status AS embedding_task_status;'
            )
            result = session.execute(query)

            if result.is_succeeded() and result.row_size() > 0:
                row = result.rows()[0]
                vals = row.values
                return {
                    "doc_id": doc_id,
                    "title": self._parse_value(vals[0]) or "",
                    "graph_ready": bool(self._parse_value(vals[1])),
                    "embeddings_ready": bool(self._parse_value(vals[2])),
                    "embedding_task_status": self._parse_value(vals[3]) or "processing",
                }

            # NebulaGraph 里找不到时返回 failed 而不是 not_found，
            # 让前端能停止轮询（not_found 没有 embeddings_ready 字段会无限轮询）
            return {
                "doc_id": doc_id,
                "embeddings_ready": False,
                "embedding_task_status": "failed",
                "error": "document_not_found_in_graph",
            }

            return {"doc_id": doc_id, "status": "not_found"}

    def update_section_summaries(self, summaries: Dict[str, str]):
        """批量更新 Section 的 summary 字段
        
        Args:
            summaries: {section_id: summary_text}
        """
        if not summaries:
            return
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            ok = fail = 0
            for section_id, summary in summaries.items():
                summary_clean = summary.replace('"', '\\"').replace('\n', ' ')[:500]
                q = f'UPDATE VERTEX ON Section "{section_id}" SET summary = "{summary_clean}";'
                r = session.execute(q)
                if r.is_succeeded():
                    ok += 1
                else:
                    fail += 1
            logger.info(f"Section summary 更新: {ok} 成功, {fail} 失败")

    def get_sections_with_summaries(self, doc_id: str) -> List[Dict]:
        """获取文档所有 Section 的摘要列表（用于 CoE Step 2 精排）
        
        Returns:
            [{"section_id": ..., "title": ..., "summary": ..., "order": ...}, ...]
        """
        results = []
        try:
            with self.get_session() as session:
                session.execute(f"USE {self.space_name};")
                q = f'MATCH (s:Section) WHERE s.doc_id == "{doc_id}" RETURN id(s) as sid, s.title as title, s.summary as summary, s.`order` as ord LIMIT 200;'
                r = session.execute(q)
                if r.is_succeeded():
                    for row in r.rows():
                        try:
                            sid_v = row.values[0]
                            title_v = row.values[1]
                            summary_v = row.values[2]
                            order_v = row.values[3]

                            sid = sid_v.get_sVal() if sid_v.getType() == 5 else b""
                            sid = sid.decode() if isinstance(sid, bytes) else sid

                            title = title_v.get_sVal() if title_v.getType() == 5 else b""
                            title = title.decode() if isinstance(title, bytes) else title

                            summary = summary_v.get_sVal() if summary_v.getType() == 5 else b""
                            summary = summary.decode() if isinstance(summary, bytes) else summary

                            order = order_v.get_iVal() if order_v.getType() == 3 else 0

                            if sid:
                                results.append({
                                    "section_id": sid,
                                    "title": title,
                                    "summary": summary,
                                    "order": order,
                                })
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"获取 Section summaries 失败: {e}")
        return sorted(results, key=lambda x: x["order"])

    def insert_entity_graph(self, doc_id: str, entities: List[Dict], relations: List[Dict]):
        """插入 LLM 抽取的实体关系图（Enhanced-KG）

        entities: [{id, name, entity_type, description, chunk_id}]
        relations: [{src_id, dst_id, relation_type, description, strength}]
        """
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")

            for ent in entities:
                eid = ent.get("id", "")
                name = ent.get("name", "").replace('"', '\\"')
                etype = ent.get("entity_type", "UNKNOWN").replace('"', '\\"')
                desc = ent.get("description", "").replace('"', '\\"').replace('\n', ' ')
                chunk_id = ent.get("chunk_id", "")

                q = (
                    f'INSERT VERTEX Entity(name, entity_type, description, doc_id, chunk_id) '
                    f'VALUES "{eid}":("{name}", "{etype}", "{desc}", "{doc_id}", "{chunk_id}");'
                )
                r = session.execute(q)
                if not r.is_succeeded():
                    self.logger.warning(f"Entity 插入失败 {eid}: {r.error_msg()}")

            for rel in relations:
                src = rel.get("src_id", "")
                dst = rel.get("dst_id", "")
                rtype = rel.get("relation_type", "RELATED").replace('"', '\\"')
                rdesc = rel.get("description", "").replace('"', '\\"').replace('\n', ' ')
                strength = float(rel.get("strength", 1.0))

                q = (
                    f'INSERT EDGE RELATION(relation_type, description, strength) '
                    f'VALUES "{src}"->"{dst}":("{rtype}", "{rdesc}", {strength});'
                )
                r = session.execute(q)
                if not r.is_succeeded():
                    self.logger.warning(f"RELATION 边插入失败 {src}->{dst}: {r.error_msg()}")

    def get_entity_neighbors(self, entity_name: str, doc_id: str = "", hops: int = 2) -> List[Dict]:
        """查询实体的 N 跳邻居（用于 Lazy Enhancer 图遍历）"""
        results = []
        try:
            with self.get_session() as session:
                session.execute(f"USE {self.space_name};")
                # 先用 LOOKUP 找到实体 VID
                if doc_id:
                    lookup_q = (
                        f'LOOKUP ON Entity WHERE Entity.name == "{entity_name}" '
                        f'AND Entity.doc_id == "{doc_id}" '
                        f'YIELD id(vertex) AS vid LIMIT 5;'
                    )
                else:
                    lookup_q = (
                        f'LOOKUP ON Entity WHERE Entity.name == "{entity_name}" '
                        f'YIELD id(vertex) AS vid LIMIT 5;'
                    )
                lr = session.execute(lookup_q)
                if not lr.is_succeeded() or lr.row_size() == 0:
                    return results

                vids = []
                for row in lr.rows():
                    v = row.values[0]
                    if v.getType() == 5:
                        p = v.get_sVal()
                        vids.append(p.decode() if isinstance(p, bytes) else p)

                if not vids:
                    return results

                ids_str = ", ".join(f'"{v}"' for v in vids)
                go_q = (
                    f'GO {hops} STEPS FROM {ids_str} OVER RELATION '
                    f'YIELD dst(edge) AS dst_id, '
                    f'RELATION.relation_type AS rel_type, '
                    f'RELATION.description AS rel_desc, '
                    f'properties($$).name AS dst_name, '
                    f'properties($$).entity_type AS dst_type;'
                )
                gr = session.execute(go_q)
                if gr.is_succeeded():
                    for row in gr.rows():
                        try:
                            def sv(v):
                                if v.getType() == 5:
                                    p = v.get_sVal()
                                    return p.decode() if isinstance(p, bytes) else p
                                return ""
                            results.append({
                                "dst_id": sv(row.values[0]),
                                "rel_type": sv(row.values[1]),
                                "rel_desc": sv(row.values[2]),
                                "dst_name": sv(row.values[3]),
                                "dst_type": sv(row.values[4]),
                            })
                        except Exception:
                            pass
        except Exception as e:
            self.logger.warning(f"get_entity_neighbors 失败: {e}")
        return results
