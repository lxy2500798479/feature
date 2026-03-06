"""
数据模型定义模块
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class NodeType(str, Enum):
    """节点类型"""
    DOCUMENT = "document"
    SECTION = "section"
    CHUNK = "chunk"
    CONCEPT = "concept"
    ENTITY = "entity"


class EdgeType(str, Enum):
    """边类型"""
    HAS_SECTION = "has_section"
    HAS_CHUNK = "has_chunk"
    BELONGS_TO = "belongs_to"
    HAS_CONCEPT = "has_concept"
    RELATED_TO = "related_to"
    NEXT_SECTION = "next_section"
    CONTAINS_CHUNK = "contains_chunk"
    NEXT_CHUNK = "next_chunk"


class DocumentMetadata(BaseModel):
    """文档元数据"""
    doc_id: str
    title: str
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    file_path: str
    file_type: str
    page_count: Optional[int] = None
    summary: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class SectionNode(BaseModel):
    """章节节点"""
    section_id: str
    doc_id: str
    title: str
    level: int
    hierarchy_path: str
    content: Optional[str] = None
    order: int
    parent_section_id: Optional[str] = None


class ChunkNode(BaseModel):
    """文本块节点"""
    chunk_id: str
    section_id: str
    doc_id: str
    text: str
    token_count: int
    position: int
    start_char: int
    end_char: int
    embedding_id: Optional[str] = None


class GraphEdge(BaseModel):
    """图边"""
    src_id: str
    dst_id: str
    edge_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class ConceptNode(BaseModel):
    """概念节点"""
    concept_id: str
    phrase: str
    doc_id: str
    freq: int = 0
    community: Optional[int] = None


class ParsedDocument(BaseModel):
    """解析后的文档"""
    metadata: DocumentMetadata
    sections: List[SectionNode] = Field(default_factory=list)
    chunks: List[ChunkNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class EntityNode(BaseModel):
    """实体节点"""
    entity_id: str
    name: str
    entity_type: str
    doc_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class RelationEdge(BaseModel):
    """关系边"""
    relation_id: str
    src_id: str
    dst_id: str
    relation_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """查询请求"""
    query: str
    top_k: int = 10
    filters: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """查询响应"""
    results: List[Dict[str, Any]]
    total: int
    query: str


class EnhancedQueryRequest(BaseModel):
    """增强查询请求"""
    query: str
    top_k: int = 10
    use_graph: bool = True
    use_vector: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)


class EnhancedQueryResponse(BaseModel):
    """增强查询响应"""
    results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    total: int
    query: str


class QueryMeta(BaseModel):
    """查询元数据"""
    query_id: str
    timestamp: datetime
    processing_time: float
