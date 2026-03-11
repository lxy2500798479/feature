"""
数据模型定义模块 - 丰富版
支持多种文档类型：商业BP、市场报告、合同、小说、技术文档等
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# 基础枚举类型
# ============================================================================

class NodeType(str, Enum):
    """节点类型"""
    DOCUMENT = "document"
    SECTION = "section"
    CHUNK = "chunk"
    CONCEPT = "concept"
    ENTITY = "entity"
    ENTITY_MENTION = "entity_mention"  # 实体提及


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


class DocumentType(str, Enum):
    """文档类型"""
    # 商业/金融类
    BUSINESS_BP = "business_bp"           # 商业计划书
    INVESTMENT_MEMO = "investment_memo"   # 投资备忘录
    FINANCIAL_REPORT = "financial_report"  # 财务报告
    PROSPECTUS = "prospectus"              # 招股说明书
    DUE_DILIGENCE = "due_diligence"        # 尽职调查报告
    
    # 市场/研究类
    MARKET_REPORT = "market_report"       # 市场报告
    INDUSTRY_ANALYSIS = "industry_analysis" # 行业分析
    COMPETITIVE_ANALYSIS = "competitive_analysis"  # 竞争分析
    
    # 法律/合同类
    CONTRACT = "contract"                 # 合同
    LEGAL_DOC = "legal_doc"                # 法律文书
    NDA = "nda"                            # 保密协议
    
    # 技术/文档类
    TECHNICAL_DOC = "technical_doc"       # 技术文档
    API_DOC = "api_doc"                    # API文档
    README = "readme"                      # 说明文档
    
    # 创意/内容类
    NOVEL = "novel"                        # 小说
    ARTICLE = "article"                     # 文章
    BLOG_POST = "blog_post"                # 博客
    NEWS = "news"                          # 新闻
    
    # 沟通类
    EMAIL = "email"                        # 邮件
    CHAT_LOG = "chat_log"                  # 聊天记录
    MEETING_MINUTES = "meeting_minutes"    # 会议纪要
    
    # 其他
    PRESENTATION = "presentation"          # 演示文稿
    WEB_PAGE = "web_page"                  # 网页
    GENERIC = "generic"                    # 通用文档


class RichEntityType(str, Enum):
    """丰富的实体类型 - 适用于多种文档"""
    
    # 组织/商业实体
    COMPANY = "company"                     # 公司
    SUBSIDIARY = "subsidiary"               # 子公司
    BRANCH = "branch"                       # 分公司/分支机构
    DEPARTMENT = "department"              # 部门
    ORGANIZATION = "organization"           # 组织
    GROUP = "group"                         # 集团
    
    # 人物相关
    PERSON = "person"                       # 人物
    FOUNDER = "founder"                     # 创始人
    CEO = "ceo"                             # 首席执行官
    CFO = "cfo"                             # 首席财务官
    CTO = "cto"                             # 首席技术官
    EMPLOYEE = "employee"                   # 员工
    INVESTOR = "investor"                   # 投资者
    SHAREHOLDER = "shareholder"             # 股东
    BOARD_MEMBER = "board_member"           # 董事会成员
    ADVISOR = "advisor"                     # 顾问
    CUSTOMER = "customer"                   # 客户
    SUPPLIER = "supplier"                   # 供应商
    PARTNER = "partner"                     # 合作伙伴
    COMPETITOR = "competitor"               # 竞争对手
    
    # 产品/服务
    PRODUCT = "product"                     # 产品
    SERVICE = "service"                     # 服务
    PLATFORM = "platform"                   # 平台
    APP = "app"                             # 应用
    FEATURE = "feature"                     # 功能
    BRAND = "brand"                         # 品牌
    
    # 市场/行业
    MARKET = "market"                       # 市场
    INDUSTRY = "industry"                   # 行业
    MARKET_SEGMENT = "market_segment"       # 市场细分
    TARGET_AUDIENCE = "target_audience"    # 目标受众
    USER_PERSONA = "user_persona"          # 用户画像
    
    # 财务相关
    FINANCIAL_METRIC = "financial_metric"  # 财务指标
    REVENUE = "revenue"                     # 收入
    PROFIT = "profit"                       # 利润
    COST = "cost"                           # 成本
    EXPENSE = "expense"                     # 费用
    INVESTMENT = "investment"               # 投资
    FUNDING = "funding"                     # 融资
    VALUATION = "valuation"                # 估值
    EQUITY = "equity"                       # 股权
    DEBT = "debt"                           # 债务
    ASSET = "asset"                        # 资产
    LIABILITY = "liability"                # 负债
    CASH_FLOW = "cash_flow"                 # 现金流
    
    # 技术相关
    TECHNOLOGY = "technology"               # 技术
    TECH_STACK = "tech_stack"               # 技术栈
    SOFTWARE = "software"                   # 软件
    HARDWARE = "hardware"                   # 硬件
    API = "api"                             # API
    DATABASE = "database"                   # 数据库
    INFRASTRUCTURE = "infrastructure"       # 基础设施
    PATENT = "patent"                       # 专利
    IP = "ip"                               # 知识产权
    
    # 法律/合规
    LEGAL_TERM = "legal_term"               # 法律条款
    CONTRACT_TERM = "contract_term"         # 合同条款
    REGULATION = "regulation"               # 法规
    COMPLIANCE = "compliance"               # 合规
    LICENSE = "license"                     # 许可证
    INTELLECTUAL_PROPERTY = "intellectual_property"  # 知识产权
    
    # 战略/运营
    STRATEGY = "strategy"                   # 战略
    BUSINESS_MODEL = "business_model"      # 商业模式
    GOAL = "goal"                           # 目标
    MILESTONE = "milestone"                 # 里程碑
    KPI = "kpi"                             # 关键绩效指标
    OKR = "okr"                             # 目标与关键成果
    RISK = "risk"                           # 风险
    OPPORTUNITY = "opportunity"             # 机会
    CHALLENGE = "challenge"                # 挑战
    
    # 项目/产品开发
    PROJECT = "project"                     # 项目
    PRODUCT_ROADMAP = "product_roadmap"   # 产品路线图
    VERSION = "version"                     # 版本
    RELEASE = "release"                     # 发布
    FEATURE_REQUEST = "feature_request"    # 功能需求
    BUG = "bug"                             # Bug
    
    # 地理位置
    LOCATION = "location"                   # 地点
    COUNTRY = "country"                     # 国家
    REGION = "region"                       # 地区
    CITY = "city"                          # 城市
    ADDRESS = "address"                     # 地址
    OFFICE = "office"                       # 办公室
    
    # 时间相关
    EVENT = "event"                         # 事件
    MILESTONE_DATE = "milestone_date"      # 里程碑日期
    QUARTER = "quarter"                     # 季度
    FISCAL_YEAR = "fiscal_year"             # 财年
    
    # 内容/创意
    CHAPTER = "chapter"                     # 章节
    SCENE = "scene"                         # 场景
    CHARACTER = "character"                # 角色
    PLOT = "plot"                           # 情节
    THEME = "theme"                         # 主题
    QUOTE = "quote"                         # 引用
    REFERENCE = "reference"                 # 引用
    
    # 通用
    METRIC = "metric"                       # 指标
    PERCENTAGE = "percentage"               # 百分比
    CURRENCY = "currency"                   # 货币
    DATE = "date"                          # 日期
    TIME = "time"                          # 时间
    DURATION = "duration"                   # 持续时间
    NUMBER = "number"                       # 数字
    URL = "url"                             # 网址
    EMAIL_ADDRESS = "email_address"         # 邮箱
    PHONE = "phone"                         # 电话
    DOCUMENT_REF = "document_ref"           # 文档引用


class RichRelationType(str, Enum):
    """丰富的关系类型"""
    
    # 组织关系
    OWNS = "owns"                           # 拥有
    ACQUIRES = "acquires"                   # 收购
    MERGES_WITH = "merges_with"             # 合并
    PART_OF = "part_of"                     # 属于
    HEADQUARTERS_IN = "headquarters_in"     # 总部位于
    
    # 人物关系
    FOUNDED_BY = "founded_by"               # 由...创立
    FOUNDED_WITH = "founded_with"           # 与...共同创立
    WORKS_AT = "works_at"                   # 在...工作
    WORKED_AT = "worked_at"                 # 曾就职于
    LEADS = "leads"                         # 领导
    REPORTS_TO = "reports_to"               # 向...汇报
    HIRES = "hires"                         # 雇佣
    FIRED_BY = "fired_by"                   # 被...解雇
    
    # 投资关系
    INVESTS_IN = "invests_in"               # 投资
    FUNDED_BY = "funded_by"                 # 由...资助
    ACQUIRED_BY = "acquired_by"             # 被...收购
    ACQUIRED = "acquired"                   # 已收购
    MERGED_WITH = "merged_with"             # 已合并
    
    # 商业关系
    COMPETES_WITH = "competes_with"         # 与...竞争
    PARTNERS_WITH = "partners_with"        # 与...合作
    SUPPLIES_TO = "supplies_to"             # 供应给
    CUSTOMER_OF = "customer_of"             # 是...的客户
    VENDOR_OF = "vendor_of"                 # 是...的供应商
    DISTRIBUTES = "distributes"             # 分销
    
    # 产品关系
    OFFERS = "offers"                       # 提供
    INCLUDES = "includes"                   # 包含
    REPLACES = "replaces"                   # 取代
    COMPARES_TO = "compares_to"             # 与...比较
    INTEGRATES_WITH = "integrates_with"    # 与...集成
    
    # 市场关系
    TARGETS = "targets"                     # 目标
    SERVES = "serves"                       # 服务于
    DOMINATES = "dominates"                  # 主导
    ENTERS = "enters"                       # 进入
    
    # 财务关系
    GENERATES = "generates"                 # 生成
    DEPENDS_ON = "depends_on"               # 取决于
    COSTS = "costs"                         # 花费
    PROFITS = "profits"                     # 盈利
    VALUED_AT = "valued_at"                 # 估值
    
    # 技术关系
    BUILT_ON = "built_on"                   # 基于
    USES = "uses"                           # 使用
    DEPRECATED_BY = "deprecated_by"         # 被...弃用
    SUPPORTS = "supports"                   # 支持
    REQUIRES = "requires"                   # 需要
    
    # 时间关系
    HAPPENED_BEFORE = "happened_before"     # 发生在...之前
    HAPPENED_AFTER = "happened_after"       # 发生在...之后
    HAPPENED_DURING = "happened_during"     # 发生在...期间
    SCHEDULED_FOR = "scheduled_for"         # 计划在
    
    # 因果关系
    CAUSES = "causes"                       # 导致
    PREVENTS = "prevents"                   # 防止
    ENABLES = "enables"                     # 使能
    BLOCKS = "blocks"                        # 阻碍
    
    # 语义关系
    DEFINES = "defines"                     # 定义
    DESCRIBES = "describes"                 # 描述
    MENTIONS = "mentions"                   # 提及
    RELATED_TO = "related_to"               # 相关于
    SIMILAR_TO = "similar_to"              # 类似
    CONTRADICTS = "contradicts"             # 矛盾
    
    # 内容关系
    WRITTEN_BY = "written_by"               # 由...撰写
    PUBLISHED_BY = "published_by"          # 由...发布
    CITES = "cites"                         # 引用
    REFERENCES = "references"               # 参考
    FOLLOWS = "follows"                     # 跟随
    PREFACES = "prefaces"                   # 为...作序


class Language(str, Enum):
    """语言"""
    ZH_CN = "zh_cn"        # 简体中文
    ZH_TW = "zh_tw"        # 繁体中文
    EN_US = "en_us"        # 英语
    JA_JP = "ja_JP"        # 日语
    KO_KR = "ko_kr"        # 韩语
    OTHER = "other"


class DocumentStatus(str, Enum):
    """文档状态"""
    DRAFT = "draft"           # 草稿
    REVIEW = "review"         # 审核中
    PUBLISHED = "published"  # 已发布
    ARCHIVED = "archived"    # 已归档
    EXPIRED = "expired"      # 已过期


class ConfidentialityLevel(str, Enum):
    """保密级别"""
    PUBLIC = "public"             # 公开
    INTERNAL = "internal"         # 内部
    CONFIDENTIAL = "confidential" # 机密
    SECRET = "secret"             # 绝密


# ============================================================================
# 基础模型
# ============================================================================

class DocumentMetadata(BaseModel):
    """文档元数据 - 基础版"""
    doc_id: str
    title: str
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    file_path: str
    file_type: str
    file_size: Optional[int] = None           # 文件大小(字节)
    page_count: Optional[int] = None
    word_count: Optional[int] = None           # 字数
    language: Language = Language.ZH_CN
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    # 文档分类
    doc_type: DocumentType = DocumentType.GENERIC
    status: DocumentStatus = DocumentStatus.DRAFT
    confidentiality: ConfidentialityLevel = ConfidentialityLevel.INTERNAL
    
    # 版本管理
    version: str = "1.0"
    previous_version: Optional[str] = None
    
    # 来源信息
    source_url: Optional[str] = None
    original_author: Optional[str] = None
    license: Optional[str] = None
    
    # 额外字段
    extra: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 商业计划书专用元数据
# ============================================================================

class BusinessBPMetadata(DocumentMetadata):
    """商业计划书元数据"""
    doc_type: DocumentType = DocumentType.BUSINESS_BP
    
    # 公司基本信息
    company_name: Optional[str] = None
    company_description: Optional[str] = None
    founded_date: Optional[datetime] = None
    headquarters: Optional[str] = None
    employee_count: Optional[str] = None      # 如 "10-50人"
    
    # 商业信息
    business_stage: Optional[str] = None      # 发展阶段: 种子/天使/A轮/B轮...
    business_model: Optional[str] = None      # 商业模式
    revenue_model: Optional[str] = None       # 盈利模式
    target_market: Optional[str] = None        # 目标市场
    market_size: Optional[str] = None         # 市场规模
    market_growth_rate: Optional[str] = None   # 市场增长率
    
    # 融资信息
    funding_stage: Optional[str] = None       # 融资阶段
    funding_amount: Optional[str] = None       # 融资金额
    funding_purpose: List[str] = Field(default_factory=list)  # 资金用途
    expected_roi: Optional[str] = None        # 预期回报
    
    # 团队信息
    team_size: Optional[int] = None
    key_team_members: List[str] = Field(default_factory=list)
    
    # 财务摘要
    revenue: Optional[str] = None
    profit: Optional[str] = None
    growth_rate: Optional[str] = None
    
    # 竞争优势
    competitive_advantage: Optional[str] = None
    barriers_to_entry: List[str] = Field(default_factory=list)


class InvestmentMemoMetadata(DocumentMetadata):
    """投资备忘录元数据"""
    doc_type: DocumentType = DocumentType.INVESTMENT_MEMO
    
    deal_name: Optional[str] = None
    deal_type: Optional[str] = None           # 投资类型: 股权/债权/可转债...
    sector: Optional[str] = None              # 行业
    investment_thesis: Optional[str] = None    # 投资论点
    risk_assessment: Optional[str] = None     # 风险评估
    recommendation: Optional[str] = None       # 建议
    target_return: Optional[str] = None        # 目标回报
    irr_expected: Optional[str] = None        # 预期IRR
    moic_expected: Optional[str] = None        # 预期MOIC


class MarketReportMetadata(DocumentMetadata):
    """市场报告元数据"""
    doc_type: DocumentType = DocumentType.MARKET_REPORT
    
    report_period: Optional[str] = None       # 报告周期
    geography: Optional[str] = None           # 地理范围
    market_size: Optional[str] = None
    market_value: Optional[str] = None
    cagr: Optional[str] = None                 # 复合年增长率
    key_players: List[str] = Field(default_factory=list)
    trends: List[str] = Field(default_factory=list)
    forecast: Optional[str] = None


class ContractMetadata(DocumentMetadata):
    """合同元数据"""
    doc_type: DocumentType = DocumentType.CONTRACT
    
    contract_type: Optional[str] = None       # 合同类型
    contract_number: Optional[str] = None      # 合同编号
    parties: List[str] = Field(default_factory=list)  # 当事人
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    signing_date: Optional[datetime] = None
    jurisdiction: Optional[str] = None        # 管辖权
    governing_law: Optional[str] = None        # 适用法律
    total_value: Optional[str] = None          # 合同金额
    currency: Optional[str] = None
    payment_terms: Optional[str] = None
    termination_clause: Optional[str] = None  # 终止条款
    renewal_terms: Optional[str] = None        # 续期条款


class TechnicalDocMetadata(DocumentMetadata):
    """技术文档元数据"""
    doc_type: DocumentType = DocumentType.TECHNICAL_DOC
    
    software_name: Optional[str] = None
    version: Optional[str] = None
    product_type: Optional[str] = None        # 产品类型
    programming_languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    runtime_environment: Optional[str] = None
    installation_guide: Optional[str] = None
    configuration: Optional[str] = None
    api_endpoints: Optional[int] = None        # API端点数量


class NovelMetadata(DocumentMetadata):
    """小说元数据"""
    doc_type: DocumentType = DocumentType.NOVEL
    
    genre: Optional[str] = None               # 题材
    sub_genre: Optional[str] = None            # 子题材
    style: Optional[str] = None                # 风格
    perspective: Optional[str] = None          # 视角
    tone: Optional[str] = None                 # 基调
    target_audience: Optional[str] = None
    word_count: Optional[int] = None
    chapter_count: Optional[int] = None
    is_series: bool = False
    series_name: Optional[str] = None
    series_order: Optional[int] = None
    original_publication: Optional[datetime] = None
    publisher: Optional[str] = None
    isbn: Optional[str] = None


class MeetingMinutesMetadata(DocumentMetadata):
    """会议纪要元数据"""
    doc_type: DocumentType = DocumentType.MEETING_MINUTES
    
    meeting_title: Optional[str] = None
    meeting_date: Optional[datetime] = None
    location: Optional[str] = None
    attendees: List[str] = Field(default_factory=list)
    absent: List[str] = Field(default_factory=list)
    facilitator: Optional[str] = None          # 主持人
    note_taker: Optional[str] = None            # 记录人
    meeting_type: Optional[str] = None          # 会议类型
    agenda: List[str] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    next_meeting_date: Optional[datetime] = None


class EmailMetadata(DocumentMetadata):
    """邮件元数据"""
    doc_type: DocumentType = DocumentType.EMAIL
    
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = Field(default_factory=list)
    cc_addresses: List[str] = Field(default_factory=list)
    bcc_addresses: List[str] = Field(default_factory=list)
    sent_date: Optional[datetime] = None
    received_date: Optional[datetime] = None
    is_reply: bool = False
    reply_to: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)
    priority: Optional[str] = None             # 高/中/低


# ============================================================================
# 文档元数据联合类型
# ============================================================================

# 文档元数据的联合类型
DocumentMetadataUnion = Union[
    BusinessBPMetadata,
    InvestmentMemoMetadata,
    MarketReportMetadata,
    ContractMetadata,
    TechnicalDocMetadata,
    NovelMetadata,
    MeetingMinutesMetadata,
    EmailMetadata,
    DocumentMetadata,
]


# ============================================================================
# 节点模型
# ============================================================================

class SectionNode(BaseModel):
    """章节节点"""
    section_id: str
    doc_id: str
    title: str
    level: int
    hierarchy_path: str
    content: Optional[str] = None
    summary: Optional[str] = None  # 章节摘要，用于宏观检索
    order: int
    parent_section_id: Optional[str] = None
    word_count: Optional[int] = None
    # 流式解析用：章节在文件中的字符偏移（不写入 Nebula）
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class ChunkNode(BaseModel):
    """文本块节点"""
    chunk_id: str
    section_id: Optional[str] = None
    doc_id: str
    text: str
    token_count: int
    position: int
    start_char: int
    end_char: int
    embedding_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """图节点"""
    node_id: str
    node_type: str
    doc_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


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
    definition: Optional[str] = None


class ParsedDocument(BaseModel):
    """解析后的文档"""
    model_config = {"extra": "allow"}
    
    metadata: DocumentMetadataUnion
    sections: List[SectionNode] = Field(default_factory=list)
    chunks: List[ChunkNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


# ============================================================================
# 实体和关系模型 - 丰富版
# ============================================================================

class EntityNode(BaseModel):
    """实体节点"""
    entity_id: str
    name: str
    entity_type: RichEntityType
    doc_id: Optional[str] = None
    
    # 实体描述
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)  # 别名
    
    # 实体属性
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    # 时间属性
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_current: bool = True
    
    # 位置属性
    location: Optional[str] = None
    
    # 数值属性 (用于财务/指标类实体)
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    
    # 来源信息
    confidence: float = 1.0                      # 置信度
    source_chunks: List[str] = Field(default_factory=list)  # 来源chunk
    first_mentioned: Optional[datetime] = None
    
    # 状态
    is_verified: bool = False
    extra: Dict[str, Any] = Field(default_factory=dict)


class EntityMention(BaseModel):
    """实体提及 - 实体在文本中的具体出现"""
    mention_id: str
    entity_id: str
    chunk_id: str
    text: str
    start_offset: int
    end_offset: int
    context: Optional[str] = None               # 上下文


class RelationEdge(BaseModel):
    """关系边"""
    relation_id: str
    src_id: str
    dst_id: str
    relation_type: RichRelationType
    
    # 关系描述
    description: Optional[str] = None
    
    # 关系强度
    weight: float = 1.0
    
    # 时间属性
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_current: bool = True
    
    # 来源信息
    confidence: float = 1.0
    source_chunks: List[str] = Field(default_factory=list)
    
    # 额外属性
    properties: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 查询模型
# ============================================================================

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
    doc_ids: Optional[List[str]] = None
    doc_types: Optional[List[DocumentType]] = None
    entity_types: Optional[List[RichEntityType]] = None
    budget_profile: str = "medium"
    enable_lazy_enhance: bool = True
    override_query_type: Optional[str] = None
    retrieval_mode: str = "auto"


class EnhancedQueryResponse(BaseModel):
    """增强查询响应"""
    query: str
    answer: str = ""
    results: List[Dict[str, Any]] = Field(default_factory=list)
    graph_results: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    took_ms: float = 0.0
    meta: Optional["QueryMeta"] = None
    entities: List[Dict[str, Any]] = Field(default_factory=list)  # LazyEntityBuilder 实体
    relations: List[Dict[str, Any]] = Field(default_factory=list)  # LazyEntityBuilder 关系


class QueryMeta(BaseModel):
    """查询元数据"""
    query_type: str = "auto"
    retrieval_paths_used: List[str] = Field(default_factory=list)
    budget_consumed: Optional[Dict[str, Any]] = None
    latency_breakdown: Optional[Dict[str, Any]] = None
    degraded: bool = False
    degradation_reasons: List[str] = Field(default_factory=list)
    trace_id: str = ""


# ============================================================================
# 知识图谱模型
# ============================================================================

class KnowledgeGraph(BaseModel):
    """知识图谱"""
    graph_id: str
    name: str
    description: Optional[str] = None
    doc_ids: List[str] = Field(default_factory=list)
    entity_count: int = 0
    relation_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class GraphQueryRequest(BaseModel):
    """图查询请求"""
    query: str
    entity_types: Optional[List[RichEntityType]] = None
    relation_types: Optional[List[RichRelationType]] = None
    depth: int = 2
    limit: int = 50


class GraphQueryResult(BaseModel):
    """图查询结果"""
    nodes: List[EntityNode]
    edges: List[RelationEdge]
    paths: List[List[str]] = Field(default_factory=list)


# ============================================================================
# 文档解析配置
# ============================================================================

class ParsingConfig(BaseModel):
    """文档解析配置"""
    extract_entities: bool = True
    extract_relations: bool = True
    extract_concepts: bool = True
    extract_summary: bool = True
    extract_keywords: bool = True
    
    # 实体提取配置
    entity_types: Optional[List[RichEntityType]] = None  # 指定则只提取这些类型
    
    # 关系提取配置
    relation_types: Optional[List[RichRelationType]] = None
    
    # 分块配置
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # 向量配置
    generate_embeddings: bool = True
    embedding_model: Optional[str] = None
    
    # 语言检测
    detect_language: bool = True
    default_language: Language = Language.ZH_CN
