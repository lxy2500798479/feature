"""
概念图谱构建器 - 零 LLM 成本的轻量级图谱构建
"""
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import math
import os
from concurrent.futures import ThreadPoolExecutor

import spacy
import networkx as nx
from networkx.algorithms import community

from src.core.models import ChunkNode
from src.utils.logger import logger


class ConceptGraphBuilder:
    """
    概念图谱构建器
    
    核心特性：
    1. 零 LLM 成本：仅使用 NLP 工具（SpaCy）
    2. 名词短语提取：识别关键概念
    3. PMI 共现计算：衡量概念关联强度
    4. 社区检测：Leiden 算法聚类
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 配置参数
        self.window_size = self.config.get("window_size", 50)  # 共现窗口大小（词数）
        self.min_phrase_freq = self.config.get("min_phrase_freq", 3)  # 最小短语频率
        self.min_pmi = self.config.get("min_pmi", 1.0)  # 最小 PMI 阈值
        self.language = self.config.get("language", "zh")  # zh | en
        # 并行：短语提取用 nlp.pipe 的进程数；共现用线程池 worker 数
        self.n_process = self.config.get("n_process")  # None 表示自动
        self.cooccur_workers = self.config.get("cooccur_workers")  # None 表示自动
        
        # 加载 SpaCy 模型
        self.nlp = self._load_spacy_model()
        
        # 统计数据
        self.phrase_freq = Counter()  # 短语频率
        self.cooccurrence = defaultdict(int)  # 共现频率
        self.total_windows = 0  # 总窗口数
    
    def _load_spacy_model(self):
        """加载 SpaCy 模型"""
        try:
            if self.language == "zh":
                nlp = spacy.load("zh_core_web_sm")
            else:
                nlp = spacy.load("en_core_web_sm")
            
            logger.info(f"已加载 SpaCy 模型，语言: {self.language}")
            return nlp
        
        except OSError:
            logger.error(f"未找到 SpaCy 模型。请安装: python -m spacy download {'zh_core_web_sm' if self.language == 'zh' else 'en_core_web_sm'}")
            raise
    
    def build_from_chunks(self, chunks: List[ChunkNode], doc_id: str) -> Dict:
        """
        从文本块构建概念图谱
        
        Args:
            chunks: 文本块列表
            doc_id: 文档ID
            
        Returns:
            Dict: {
                "nodes": List[Dict],  # 概念节点
                "edges": List[Dict],  # 概念关系
                "communities": Dict   # 社区结构
            }
        """
        import time
        
        total_start = time.time()
        logger.info(f"为文档构建概念图谱: {doc_id} (文本块数: {len(chunks)})")
        
        # 重置统计
        self.phrase_freq.clear()
        self.cooccurrence.clear()
        self.total_windows = 0
        
        # 1. 提取名词短语
        step_start = time.time()
        texts = [c.text for c in chunks]
        n_process = self.n_process
        if n_process is None:
            n_process = min(4, (os.cpu_count() or 2))
        if n_process <= 1:
            n_process = 1
        chunk_phrases_list = []
        if n_process > 1:
            for doc in self.nlp.pipe(texts, n_process=n_process, batch_size=50):
                phrases = self._doc_to_phrases(doc)
                chunk_phrases_list.append(phrases)
                for phrase in phrases:
                    self.phrase_freq[phrase] += 1
        else:
            for text in texts:
                phrases = self._extract_noun_phrases(text)
                chunk_phrases_list.append(phrases)
                for phrase in phrases:
                    self.phrase_freq[phrase] += 1
        all_phrases = [p for phrases in chunk_phrases_list for p in phrases]
        extract_time = time.time() - step_start
        logger.info(f"名词短语提取完成 (耗时: {extract_time:.2f}秒, 唯一短语: {len(set(all_phrases))})")
        
        # 2. 计算共现关系
        step_start = time.time()
        self._compute_cooccurrence_parallel(chunk_phrases_list)
        cooccur_time = time.time() - step_start
        logger.info(f"共现计算完成 (耗时: {cooccur_time:.2f}秒, 共现对数: {len(self.cooccurrence)})")
        
        # 3. 过滤低频短语
        step_start = time.time()
        filtered_phrases = self._filter_phrases()
        filter_time = time.time() - step_start
        logger.info(f"短语过滤完成 (耗时: {filter_time:.2f}秒, 剩余: {len(filtered_phrases)} 个)")
        
        # 4. 计算 PMI 并构建图
        step_start = time.time()
        graph = self._build_graph(filtered_phrases)
        graph_time = time.time() - step_start
        logger.info(f"图构建完成 (耗时: {graph_time:.2f}秒, 节点: {graph.number_of_nodes()}, 边: {graph.number_of_edges()})")
        
        # 5. 社区检测
        step_start = time.time()
        communities = self._detect_communities(graph)
        community_time = time.time() - step_start
        logger.info(f"社区检测完成 (耗时: {community_time:.2f}秒, 社区数: {len(set(communities.values())) if communities else 0})")
        
        # 6. 生成输出
        step_start = time.time()
        result = self._generate_output(graph, communities, doc_id)
        output_time = time.time() - step_start
        
        total_time = time.time() - total_start
        logger.info(f"概念图谱构建总耗时: {total_time:.2f}秒")
        
        return result
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """提取名词短语"""
        doc = self.nlp(text)
        return self._doc_to_phrases(doc)

    def _doc_to_phrases(self, doc) -> List[str]:
        """从 SpaCy Doc 提取概念"""
        phrases = []
        seen = set()

        # 1. NER 实体
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "GPE", "ORG", "LOC", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT"):
                phrase = ent.text.strip().lower()
                if self._is_valid_phrase(phrase) and phrase not in seen:
                    seen.add(phrase)
                    phrases.append(phrase)

        # 2. 名词短语
        current_phrase = []
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                current_phrase.append(token.text)
            else:
                if current_phrase:
                    phrase = "".join(current_phrase) if self.language == "zh" else " ".join(current_phrase)
                    phrase = phrase.lower()
                    if self._is_valid_phrase(phrase) and phrase not in seen:
                        seen.add(phrase)
                        phrases.append(phrase)
                    current_phrase = []
        if current_phrase:
            phrase = "".join(current_phrase) if self.language == "zh" else " ".join(current_phrase)
            phrase = phrase.lower()
            if self._is_valid_phrase(phrase) and phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)

        return phrases
    
    _ZH_STOPWORDS: set = {
        "自己", "他们", "她们", "我们", "你们", "这个", "那个", "什么", "这些", "那些",
        "他", "她", "它", "我", "你", "其", "谁", "哪", "几",
        "时候", "地方", "方面", "之间", "以后", "之后", "以前", "之前", "里面", "外面",
        "上面", "下面", "前面", "后面", "左边", "右边", "中间", "旁边", "身边",
        "今天", "明天", "昨天", "现在", "当时", "此时", "那时", "刚才", "马上",
        "东西", "事情", "问题", "样子", "意思", "感觉", "声音", "办法", "情况", "结果",
        "过程", "原因", "目的", "机会", "条件", "程度", "可能", "部分", "方式", "方法",
        "一点", "一些", "一下", "一种", "一切", "一番", "所有", "全部",
        "对方", "那边", "这边", "那里", "这里", "身上", "手中", "心中", "眼中", "脑海",
        "的", "了", "和", "是", "在", "有", "与", "及", "等", "也", "都", "就", "而",
        "个", "只", "条", "种", "位", "名", "头", "句", "片", "道",
    }

    def _is_valid_phrase(self, phrase: str) -> bool:
        """验证短语是否有效"""
        if len(phrase) < 2:
            return False
        if phrase.isdigit():
            return False
        if len(phrase.split()) > 5:
            return False
        if phrase in self._ZH_STOPWORDS:
            return False
        if all(c in "一二三四五六七八九十百千万亿零两" for c in phrase):
            return False
        return True
    
    def _compute_cooccurrence(self, phrases: List[str]):
        """单 chunk 共现计算"""
        cooccur, tw = self._compute_cooccurrence_return(phrases)
        for k, v in cooccur.items():
            self.cooccurrence[k] += v
        self.total_windows += tw

    def _compute_cooccurrence_return(self, phrases: List[str]) -> Tuple[Dict[Tuple[str, str], int], int]:
        """计算单段文本的共现关系"""
        unique_phrases = []
        seen = set()
        for p in phrases:
            if p not in seen:
                unique_phrases.append(p)
                seen.add(p)
        local_cooccur: Dict[Tuple[str, str], int] = defaultdict(int)
        local_windows = 0
        for i in range(len(unique_phrases)):
            window_end = min(i + self.window_size, len(unique_phrases))
            for j in range(i + 1, window_end):
                phrase1 = unique_phrases[i]
                phrase2 = unique_phrases[j]
                pair = tuple(sorted([phrase1, phrase2]))
                local_cooccur[pair] += 1
            local_windows += 1
        return dict(local_cooccur), local_windows

    def _compute_cooccurrence_parallel(self, chunk_phrases_list: List[List[str]]):
        """多线程并行计算各 chunk 共现"""
        workers = self.cooccur_workers
        if workers is None:
            workers = min(8, max(1, (os.cpu_count() or 2)))
        if workers <= 1 or len(chunk_phrases_list) < 4:
            for phrases in chunk_phrases_list:
                self._compute_cooccurrence(phrases)
            return
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self._compute_cooccurrence_return, chunk_phrases_list))
        for cooccur, tw in results:
            for pair, cnt in cooccur.items():
                self.cooccurrence[pair] += cnt
            self.total_windows += tw
    
    def _filter_phrases(self) -> Set[str]:
        """过滤低频短语"""
        return {
            phrase for phrase, freq in self.phrase_freq.items()
            if freq >= self.min_phrase_freq
        }
    
    def _build_graph(self, phrases: Set[str]) -> nx.Graph:
        """构建带权概念图"""
        graph = nx.Graph()
        
        for phrase in phrases:
            graph.add_node(phrase, freq=self.phrase_freq[phrase])
        
        for (phrase1, phrase2), cooccur_count in self.cooccurrence.items():
            if phrase1 not in phrases or phrase2 not in phrases:
                continue
            
            pmi = self._calculate_pmi(phrase1, phrase2, cooccur_count)
            
            if pmi >= self.min_pmi:
                graph.add_edge(phrase1, phrase2, weight=pmi, cooccur=cooccur_count)
        
        return graph
    
    def _calculate_pmi(self, phrase1: str, phrase2: str, cooccur_count: int) -> float:
        """计算点互信息（PMI）"""
        if self.total_windows == 0:
            return 0.0
        
        p_phrase1 = self.phrase_freq[phrase1] / self.total_windows
        p_phrase2 = self.phrase_freq[phrase2] / self.total_windows
        p_cooccur = cooccur_count / self.total_windows
        
        if p_cooccur == 0 or p_phrase1 == 0 or p_phrase2 == 0:
            return 0.0
        
        pmi = math.log(p_cooccur / (p_phrase1 * p_phrase2))
        
        return pmi
    
    def _detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """社区检测：使用 Leiden 或 Louvain 算法"""
        if graph.number_of_nodes() == 0:
            return {}
        
        try:
            import leidenalg
            import igraph as ig
            
            ig_graph = self._nx_to_igraph(graph)
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition
            )
            
            communities = {}
            for i, community in enumerate(partition):
                for node_idx in community:
                    node_name = ig_graph.vs[node_idx]['name']
                    communities[node_name] = i
            
            logger.info("使用 Leiden 算法进行社区检测")
        
        except ImportError:
            logger.info("Leiden 不可用，使用 Louvain 算法")
            partition = community.louvain_communities(graph, weight='weight')
            
            communities = {}
            for i, comm in enumerate(partition):
                for node in comm:
                    communities[node] = i
        
        return communities
    
    def _nx_to_igraph(self, nx_graph: nx.Graph):
        """将 NetworkX 图转换为 igraph"""
        import igraph as ig
        
        nodes = list(nx_graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        edges = [
            (node_to_idx[u], node_to_idx[v])
            for u, v in nx_graph.edges()
        ]
        
        ig_graph = ig.Graph(n=len(nodes), edges=edges)
        ig_graph.vs['name'] = nodes
        
        weights = [nx_graph[u][v]['weight'] for u, v in nx_graph.edges()]
        ig_graph.es['weight'] = weights
        
        return ig_graph
    
    def _generate_output(self, graph: nx.Graph, communities: Dict[str, int], doc_id: str) -> Dict:
        """生成输出数据结构"""
        nodes = []
        for node in graph.nodes():
            node_id = f"{doc_id}_concept_{hash(node) % 1000000}"
            nodes.append({
                "id": node_id,
                "phrase": node,
                "freq": graph.nodes[node].get('freq', 0),
                "community": communities.get(node, -1)
            })
        
        edges = []
        node_to_id = {n['phrase']: n['id'] for n in nodes}
        
        for u, v, data in graph.edges(data=True):
            edges.append({
                "from": node_to_id[u],
                "to": node_to_id[v],
                "weight": data.get('weight', 0.0),
                "cooccur": data.get('cooccur', 0)
            })
        
        community_structure = defaultdict(list)
        for node in nodes:
            community_structure[node['community']].append(node['id'])
        
        return {
            "nodes": nodes,
            "edges": edges,
            "communities": dict(community_structure),
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "total_communities": len(community_structure),
                "avg_degree": 2 * len(edges) / len(nodes) if len(nodes) > 0 else 0
            }
        }
