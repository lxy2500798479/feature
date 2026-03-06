"""
文本分块器 - 支持多种分块策略
"""
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.utils.logger import logger


@dataclass
class TextChunk:
    """文本块数据类"""
    text: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict = None


class BaseChunker(ABC):
    """分块器基类"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    @abstractmethod
    def chunk(self, text: str) -> List[TextChunk]:
        """分块方法"""
        pass


class FixedSizeChunker(BaseChunker):
    """
    固定大小分块器
    
    策略：按固定字符数或 token 数分块
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 512)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        self.unit = self.config.get("unit", "char")  # char | token
        
        if self.unit == "token":
            self._init_tokenizer()
    
    def _init_tokenizer(self):
        """初始化 tokenizer"""
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not installed, falling back to simple split")
            self.tokenizer = None
    
    def chunk(self, text: str) -> List[TextChunk]:
        """固定大小分块"""
        if self.unit == "token" and self.tokenizer:
            return self._chunk_by_tokens(text)
        else:
            return self._chunk_by_chars(text)
    
    def _chunk_by_chars(self, text: str) -> List[TextChunk]:
        """按字符数分块"""
        chunks = []
        position = 0
        
        while position < len(text):
            end = min(position + self.chunk_size, len(text))
            chunk_text = text[position:end]
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=position,
                end_char=end,
                token_count=len(chunk_text.split())
            ))
            
            position += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _chunk_by_tokens(self, text: str) -> List[TextChunk]:
        """按 token 数分块"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        position = 0
        char_position = 0
        
        while position < len(tokens):
            end = min(position + self.chunk_size, len(tokens))
            chunk_tokens = tokens[position:end]
            
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # 计算字符位置
            char_start = char_position
            char_end = char_start + len(chunk_text)
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=char_start,
                end_char=char_end,
                token_count=len(chunk_tokens)
            ))
            
            # 计算下一个位置
            if position + self.chunk_size - self.chunk_overlap >= len(tokens):
                break
            
            overlap_tokens = tokens[position:end][:self.chunk_overlap] if len(tokens[position:end]) > self.chunk_overlap else tokens[position:end]
            overlap_text = self.tokenizer.decode(overlap_tokens)
            char_position += len(overlap_text)
            position += self.chunk_size - self.chunk_overlap
        
        return chunks


class SlidingWindowChunker(BaseChunker):
    """
    滑动窗口分块器
    
    策略：使用滑动窗口，窗口之间有重叠
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.window_size = self.config.get("window_size", 512)
        self.step_size = self.config.get("step_size", 256)
        self.unit = self.config.get("unit", "char")
        
        if self.unit == "token":
            self._init_tokenizer()
    
    def _init_tokenizer(self):
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not installed")
            self.tokenizer = None
    
    def chunk(self, text: str) -> List[TextChunk]:
        if self.unit == "token" and self.tokenizer:
            return self._chunk_by_tokens(text)
        return self._chunk_by_chars(text)
    
    def _chunk_by_chars(self, text: str) -> List[TextChunk]:
        chunks = []
        position = 0
        idx = 0
        
        while position < len(text):
            end = min(position + self.window_size, len(text))
            chunk_text = text[position:end]
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=position,
                end_char=end,
                token_count=len(chunk_text.split()),
                metadata={"window_id": idx}
            ))
            
            position += self.step_size
            idx += 1
        
        return chunks
    
    def _chunk_by_tokens(self, text: str) -> List[TextChunk]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        position = 0
        idx = 0
        
        while position < len(tokens):
            end = min(position + self.window_size, len(tokens))
            chunk_tokens = tokens[position:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=position,
                end_char=end,
                token_count=len(chunk_tokens),
                metadata={"window_id": idx}
            ))
            
            position += self.step_size
            idx += 1
        
        return chunks


class ParagraphChunker(BaseChunker):
    """
    段落分块器
    
    策略：按段落分割，保留段落边界
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.min_paragraph_size = self.config.get("min_paragraph_size", 50)
    
    def chunk(self, text: str) -> List[TextChunk]:
        # 按换行符分割段落
        paragraphs = text.split("\n")
        
        chunks = []
        position = 0
        idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(para) < self.min_paragraph_size:
                continue
            
            chunk = TextChunk(
                text=para,
                start_char=position,
                end_char=position + len(para),
                token_count=len(para.split()),
                metadata={"paragraph_id": idx}
            )
            chunks.append(chunk)
            
            position += len(para) + 1  # +1 for newline
            idx += 1
        
        return chunks


class SemanticChunker(BaseChunker):
    """
    语义分块器
    
    策略：基于句子边界分块
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.max_chunk_size = self.config.get("max_chunk_size", 1024)
        self.sentence_endings = self.config.get("sentence_endings", "。！？.!?")
    
    def chunk(self, text: str) -> List[TextChunk]:
        import re
        
        # 按句子结束符分割
        pattern = f"[{self.sentence_endings}]+"
        sentences = re.split(pattern, text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        position = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # 如果单个句子超过最大大小，进一步分割
            if sentence_size > self.max_chunk_size:
                # 先保存当前累积的块
                if current_chunk:
                    chunk_text = "".join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        start_char=position - len(chunk_text),
                        end_char=position,
                        token_count=len(chunk_text.split()),
                        metadata={"chunk_id": chunk_idx}
                    ))
                    chunk_idx += 1
                    current_chunk = []
                    current_size = 0
                
                # 对超长句子按固定大小分割
                sub_chunks = self._split_long_sentence(sentence, position)
                chunks.extend(sub_chunks)
                position += sentence_size + 1
                continue
            
            # 如果加上这个句子会超过最大大小，先保存当前块
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = "".join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_char=position - len(chunk_text),
                    end_char=position,
                    token_count=len(chunk_text.split()),
                    metadata={"chunk_id": chunk_idx}
                ))
                chunk_idx += 1
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1
            position += sentence_size + 1
        
        # 保存最后一个块
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=position - len(chunk_text),
                end_char=position,
                token_count=len(chunk_text.split()),
                metadata={"chunk_id": chunk_idx}
            ))
        
        return chunks
    
    def _split_long_sentence(self, text: str, start_pos: int) -> List[TextChunk]:
        """分割长句子"""
        chunks = []
        position = start_pos
        idx = 0
        
        while position < start_pos + len(text):
            end = min(position + self.max_chunk_size, start_pos + len(text))
            chunk_text = text[position - start_pos:end - start_pos]
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=position,
                end_char=end,
                token_count=len(chunk_text.split()),
                metadata={"chunk_id": idx, "split": True}
            ))
            
            position = end
            idx += 1
        
        return chunks


class ChunkerFactory:
    """分块器工厂"""
    
    _chunkers = {
        "fixed": FixedSizeChunker,
        "sliding": SlidingWindowChunker,
        "paragraph": ParagraphChunker,
        "semantic": SemanticChunker,
    }
    
    @classmethod
    def create(cls, strategy: str = "fixed", config: Optional[Dict] = None) -> BaseChunker:
        """创建分块器"""
        if strategy not in cls._chunkers:
            logger.warning(f"未知的分块策略: {strategy}，使用 fixed")
            strategy = "fixed"
        
        return cls._chunkers[strategy](config)
    
    @classmethod
    def register(cls, name: str, chunker_class: type):
        """注册新的分块器"""
        cls._chunkers[name] = chunker_class
