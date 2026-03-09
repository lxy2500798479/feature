"""
远程嵌入服务客户端 - BGE-M3
"""
import requests
from typing import List, Optional
import numpy as np

from src.config import settings
from src.utils.logger import logger


class RemoteEmbedder:
    """远程嵌入服务客户端"""
    
    def __init__(self, service_url: Optional[str] = None):
        self.service_url = service_url or settings.EMBEDDING_SERVICE_URL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        
        if not self.service_url:
            raise ValueError("EMBEDDING_SERVICE_URL not configured")
        
        logger.info(f"已初始化 RemoteEmbedder: {self.service_url}")
    
    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> np.ndarray:
        """批量生成文本嵌入"""
        import time
        
        if not texts:
            return np.array([])
        
        # 仅在批量处理时打印详细日志
        if len(texts) > 1:
            logger.info(f"🌐 通过远程服务为 {len(texts)} 个文本生成嵌入...")
        
        for attempt in range(max_retries):
            try:
                api_start = time.time()
                response = requests.post(
                    f"{self.service_url}/embeddings",
                    json={"texts": texts, "normalize": normalize},
                    timeout=300
                )
                response.raise_for_status()
                api_time = time.time() - api_start
                
                result = response.json()
                embeddings = np.array(result.get("embeddings", []))
                
                if embeddings.shape[0] != len(texts):
                    logger.error(f"嵌入数量不匹配: 期望 {len(texts)}，实际 {embeddings.shape[0]}")
                    return np.zeros((len(texts), self.dimension))
                
                # 仅在批量处理时打印成功日志
                if len(texts) > 1:
                    logger.info(f"✅ 远程嵌入成功: API调用 {api_time:.2f}秒")
                return embeddings
            
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ 远程嵌入服务失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"❌ 远程嵌入服务在 {max_retries} 次尝试后失败: {e}")
                    return np.zeros((len(texts), self.dimension))
        
        return np.zeros((len(texts), self.dimension))
    
    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """生成单个文本的嵌入"""
        embeddings = self.embed_texts([text], normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.dimension)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """分批生成嵌入"""
        import time
        
        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.batch_size
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"📦 分批生成嵌入: {len(texts)} 个文本, {num_batches} 个批次")
        
        all_embeddings = []
        total_start = time.time()
        last_log_time = total_start
        log_interval_sec = 15
        
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i:i + batch_size]
            
            batch_start = time.time()
            embeddings = self.embed_texts(batch, normalize=normalize)
            batch_time = time.time() - batch_start
            
            all_embeddings.append(embeddings)
            
            now = time.time()
            if now - last_log_time >= log_interval_sec or batch_num == num_batches:
                elapsed = now - total_start
                pct = 100.0 * batch_num / num_batches
                logger.info(f"  ├─ 嵌入进度: {batch_num}/{num_batches} ({pct:.0f}%), 已耗时 {elapsed:.1f}秒")
                last_log_time = now
        
        total_time = time.time() - total_start
        logger.info(f"  └─ 所有批次完成: 总耗时 {total_time:.2f}秒")
        
        return np.vstack(all_embeddings)
