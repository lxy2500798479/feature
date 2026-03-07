"""
LLM 客户端
"""
from typing import Optional, Dict, Any, List, Generator
import re
import requests
from src.config import settings
from src.utils.logger import logger


def _extract_from_thinking_process(text: str) -> str:
    """从 'Thinking Process:...' 格式中提取最终答案。

    qwen3.5-27b 等模型不使用 <think> XML 标签，而是以 'Thinking Process:' 开头
    用英文写出分步推理，最后给出中文/目标语言的答案。
    策略：查找 Final Answer / Summary / Output 等常见标记；
    若无，则提取最后一个连续非英文（中文）段落。
    """
    # 常见英文结束标记后面跟着正文
    for marker in (
        "**Final Answer:**",
        "**Final Answer**:",
        "**Final Summary:**",
        "**Summary:**",
        "**Output:**",
        "**Answer:**",
        "Final Answer:",
        "Final Summary:",
        "Summary:",
        "Output:",
    ):
        if marker in text:
            after = text.split(marker, 1)[1].strip()
            if after:
                return after

    # 没有标记时：找最后一个中文主导段落（思考是英文，答案通常是中文）
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in reversed(paragraphs):
        # 统计中文字符比例
        zh_count = sum(1 for c in para if "\u4e00" <= c <= "\u9fff")
        if zh_count > 10 or (zh_count > 0 and zh_count / max(len(para), 1) > 0.3):
            return para

    return ""


def _extract_content(message: dict) -> str:
    """从 chat completion message 中提取文本内容。

    兼容多种推理模型输出格式：
    1. 普通模型：message["content"] 直接是答案
    2. qwen3 + reasoning-parser：content=None, reasoning 含 <think>...</think> + 答案
    3. qwen3.5-27b 直接输出：content 以 "Thinking Process:" 开头，后面是分步英文推理 + 中文答案
    4. deepseek-r1 等：content 含 <think>...</think>答案
    """
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""

    if content:
        # 处理 <think>...</think> XML 格式
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if cleaned:
            # 处理 qwen3.5-27b 的 "Thinking Process:" 英文推理格式
            if cleaned.startswith("Thinking Process:"):
                answer = _extract_from_thinking_process(cleaned)
                if answer:
                    return answer
                # 推理未完成（max_tokens 截断），返回空让调用方降级
                logger.warning("qwen3.5 thinking 未完成（max_tokens 不足），答案提取失败")
                return ""
            return cleaned

    # content 为空时，从 reasoning 字段提取（reasoning-parser 场景）
    if reasoning:
        if "</think>" in reasoning:
            after = reasoning.split("</think>", 1)[1].strip()
            if after:
                return after
        logger.warning("推理模型 reasoning 未完成（max_tokens 不足），返回空内容")
        return ""

    return ""


class LLMClient:
    """LLM API 客户端（兼容普通模型和 qwen3/deepseek-r1 推理模型）"""
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60
    ):
        self.api_url = api_url or settings.LLM_API_URL
        self.api_key = api_key or settings.LLM_API_KEY
        self.model = model or settings.LLM_MODEL or settings.SUMMARY_MODEL
        self.timeout = timeout
    
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        no_think: bool = False,
        **kwargs
    ) -> str:
        """发送聊天请求
        
        Args:
            no_think: True 时向 qwen3 等推理模型发送禁用 thinking 的指令，
                      减少 token 消耗（摘要生成等不需要推理的场景）
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # qwen3 禁用思考模式：在用户消息前加 /no_think 标记
        actual_prompt = prompt
        if no_think:
            actual_prompt = "/no_think\n" + prompt

        messages.append({"role": "user", "content": actual_prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return _extract_content(result["choices"][0]["message"])
            
            return ""
        
        except Exception as e:
            logger.error(f"LLM 请求失败: {e}")
            return ""
    
    def chat_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        no_think: bool = False,
    ) -> Generator[str, None, None]:
        """流式聊天请求（自动过滤 <think> 推理块）"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        actual_prompt = prompt
        if no_think:
            actual_prompt = "/no_think\n" + prompt

        messages.append({"role": "user", "content": actual_prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            in_think = False
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        import json
                        try:
                            chunk = json.loads(data)
                            if "choices" not in chunk or not chunk["choices"]:
                                continue
                            delta = chunk["choices"][0].get("delta", {})
                            token = delta.get("content") or ""
                            if not token:
                                continue
                            # 过滤 <think> 推理块
                            if "<think>" in token:
                                in_think = True
                            if in_think:
                                if "</think>" in token:
                                    in_think = False
                                    after = token.split("</think>", 1)[1]
                                    if after:
                                        yield after
                                continue
                            yield token
                        except:
                            continue
        
        except Exception as e:
            logger.error(f"LLM 流式请求失败: {e}")
            yield ""
