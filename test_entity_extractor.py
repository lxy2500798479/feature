"""
Entity Extractor 测试脚本
"""
import json
import re
from src.config import settings
from src.utils.llm_client import LLMClient


_EXTRACT_PROMPT = """直接从下面文本中提取实体和关系，输出纯JSON，不要其他内容：

文本：{text}

输出格式：{{"entities":[{"name":"名","type":"类型","description":"描述"}],"relations":[{"src":"A","dst":"B","type":"关系","strength":0.9}]}}"""


def _try_extract_json(text: str) -> dict:
    """从任意文本中提取 JSON（健壮提取）"""
    import re
    # 去掉 markdown 代码块、编号列表头
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text)
    text = re.sub(r"^\d+\.\s+\*\*.*?\*\*:?", "", text, flags=re.MULTILINE)
    text = text.strip()

    # 1. 直接解析整段
    try:
        parsed = json.loads(text)
        if "entities" in parsed:
            return parsed
    except:
        pass

    # 2. 逐行找 {"name": ...} 或 {"src": ...} 格式
    entities = []
    relations = []
    for line in text.split('\n'):
        line = line.strip().lstrip('*-').strip()
        # 去掉前缀如 "Entities:"
        line = re.sub(r"^(Entities|Relations|Entity|Relation)\s*:?\s*", "", line, flags=re.IGNORECASE)
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "name" in obj:
                entities.append(obj)
            elif "src" in obj and "dst" in obj:
                relations.append(obj)
        except:
            pass

    if entities or relations:
        return {"entities": entities, "relations": relations}

    return {"entities": [], "relations": []}


def test_entity_extraction():
    # 初始化 LLM
    llm = LLMClient(
        api_url=settings.SUMMARY_API_URL,
        api_key=settings.SUMMARY_API_KEY,
        model=settings.SUMMARY_MODEL,
        timeout=300,
    )

    # 测试文本
    test_text = """
    雷军，1969年出生于湖北仙桃，小米科技创始人兼CEO。2010年创立小米公司，
    先后推出了小米手机、小米电视等智能硬件产品。2021年小米宣布进军智能电动汽车领域，
    2024年发布了首款量产车型SU7，对标特斯拉Model S。
    """

    prompt = _EXTRACT_PROMPT.replace("{text}", test_text)

    print("=" * 60)
    print("发送的 Prompt:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)

    print("\n正在调用 LLM...")
    raw = llm.chat(
        prompt=prompt,
        system_prompt="你是一个严格的信息抽取助手，只输出 JSON。",
        max_tokens=1024,
        temperature=0.0,
        no_think=True,
    )

    print("\n" + "=" * 60)
    print("LLM 原始返回:")
    print("=" * 60)
    print(repr(raw))
    print("=" * 60)

    # 用新的健壮提取
    print("\n尝试健壮提取 JSON...")
    parsed = _try_extract_json(raw)
    if parsed.get("entities") or parsed.get("relations"):
        print("✅ 提取成功!")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        print("❌ 未能提取到实体关系")


if __name__ == "__main__":
    test_entity_extraction()
