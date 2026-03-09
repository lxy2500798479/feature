#!/usr/bin/env python3
"""测试章节解析 - 检查斗罗大陆是否被正确识别"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.txt_parser import TxtParser


def test_detection():
    """测试检测"""
    parser = TxtParser({"structure_mode": "auto"})

    test_file = "/Users/luoxingyao/repo/feature/mockData/唐家三少《斗罗大陆》精校全本_utf8.txt"

    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"测试 _has_headings:")
    result = parser._has_headings(content)
    print(f"  结果: {result}")

    # 打印检测到的标题
    import re
    heading_patterns = [
        r"^#{1,6}\s+",
        r"^第[一二三四五六七八九十百千\d]+[章节卷部篇集]\s",
        r"^第[一二三四五六七八九十百千\d]+章\s",
        r"^第[一二三四五六七八九十]+[篇部集]\s",
        r"^\d+\.\d+\.\d+",
        r"^\d+\.\d+",
        r"^[①②③④⑤⑥⑦⑧⑨⑩]",
    ]

    lines = content.split("\n")
    print("\n检测到的标题（前20个）:")
    heading_count = 0
    for line in lines[:200]:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        for pattern in heading_patterns:
            if re.match(pattern, line_stripped):
                print(f"  ✓ {line_stripped[:50]}")
                heading_count += 1
                if heading_count >= 20:
                    break
        if heading_count >= 20:
            break


if __name__ == "__main__":
    test_detection()
