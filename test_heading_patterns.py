#!/usr/bin/env python3
"""测试章节解析"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.txt_parser import TxtParser


def test_heading_patterns():
    """测试章节解析"""
    parser = TxtParser({"structure_mode": "auto"})

    # 斗罗大陆的内容
    content = """斗罗大陆
作者：唐家三少


内容简介：
　　唐门外门弟子唐三，因偷学内门绝学为唐门所不容...


第一卷 斗罗世界


引子 穿越的唐家三少
　　巴蜀，历来有天府之国的美誉...
"""

    print("测试 _has_headings:")
    result = parser._has_headings(content)
    print(f"  结果: {result}")

    print("\n测试 _parse_headings:")
    sections = parser._parse_headings(content, "test_doc")
    print(f"  解析到 {len(sections)} 个章节")

    for s in sections:
        print(f"    [{s.level}] {s.title}")


if __name__ == "__main__":
    test_heading_patterns()
