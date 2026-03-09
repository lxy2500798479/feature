#!/usr/bin/env python3
"""测试章节识别 - 检查为什么斗罗大陆被识别为有章节"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.txt_parser import TxtParser


def test_heading_detection():
    """测试章节检测"""
    parser = TxtParser({"structure_mode": "auto"})

    test_content = """
    斗罗大陆 第一章 起点
    这是一个斗罗大陆的故事。
    唐三和小舞在这片大陆上成长。
    
    2. 这是第二章
    内容内容内容
    
    3. 第三章
    更多内容
    
    第三章 新的章节
    这也是章节
    
    第十章 大结局
    故事的结尾
    """

    result = parser._has_headings(test_content)
    print(f"检测结果: {result}")

    # 打印检测到的标题
    import re

    heading_patterns = [
        r"^第[一二三四五六七八九十百千\d]+[章节卷部篇集]",
        r"^#{1,6}\s+",
        r"^\d+\.\s+",
        r"^\d+\.\d+\s+",
        r"^\d+\.\d+\.\d+\s+",
    ]

    lines = test_content.split("\n")
    print("\n检测到的标题:")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for pattern in heading_patterns:
            if re.match(pattern, line):
                print(f"  ✓ {line}")
                break


if __name__ == "__main__":
    test_heading_detection()
