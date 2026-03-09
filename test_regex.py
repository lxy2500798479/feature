#!/usr/bin/env python3
"""测试章节解析"""
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """测试章节解析"""
    test_file = "/Users/luoxingyao/repo/feature/mockData/唐家三少《斗罗大陆》精校全本_utf8.txt"

    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()

    print("=== 按行显示前50行 ===")
    lines = content.split("\n")
    for i, line in enumerate(lines[:50]):
        print(f"{i}: {repr(line[:80])}")

    print("\n=== 测试正则匹配 ===")
    heading_patterns = [
        (r"^#{1,6}\s+", "Markdown"),
        (r"^第[一二三四五六七八九十百千\d]+[章节卷部篇集]\s", "章节卷部篇集+空格"),
        (r"^第[一二三四五六七八九十百千\d]+章\s", "章+空格"),
        (r"^第[一二三四五六七八九十]+[篇部集]\s", "篇部集+空格"),
        (r"^\d+\.\d+\.\d+", "1.1.1"),
        (r"^\d+\.\d+", "1.1"),
    ]

    for i, line in enumerate(lines[:50]):
        line = line.strip()
        if not line:
            continue
        for pattern, name in heading_patterns:
            if re.match(pattern, line):
                print(f"  ✓ [{name}] {line[:50]}")
                break


if __name__ == "__main__":
    main()
