#!/usr/bin/env python3
"""测试章节解析 - 打印斗罗大陆的前几百个字符"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """打印文件内容"""
    test_file = "/Users/luoxingyao/repo/feature/mockData/唐家三少《斗罗大陆》精校全本_utf8.txt"

    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()

    print("=== 前500字符 ===")
    print(repr(content[:500]))

    print("\n=== 按行显示前30行 ===")
    lines = content.split("\n")
    for i, line in enumerate(lines[:30]):
        print(f"{i}: {repr(line[:60])}")


if __name__ == "__main__":
    main()
