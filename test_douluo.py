#!/usr/bin/env python3
"""测试斗罗大陆章节解析"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.txt_parser import TxtParser


def test_douluo_parsing():
    """测试斗罗大陆章节解析"""
    test_file = "/Users/luoxingyao/repo/feature/mockData/唐家三少《斗罗大陆》精校全本_utf8.txt"

    parser = TxtParser({"structure_mode": "auto"})
    result = parser.parse(test_file)

    print(f"\n=== 斗罗大陆解析结果 ===")
    print(f"Sections 数量: {len(result.sections)}")
    print(f"Chunks 数量: {len(result.chunks)}")

    print(f"\n--- 前20个 Sections ---")
    for s in result.sections[:20]:
        print(f"  [{s.level}] {s.title} (id: {s.section_id}, path: {s.hierarchy_path})")

    if len(result.sections) > 20:
        print(f"\n  ... 还有 {len(result.sections) - 20} 个 sections")

    print("\n✅ 测试完成!")


if __name__ == "__main__":
    test_douluo_parsing()
