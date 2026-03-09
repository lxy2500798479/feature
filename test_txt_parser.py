#!/usr/bin/env python3
"""测试 TXT 解析器的 AI 大纲生成功能"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.txt_parser import TxtParser


def test_ai_outline():
    """测试 AI 大纲生成"""
    # 创建一个测试 TXT 文件（无结构）
    test_content = """
    这是一本关于人工智能的科普书籍。

    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它试图理解智能的本质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

    机器学习是人工智能的核心，是使计算机具有智能的根本途径。它是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。

    深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。

    自然语言处理（Natural Language Processing，简称NLP）是计算机科学领域与人工智能领域中的一个重要方向。

    计算机视觉是研究如何让计算机从图像或视频中获取高层次信息的学科。

    强化学习是机器学习中的一个领域，主要关注智能体如何在环境中采取行动以最大化累积奖励。

    人工智能在各行各业都有广泛的应用，包括医疗、金融、教育、交通等领域。

    未来，人工智能将继续发展，可能会对人类社会产生深远的影响。
    """

    # 写入测试文件
    test_file = "/Users/luoxingyao/repo/feature/test_txt_parser.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)

    # 解析
    parser = TxtParser({"structure_mode": "auto"})
    result = parser.parse(test_file)

    print(f"\n=== 测试结果 ===")
    print(f"Sections 数量: {len(result.sections)}")
    print(f"Chunks 数量: {len(result.chunks)}")

    print(f"\n--- Sections ---")
    for s in result.sections:
        print(f"  [{s.level}] {s.title} (id: {s.section_id}, path: {s.hierarchy_path})")

    print(f"\n--- Chunks (前3个) ---")
    for c in result.chunks[:3]:
        print(f"  {c.chunk_id}: {c.text[:50]}...")

    # 清理
    os.remove(test_file)
    print("\n✅ 测试完成!")


def test_with_real_file():
    """测试真实文件"""
    import glob

    # 查找 mockData 目录下的 txt 文件
    txt_files = glob.glob("/Users/luoxingyao/repo/feature/mockData/**/*.txt", recursive=True)

    if not txt_files:
        print("未找到测试 TXT 文件")
        return

    test_file = txt_files[0]
    print(f"\n使用测试文件: {test_file}")

    parser = TxtParser({"structure_mode": "auto"})
    result = parser.parse(test_file)

    print(f"\n=== 测试结果 ===")
    print(f"文件: {test_file}")
    print(f"Sections 数量: {len(result.sections)}")
    print(f"Chunks 数量: {len(result.chunks)}")

    print(f"\n--- Sections (前10个) ---")
    for s in result.sections[:10]:
        print(f"  [{s.level}] {s.title} (id: {s.section_id})")

    if len(result.sections) > 10:
        print(f"  ... 还有 {len(result.sections) - 10} 个 sections")

    print("\n✅ 测试完成!")


if __name__ == "__main__":
    import glob

    # 检查是否有真实文件
    txt_files = glob.glob("/Users/luoxingyao/repo/feature/mockData/**/*.txt", recursive=True)

    if txt_files:
        test_with_real_file()
    else:
        test_ai_outline()
