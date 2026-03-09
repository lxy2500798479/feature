#!/usr/bin/env python3
"""快速测试 AI Outline Service"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.outline_service import AIOutlineService


def test_ai_outline():
    """测试 AI 大纲生成"""
    service = AIOutlineService()

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

    print("开始测试 AI Outline Service...")
    sections = service.generate_outline(test_content, "test_doc")

    print(f"\n=== 测试结果 ===")
    print(f"生成了 {len(sections)} 个章节")

    for s in sections:
        print(f"  [{s.level}] {s.title} (id: {s.section_id})")


if __name__ == "__main__":
    test_ai_outline()
