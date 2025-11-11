#!/usr/bin/env python3
"""
MemCube政治理论概念图扩增系统 - 数据库版快速启动脚本
支持图数据库和向量数据库的完整工作流程
"""

import sys
import os
import logging
from pathlib import Path
import yaml
import json

# 添加src目录到路径
sys.path.append('src')

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/memcube_database.log', encoding='utf-8')
        ]
    )

def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("MemCube 数据库版环境检查")
    print("=" * 60)

    # 检查数据文件
    data_files = [
        "data/political_seed_concepts.txt",
        "data/political_seed_concepts.json",
        "data/transformed_political_data.json"
    ]

    for file_path in data_files:
        if Path(file_path).exists():
            print(f"[OK] {file_path} 存在")
        else:
            print(f"[FAIL] {file_path} 不存在，请先运行数据准备脚本")
            return False

    # 检查配置文件
    config_files = ["config/config.yaml", "config/api_keys.yaml"]
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"[OK] {file_path} 存在")
        else:
            print(f"[FAIL] {file_path} 不存在")
            return False

    return True

def check_database_availability():
    """检查数据库可用性"""
    print("\n" + "=" * 60)
    print("数据库连接检查")
    print("=" * 60)

    try:
        # 加载配置
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 检查图数据库
        graph_config = config.get('graph_database', {})
        if graph_config.get('enabled', False):
            try:
                from graph_database_client import get_graph_client
                graph_client = get_graph_client()
                if graph_client and graph_client.connect():
                    print("[OK] 图数据库连接成功")
                    graph_client.disconnect()
                else:
                    print("[FAIL] 图数据库连接失败")
            except Exception as e:
                print(f"[FAIL] 图数据库不可用: {e}")
        else:
            print("- 图数据库未启用")

        # 检查向量数据库
        vector_config = config.get('vector_database', {})
        if vector_config.get('enabled', False):
            try:
                from vector_database_client import get_vector_client
                vector_client = get_vector_client()
                if vector_client and vector_client.connect():
                    print("[OK] 向量数据库连接成功")
                    vector_client.disconnect()
                else:
                    print("[FAIL] 向量数据库连接失败")
            except Exception as e:
                print(f"[FAIL] 向量数据库不可用: {e}")
        else:
            print("- 向量数据库未启用")

        return True

    except Exception as e:
        print(f"[FAIL] 配置加载失败: {e}")
        return False

def run_database_demo():
    """运行数据库功能演示"""
    print("\n" + "=" * 60)
    print("数据库功能演示")
    print("=" * 60)

    try:
        from concept_graph import ConceptGraph
        from embedding_client import get_embedding_client
        from graph_database_client import get_graph_client
        from vector_database_client import get_vector_client, PoliticalTheoryVectorSearch

        # 读取种子概念
        with open('data/political_seed_concepts.txt', 'r', encoding='utf-8') as f:
            seed_concepts = [line.strip() for line in f.readlines()[:50]]  # 取前50个概念演示

        print(f"加载 {len(seed_concepts)} 个种子概念用于演示")

        # 初始化概念图（会自动连接到配置的数据库）
        print("\n1. 初始化概念图和数据库连接...")
        concept_graph = ConceptGraph(seed_concepts)

        # 获取数据库客户端
        graph_client = get_graph_client()
        vector_client = get_vector_client()
        embedding_client = get_embedding_client()

        print("\n2. 数据库初始化状态:")
        if graph_client:
            print("   [OK] 图数据库: 已连接")
        else:
            print("   [-] 图数据库: 未启用或连接失败")

        if vector_client:
            print("   [OK] 向量数据库: 已连接")
        else:
            print("   [-] 向量数据库: 未启用或连接失败")

        # 演示向量搜索
        if vector_client:
            print("\n3. 向量搜索演示:")
            vector_search = PoliticalTheoryVectorSearch(vector_client)

            # 搜索相似概念
            query_concept = "马克思主义"
            if query_concept in seed_concepts:
                query_embedding = embedding_client.encode([query_concept])[0]
                similar_concepts = vector_search.search_similar_concepts(
                    query_embedding, top_k=5
                )

                print(f"   查询概念: {query_concept}")
                print("   相似概念:")
                for i, concept in enumerate(similar_concepts[:3], 1):
                    name = concept.get('metadata', {}).get('name', '未知')
                    score = concept.get('score', 0)
                    print(f"   {i}. {name} (相似度: {score:.3f})")

        # 演示图数据库查询
        if graph_client:
            print("\n4. 图数据库查询演示:")
            try:
                # 获取第一个种子概念
                test_concept = seed_concepts[0]
                neighbors = graph_client.get_neighbors(test_concept, direction="both")
                print(f"   概念: {test_concept}")
                print(f"   邻居数量: {len(neighbors)}")

                # 搜索概念
                search_results = graph_client.search_nodes("主义", limit=5)
                print(f"   '主义'相关概念: {len(search_results)} 个")

                # 获取图统计
                stats = graph_client.get_graph_statistics()
                print(f"   图统计: {stats}")

            except Exception as e:
                print(f"   查询失败: {e}")

        print("\n5. 演示完成！")
        return True

    except Exception as e:
        print(f"[FAIL] 演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("MemCube 政治理论概念图扩增系统 - 数据库版")
    print("快速启动脚本")
    print("=" * 60)

    # 创建必要目录
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # 设置日志
    setup_logging()

    # 环境检查
    if not check_environment():
        print("\n[FAIL] 环境检查失败，请检查上述问题")
        return

    # 数据库检查
    if not check_database_availability():
        print("\n[WARN] 数据库连接有问题，但可以继续使用内存模式")

    # 询问用户是否运行演示
    print("\n" + "=" * 60)
    response = input("是否运行数据库功能演示？(y/n): ").lower().strip()

    if response in ['y', 'yes', '是']:
        if not run_database_demo():
            print("\n[FAIL] 演示运行失败")
            return

    # 使用选项
    print("\n" + "=" * 60)
    print("使用选项:")
    print("1. 完整概念图扩增 + QA生成")
    print("2. 仅概念图扩增")
    print("3. 仅QA生成")
    print("4. 退出")

    choice = input("请选择 (1-4): ").strip()

    try:
        if choice == "1":
            print("\n[INFO] 运行完整流程...")
            os.system("python -m src.main --stage all")
        elif choice == "2":
            print("\n[INFO] 运行概念图扩增...")
            os.system("python -m src.main --stage concept-expansion")
        elif choice == "3":
            print("\n[INFO] 运行QA生成...")
            os.system("python -m src.main --stage qa-generation")
        elif choice == "4":
            print("\n[INFO] 退出程序")
            return
        else:
            print("\n[FAIL] 无效选择")
    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断，退出程序")
    except Exception as e:
        print(f"\n[FAIL] 运行失败: {e}")

if __name__ == "__main__":
    main()