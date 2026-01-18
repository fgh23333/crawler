#!/usr/bin/env python3
"""
单轮概念扩增脚本
"""

import sys
import os
from pathlib import Path

# 添加src到路径
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def single_round_expansion():
    """执行单轮概念扩增"""
    print("=" * 60)
    print("单轮概念扩增 + Neo4j保存")
    print("=" * 60)

    try:
        # 导入概念扩增器
        from concept_graph import ConceptExpander

        # 检查配置
        config_path = project_root / "config" / "config.yaml"
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return False

        print(f"📋 使用配置文件: {config_path}")

        # 创建概念扩增器
        print("\n🔧 初始化概念扩增器...")
        expander = ConceptExpander(str(config_path))

        # 测试连接
        print("🔍 测试数据库连接...")
        if not expander.test_connections():
            print("❌ 数据库连接测试失败")
            print("请检查:")
            print("1. Neo4j是否正在运行")
            print("2. Qdrant是否正在运行")
            print("3. API密钥是否正确配置")
            return False

        print("✅ 所有连接正常")

        # 检查配置
        concept_config = expander.config.get('concept_expansion', {})
        max_iterations = concept_config.get('max_iterations', 1)
        save_to_neo4j = concept_config.get('save_to_neo4j_after_each_iteration', False)

        print(f"\n⚙️ 扩增配置:")
        print(f"- 最大迭代次数: {max_iterations}")
        print(f"- 保存到Neo4j: {save_to_neo4j}")
        print(f"- 批处理大小: {concept_config.get('batch_size', '未设置')}")
        print(f"- 并发数: {concept_config.get('max_workers', '未设置')}")

        # 确认执行
        print("\n🚀 开始概念扩增...")
        print("这可能需要几分钟时间，请耐心等待...")

        # 执行扩增
        results = expander.run_full_expansion()

        if results:
            print(f"\n✅ 扩增完成！")
            print(f"📊 结果统计:")
            print(f"- 迭代次数: {len(results)}")

            if results:
                final_metrics = results[-1].get('metrics', {})
                print(f"- 总节点数: {final_metrics.get('nodes', 0)}")
                print(f"- 总边数: {final_metrics.get('edges', 0)}")
                print(f"- 平均度数: {final_metrics.get('avg_degree', 0):.2f}")

            print(f"\n💾 数据已保存到:")
            print(f"- Neo4j数据库 (如果启用)")
            print(f"- results/ 目录")
            print(f"- data/concept_graph/ 目录")

            return True
        else:
            print("❌ 扩增失败，没有产生结果")
            return False

    except Exception as e:
        print(f"❌ 执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    try:
        success = single_round_expansion()

        if success:
            print("\n🎉 单轮概念扩增成功完成！")
            print("\n下一步:")
            print("1. 查看Neo4j中的概念图谱")
            print("2. 运行python configure_expansion.py 调整配置")
            print("3. 运行更多轮次的扩增（如果需要）")
            return 0
        else:
            print("\n❌ 单轮概念扩增失败")
            print("\n请检查:")
            print("1. 配置文件是否正确")
            print("2. 数据库服务是否运行")
            print("3. API密钥是否配置")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
        return 1
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())