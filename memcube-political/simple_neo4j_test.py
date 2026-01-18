#!/usr/bin/env python3
"""
简化的Neo4j测试
"""
import time
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_neo4j_batch_operations():
    """测试Neo4j批量操作"""

    uri = "bolt://35.212.244.212:7687"
    username = "neo4j"
    password = "MY_STRONG_PASSWORD"  # 使用实际密码

    try:
        # 优化的连接配置
        driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_lifetime=30 * 60,  # 30分钟
            max_connection_pool_size=20,        # 增加连接池
            connection_acquisition_timeout=60,  # 60秒超时
            max_transaction_retry_time=30     # 30秒重试
        )

        logger.info("Neo4j连接成功")

        with driver.session() as session:
            # 清理测试数据
            session.run("MATCH (n:BatchTest) DELETE n")

            # 测试不同大小的批量写入
            batch_sizes = [10, 25, 50]  # 测试不同批量大小

            for batch_size in batch_sizes:
                logger.info(f"测试批量大小: {batch_size}")

                # 创建测试数据
                test_data = []
                for i in range(batch_size):
                    test_data.append({
                        'name': f'BatchTest_Node_{batch_size}_{i}',
                        'batch_size': batch_size,
                        'value': i
                    })

                try:
                    start_time = time.time()

                    # 执行批量创建
                    result = session.run("""
                        UNWIND $batch AS props
                        CREATE (n:BatchTest)
                        SET n += props
                        RETURN count(n) as created
                    """, batch=test_data)

                    end_time = time.time()
                    created_count = result.single()['created']

                    logger.info(f"  批量大小 {batch_size}: 创建 {created_count} 个节点, 耗时: {end_time - start_time:.2f}s")

                    # 添加延迟避免过载
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"  批量大小 {batch_size} 失败: {e}")

            # 统计测试结果
            result = session.run("MATCH (n:BatchTest) RETURN count(n) as total")
            total_created = result.single()['total']
            logger.info(f"总共创建了 {total_created} 个测试节点")

            # 清理测试数据
            session.run("MATCH (n:BatchTest) DELETE n")
            logger.info("测试数据清理完成")

        driver.close()
        logger.info("测试完成")
        return True

    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

def check_neo4j_server_info():
    """检查Neo4j服务器信息"""

    uri = "bolt://35.212.244.212:7687"
    username = "neo4j"
    password = "MY_STRONG_PASSWORD"

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))

        with driver.session() as session:
            # 获取服务器信息
            result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions")
            components = list(result)

            # 获取数据库信息
            result = session.run("SHOW DATABASES YIELD name RETURN name")
            databases = [record["name"] for record in result]

            # 获取节点和关系数量
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]

            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result.single()["rel_count"]

            print("=== Neo4j服务器信息 ===")
            print(f"数据库: {databases}")
            print(f"节点数量: {node_count}")
            print(f"关系数量: {rel_count}")
            print("组件信息:")
            for record in components:
                print(f"  {record['name']}: {record['versions']}")

        driver.close()
        return True

    except Exception as e:
        print(f"获取服务器信息失败: {e}")
        return False

if __name__ == "__main__":
    print("Neo4j连接和性能测试")
    print("=" * 40)

    # 检查服务器信息
    print("\n1. 服务器信息检查:")
    check_neo4j_server_info()

    # 执行批量操作测试
    print("\n2. 批量操作测试:")
    if test_neo4j_batch_operations():
        print("\n测试通过 - Neo4j连接和批量操作正常")
    else:
        print("\n测试失败 - 请检查Neo4j配置")