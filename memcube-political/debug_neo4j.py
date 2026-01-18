#!/usr/bin/env python3
"""
Neo4j连接调试脚本
"""
import time
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable, ConfigurationError

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neo4j_connection():
    """测试Neo4j连接"""

    # 连接配置
    uri = "bolt://35.212.244.212:7687"
    username = "neo4j"
    password = "MY_STRONG_PASSWORD"  # 请使用你的实际密码

    logger.info(f"尝试连接Neo4j: {uri}")
    logger.info(f"用户名: {username}")

    try:
        # 测试1: 基本连接
        logger.info("=== 测试1: 基本连接 ===")
        driver = GraphDatabase.driver(uri, auth=(username, password))

        # 测试2: 验证连接
        logger.info("=== 测试2: 验证连接 ===")
        driver.verify_connectivity()
        logger.info("✅ 连接验证成功")

        # 测试3: 简单查询
        logger.info("=== 测试3: 简单查询 ===")
        with driver.session() as session:
            result = session.run("RETURN 'Hello Neo4j' as message")
            record = result.single()
            logger.info(f"✅ 查询成功: {record['message']}")

        # 测试4: 获取数据库信息
        logger.info("=== 测试4: 获取数据库信息 ===")
        with driver.session() as session:
            result = session.run("CALL db.labels() YIELD label RETURN count(*) as count, label")
            labels = []
            for record in result:
                labels.append(f"{record['label']}: {record['count']}")

            if labels:
                logger.info(f"✅ 数据库标签: {', '.join(labels)}")
            else:
                logger.info("✅ 数据库为空或没有标签")

        # 测试5: 性能测试 - 批量写入测试
        logger.info("=== 测试5: 批量写入测试 ===")
        with driver.session() as session:
            # 先清理测试数据
            session.run("MATCH (n:TestNode) DELETE n")

            # 批量创建测试节点
            test_nodes = []
            for i in range(10):  # 只测试10个节点
                test_nodes.append({
                    'name': f'Test Node {i}',
                    'value': i * 10
                })

            start_time = time.time()

            # 批量创建
            result = session.run("""
                UNWIND $batch AS props
                CREATE (n:TestNode)
                SET n.name = props.name,
                    n.value = props.value,
                    n.created = timestamp()
                RETURN count(n) as created
            """, batch=test_nodes)

            end_time = time.time()
            created_count = result.single()['created']

            logger.info(f"✅ 批量创建成功: {created_count} 个节点")
            logger.info(f"✅ 耗时: {end_time - start_time:.2f} 秒")

        # 清理测试数据
        logger.info("=== 清理测试数据 ===")
        with driver.session() as session:
            result = session.run("MATCH (n:TestNode) DELETE n RETURN count(n) as deleted")
            deleted_count = result.single()['deleted']
            logger.info(f"✅ 删除测试数据: {deleted_count} 个节点")

        driver.close()
        logger.info("✅ 所有测试通过！")
        return True

    except AuthError as e:
        logger.error(f"❌ 认证错误: {e}")
        logger.error("请检查用户名和密码是否正确")
        return False

    except ServiceUnavailable as e:
        logger.error(f"❌ 服务不可用: {e}")
        logger.error("Neo4j服务可能未启动或网络连接问题")
        return False

    except ConfigurationError as e:
        logger.error(f"❌ 配置错误: {e}")
        return False

    except Exception as e:
        logger.error(f"❌ 其他错误: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        return False

def test_connection_pool():
    """测试连接池设置"""
    logger.info("=== 测试连接池配置 ===")

    uri = "bolt://35.212.244.212:7687"
    username = "neo4j"
    password = "MY_STRONG_PASSWORD"

    try:
        # 使用连接池配置
        driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_lifetime=30 * 60,  # 30分钟
            max_connection_pool_size=50,        # 最大50个连接
            connection_acquisition_timeout=60  # 60秒超时
        )

        logger.info("✅ 连接池配置成功")

        # 测试多个并发会话
        logger.info("测试并发会话...")
        sessions = []
        try:
            for i in range(5):
                session = driver.session()
                result = session.run("RETURN 'Session ' + $id as message", id=i)
                record = result.single()
                logger.info(f"  会话 {i}: {record['message']}")
                sessions.append(session)

        finally:
            for session in sessions:
                session.close()

        driver.close()
        logger.info("✅ 并发会话测试成功")

    except Exception as e:
        logger.error(f"❌ 连接池测试失败: {e}")

if __name__ == "__main__":
    print("Neo4j连接诊断工具")
    print("=" * 50)

    # 基本连接测试
    if test_neo4j_connection():
        print("\n✅ Neo4j连接正常，可以继续使用")

        # 连接池测试
        test_connection_pool()
    else:
        print("\n❌ Neo4j连接存在问题，请检查服务器状态")