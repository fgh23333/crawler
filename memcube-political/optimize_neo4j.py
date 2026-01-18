#!/usr/bin/env python3
"""
Neo4j优化建议脚本
"""
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_current_neo4j_config():
    """获取当前Neo4j配置建议"""

    config_recommendations = {
        "连接配置": {
            "uri": "bolt://35.212.244.212:7687",
            "max_connection_lifetime": 30 * 60,  # 30分钟
            "max_connection_pool_size": 20,      # 增加到20
            "connection_acquisition_timeout": 60,  # 60秒
            "max_transaction_retry_time": 30      # 30秒
        },

        "性能优化": {
            "batch_size": 50,                    # 减少到50，避免过大
            "batch_timeout": 60,                 # 60秒超时
            "retry_attempts": 3,                 # 重试3次
            "retry_delay": 1.0                   # 1秒延迟
        },

        "监控设置": {
            "enable_logs": True,
            "log_level": "INFO",
            "connection_timeout": 30             # 连接超时
        }
    }

    return config_recommendations

def create_optimized_neo4j_driver():
    """创建优化的Neo4j驱动"""

    config = get_current_neo4j_config()

    uri = "bolt://35.212.244.212:7687"
    username = "neo4j"
    password = "MY_STRONG_PASSWORD"  # 使用你的实际密码

    try:
        driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_lifetime=config["连接配置"]["max_connection_lifetime"],
            max_connection_pool_size=config["连接配置"]["max_connection_pool_size"],
            connection_acquisition_timeout=config["连接配置"]["connection_acquisition_timeout"],
            max_transaction_retry_time=config["性能优化"]["max_transaction_retry_time"]
        )

        logger.info("✅ 优化的Neo4j驱动创建成功")
        return driver

    except Exception as e:
        logger.error(f"❌ 创建Neo4j驱动失败: {e}")
        return None

def test_optimized_performance():
    """测试优化后的性能"""

    driver = create_optimized_neo4j_driver()
    if not driver:
        return False

    try:
        # 测试批量写入优化
        logger.info("=== 测试优化的批量写入 ===")

        with driver.session() as session:
            # 清理测试数据
            session.run("MATCH (n:TestOptimized) DELETE n")

            # 优化的批量写入
            batch_size = 50  # 优化的批量大小

            for batch_num in range(1, 4):  # 3个批次
                test_data = []
                for i in range(batch_size):
                    item_num = (batch_num - 1) * batch_size + i
                    test_data.append({
                        'name': f'Optimized Node {item_num}',
                        'batch': batch_num,
                        'value': item_num * 2
                    })

                logger.info(f"写入第 {batch_num} 批，包含 {len(test_data)} 个节点")

                import time
                start_time = time.time()

                result = session.run("""
                    UNWIND $batch AS props
                    CREATE (n:TestOptimized)
                    SET n += props
                    RETURN count(n) as created
                """, batch=test_data)

                end_time = time.time()
                created_count = result.single()['created']

                logger.info(f"✅ 第 {batch_num} 批完成: {created_count} 个节点，耗时: {end_time - start_time:.2f}s")

                # 添加批次间延迟
                time.sleep(0.5)  # 500ms延迟

        # 清理
        with driver.session() as session:
            result = session.run("MATCH (n:TestOptimized) DELETE n RETURN count(n) as deleted")
            deleted_count = result.single()['deleted']
            logger.info(f"✅ 清理完成: {deleted_count} 个节点")

        driver.close()
        return True

    except Exception as e:
        logger.error(f"❌ 性能测试失败: {e}")
        driver.close()
        return False

def print_recommendations():
    """打印优化建议"""

    config = get_current_neo4j_config()

    print("🔧 Neo4j优化建议:")
    print("=" * 50)

    print("\n1. 连接配置优化:")
    print(f"   - 连接池大小: {config['连接配置']['max_connection_pool_size']} (推荐20-50)")
    print(f"   - 连接生命周期: {config['连接配置']['max_connection_lifetime']}秒")
    print(f"   - 连接获取超时: {config['连接配置']['connection_acquisition_timeout']}秒")

    print("\n2. 批量操作优化:")
    print(f"   - 批量大小: {config['性能优化']['batch_size']} (推荐20-100)")
    print(f"   - 批量超时: {config['性能优化']['batch_timeout']}秒")
    print(f"   - 重试次数: {config['性能优化']['retry_attempts']}")

    print("\n3. 错误处理:")
    print("   - 启用连接重试机制")
    print("   - 设置合适的超时时间")
    print("   - 监控连接池状态")

    print("\n4. 服务器端检查:")
    print("   - 确保Neo4j有足够内存")
    print("   - 检查max_connections配置")
    print("   - 监控CPU和磁盘I/O")

    print("\n5. 代码优化:")
    print("   - 使用事务批量提交")
    print("   - 避免长时间运行的事务")
    print("   - 及时释放数据库连接")

if __name__ == "__main__":
    print("Neo4j优化和诊断工具")
    print("=" * 50)

    # 打印建议
    print_recommendations()

    # 测试优化后的配置
    print("\n🚀 测试优化配置...")
    if test_optimized_performance():
        print("\n✅ 优化配置测试通过！")
    else:
        print("\n❌ 优化配置测试失败，请检查配置")