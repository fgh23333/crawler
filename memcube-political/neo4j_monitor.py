#!/usr/bin/env python3
"""
Neo4j监控和健康检查
"""
import time
import logging
import threading
from datetime import datetime
from neo4j import GraphDatabase, exceptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jMonitor:
    def __init__(self, uri, username, password):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.monitoring = False
        self.health_history = []

    def connect(self):
        """连接到Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=30 * 60,
                max_connection_pool_size=20,
                connection_acquisition_timeout=60
            )
            logger.info("Neo4j监控连接成功")
            return True
        except Exception as e:
            logger.error(f"Neo4j监控连接失败: {e}")
            return False

    def health_check(self):
        """健康检查"""
        if not self.driver:
            return False, "未连接"

        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
                return True, "健康"
        except exceptions.ServiceUnavailable:
            return False, "服务不可用"
        except exceptions.AuthError:
            return False, "认证失败"
        except Exception as e:
            return False, f"未知错误: {str(e)}"

    def start_monitoring(self, interval=30):
        """开始监控"""
        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                try:
                    is_healthy, status = self.health_check()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    health_record = {
                        'timestamp': timestamp,
                        'healthy': is_healthy,
                        'status': status
                    }

                    self.health_history.append(health_record)

                    # 只保留最近100条记录
                    if len(self.health_history) > 100:
                        self.health_history = self.health_history[-100:]

                    if is_healthy:
                        logger.info(f"Neo4j健康检查通过: {timestamp}")
                    else:
                        logger.error(f"Neo4j健康检查失败: {status} - {timestamp}")

                except Exception as e:
                    logger.error(f"健康检查异常: {e}")

                time.sleep(interval)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"Neo4j监控启动，检查间隔: {interval}秒")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False

    def disconnect(self):
        """断开连接"""
        if self.driver:
            self.driver.close()
            self.driver = None

    def get_health_summary(self):
        """获取健康摘要"""
        if not self.health_history:
            return "暂无监控数据"

        recent_checks = self.health_history[-10:]  # 最近10次检查
        healthy_count = sum(1 for check in recent_checks if check['healthy'])
        total_count = len(recent_checks)

        if total_count == 0:
            return "暂无监控数据"

        health_rate = healthy_count / total_count * 100
        return f"最近{total_count}次检查，健康率: {health_rate:.1f}%"

if __name__ == "__main__":
    # 配置
    uri = "bolt://35.212.244.212:7687"
    username = "neo4j"
    password = "MY_STRONG_PASSWORD"

    # 创建监控器
    monitor = Neo4jMonitor(uri, username, password)

    try:
        if monitor.connect():
            print("Neo4j监控器启动成功")

            # 启动健康检查监控
            monitor.start_monitoring(interval=10)  # 每10秒检查一次

            # 运行2分钟的监控
            time.sleep(120)

            # 停止监控
            monitor.stop_monitoring()

            # 显示健康摘要
            print(f"健康摘要: {monitor.get_health_summary()}")

            # 显示最近的一些检查结果
            print("最近的健康检查记录:")
            for i, record in enumerate(monitor.health_history[-5:]):
                status_symbol = "✅" if record['healthy'] else "❌"
                print(f"  {status_symbol} {record['timestamp']}: {record['status']}")
        else:
            print("Neo4j监控器启动失败")

    finally:
        monitor.disconnect()